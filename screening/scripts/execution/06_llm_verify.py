#!/usr/bin/env python3
"""
06_llm_verify.py
================
LLM agent verifies each candidate's Grotthuss proton relay capability
and structural suitability using Gemini or Claude via Vertex AI.

Changes from original:
  - ALL rule-accepted molecules are sent to LLM (no 200-molecule sampling)
  - Supports Vertex AI service account authentication
  - Supports model selection via --model argument
  - Cleaner logging and error handling

Pipeline position:
  After : 05_predict_pa.py  (molecular_pa.parquet exists)
  Before: 07_pareto_select.py

Reads from:
    data/screening/iter{N}/molecular_pa.parquet

Writes to:
    data/screening/iter{N}/llm_verdicts.parquet

Usage:
    # Gemini via API key (original method):
    python screening/scripts/06_llm_verify.py --iter 1

    # Vertex AI with service account (recommended):
    python screening/scripts/06_llm_verify.py --iter 1 \\
        --model vertex_ai/gemini-2.0-pro-preview \\
        --vertex-key /path/to/service-account.json

    # Dry run (first 20 molecules only):
    python screening/scripts/06_llm_verify.py --iter 1 --dry-run \\
        --model vertex_ai/gemini-2.0-pro-preview \\
        --vertex-key /path/to/service-account.json

    # Skip LLM entirely (rule-based only):
    python screening/scripts/06_llm_verify.py --iter 1 --skip-llm
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = Path(__file__).parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_DIR   = PROJECT / "data" / "screening"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def load_gemini_api_key() -> str:
    """Load Gemini API key from api_keys.txt or environment."""
    key_file = SCREENING.parent / "language modeling" / "api_keys.txt"
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "GEMINI_API_KEY" in line or "Vertex_AI_API" in line:
                return line.split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("GEMINI_API_KEY", "")


def load_vertex_key(path: str) -> dict:
    """Load Vertex AI service account JSON key."""
    with open(path) as f:
        sa_key = json.load(f)
    if "project_id" not in sa_key:
        log.error("'project_id' not found in service account key file.")
        sys.exit(1)
    return sa_key


# ---------------------------------------------------------------------------
# Functional group classification
# ---------------------------------------------------------------------------

SMARTS_MAP = {
    "nitrile":         "[NX1]#[CX2]",
    "amide":           "[NX3][CX3](=[OX1])",
    "aromatic_N":      "n",
    "ether":           "[OX2]([CX4])[CX4]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carbonyl":        "[CX3]=[OX1]",
    "primary_amine":   "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "tertiary_amine":  "[NX3;H0;!$(NC=O);!$([n])]",
}
SMARTS_COMPILED = {k: Chem.MolFromSmarts(v) for k, v in SMARTS_MAP.items()}


def classify_functional_group(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"
    for name, pat in SMARTS_COMPILED.items():
        if pat and mol.HasSubstructMatch(pat):
            return name
    return "other"


# ---------------------------------------------------------------------------
# Known failure mode SMARTS
# ---------------------------------------------------------------------------

FAILURE_SMARTS = {
    "cumulated_double_bonds": Chem.MolFromSmarts("[CX2]=[CX2]=[*]"),
    "isocyanate":             Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]"),
    "ketene":                 Chem.MolFromSmarts("[CX2]=[CX2]=[OX1]"),
    "formal_charge":          Chem.MolFromSmarts("[+1,+2,-1,-2]"),
    "nitro_group":            Chem.MolFromSmarts("[NX3](=O)=O"),
}

PA_TRAIN_MAX   = 280.0   # kcal/mol — k-means training maximum
UNCERTAINTY_MAX = 20.0   # kcal/mol — flag threshold


# ---------------------------------------------------------------------------
# Rule-based pre-screen
# ---------------------------------------------------------------------------

def rule_based_check(row: pd.Series) -> dict:
    """
    Fast checks before LLM. Returns verdict and any flags.
    Only hard structural failures get 'reject' — everything else
    gets 'accept' or 'flag' and will be forwarded to the LLM.
    """
    smiles      = row["smiles"]
    pa_pred     = row["pa_pred_kcalmol"]
    delta_pred  = row["delta_pred"]
    uncertainty = row["uncertainty"]
    fg          = classify_functional_group(smiles)
    mol         = Chem.MolFromSmiles(smiles)

    flags   = []
    verdict = "accept"

    # Hard reject: known failure mode substructures
    if mol:
        for name, pat in FAILURE_SMARTS.items():
            if pat and mol.HasSubstructMatch(pat):
                flags.append(f"failure_mode:{name}")
                verdict = "reject"
                break

    # Soft flags (LLM will still evaluate these)
    if pa_pred > PA_TRAIN_MAX + 20:
        flags.append(f"pa_above_domain:{pa_pred:.0f}")
        if verdict == "accept":
            verdict = "flag"

    if uncertainty > UNCERTAINTY_MAX:
        flags.append(f"high_uncertainty:{uncertainty:.1f}")
        if verdict == "accept":
            verdict = "flag"

    # Flag extreme correction outliers (>4 sigma from training mean 17.7)
    z = abs(delta_pred - 17.7) / 12.5
    if z > 4.0:
        flags.append(f"extreme_correction:{delta_pred:.1f}kcal({z:.1f}sigma)")
        if verdict == "accept":
            verdict = "flag"

    return {
        "rule_verdict":     verdict,
        "functional_group": fg,
        "rule_flags":       "; ".join(flags) if flags else "none",
        "n_flags":          len(flags),
    }


# ---------------------------------------------------------------------------
# LLM system prompt and few-shot examples
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are assessing whether a small molecule can function as a proton carrier \
via the Grotthuss mechanism in anhydrous polymer electrolyte fuel cells.

## Background

The Grotthuss mechanism requires a molecule to act as BOTH proton donor and \
acceptor in sequence. For this to work, the molecule must have:
  1. A basic nitrogen lone pair that accepts a proton (N: + H+ -> NH+)
  2. An N-H bond that can donate a proton to the next molecule (NH+ -> N: + H+)
  3. These two functions must be chemically accessible on an aromatic \
N-heterocycle where protonated and neutral forms are both stable.

Known good carriers: imidazole, benzimidazole, pyrazole, 1,2,4-triazole.
Poor carriers: nitriles (N lone pair in triple bond, no N-H), carbonyl oxygens.

## Your two tasks

1. GROTTHUSS ASSESSMENT: Does this molecule have both a lone-pair nitrogen \
acceptor AND an N-H donor for proton relay? Is the nitrogen aromatic? Is the \
N-H on an aromatic ring? Can the molecule act in both roles without structural \
rearrangement?

2. STRUCTURAL NOVELTY CONCERN: Does the molecule contain unusual structural \
features that suggest the ML model may be unreliable? Examples: highly strained \
rings, unusual heteroatom combinations (P, S alongside N), zwitterionic character, \
allene-like cumulated double bonds, scaffolds very different from N-heterocycles.

Note: Numerical checks (PA range, uncertainty, correction magnitude) have already \
been done by a rule-based filter. Do not repeat them.

## Output schema — respond ONLY with valid JSON, no other text, start with {

{
  "grotthuss_capable": true | false,
  "grotthuss_reasoning": "one sentence: which atoms serve as donor/acceptor, \
or why the mechanism is not feasible",
  "structural_concern": null | "one sentence describing the unusual feature",
  "verdict": "accept" | "flag" | "reject"
}

Verdict rules:
- "accept":  grotthuss_capable=true AND no structural concern
- "flag":    grotthuss_capable=true BUT has structural concern, OR borderline
- "reject":  grotthuss_capable=false

## Few-shot examples

Input: SMILES=c1cn[nH]c1
Output: {"grotthuss_capable": true, "grotthuss_reasoning": "Pyridine-like N \
(position 2) accepts proton; pyrrole-like N-H (position 1) donates proton; \
both on aromatic 5-membered ring enabling fast reorientation.", \
"structural_concern": null, "verdict": "accept"}cle

Input: SMILES=N#Cc1ccc(N)cc1
Output: {"grotthuss_capable": false, "grotthuss_reasoning": "Nitrile N lone \
pair is part of the C-triple-N bond and too weakly basic to accept a proton; \
no N-H available for donation from the nitrile group.", \
"structural_concern": null, "verdict": "reject"}

Input: SMILES=CP(C)(=O)c1cc[nH]c(=N)n1
Output: {"grotthuss_capable": true, "grotthuss_reasoning": "Imidazole-like \
ring provides N: acceptor and N-H donor sites for Grotthuss relay.", \
"structural_concern": "Phosphonamide group P(C)(=O) is unusual relative to \
N-heterocycle training data; model reliability uncertain.", "verdict": "flag"}\
"""


# ---------------------------------------------------------------------------
# LLM query (single molecule)
# ---------------------------------------------------------------------------

def query_llm(row: dict, model: str,
              sa_key: dict | None,
              project_id: str | None,
              api_key: str | None) -> dict:
    """Query LLM for one molecule. Returns parsed response dict."""
    import litellm
    litellm.set_verbose = False

    prompt = (
        f"Assess this candidate molecule:\n\n"
        f"SMILES: {row['smiles']}\n"
        f"Molecular weight: {row.get('MW', 'N/A')} Da\n"
        f"Functional group class: {row['functional_group']}\n"
        f"PA_pred (ML corrected): {row['pa_pred_kcalmol']:.1f} kcal/mol\n"
        f"Number of protonation sites: {row.get('n_sites', 'N/A')}\n\n"
        f"Respond ONLY with valid JSON starting with {{"
    )

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=512,
    )

    # Authentication
    if sa_key is not None:
        # Vertex AI service account
        kwargs["vertex_project"]     = project_id
        kwargs["vertex_location"]    = "us-east5"
        kwargs["vertex_credentials"] = sa_key
    elif api_key:
        # Gemini API key
        os.environ["GEMINI_API_KEY"] = api_key

    for attempt in range(3):
        try:
            resp = litellm.completion(**kwargs)
            text = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "llm_verdict":           parsed.get("verdict", "flag"),
                    "grotthuss_capable":     parsed.get("grotthuss_capable"),
                    "grotthuss_reasoning":   parsed.get("grotthuss_reasoning", ""),
                    "structural_concern":    parsed.get("structural_concern"),
                    "llm_error":             None,
                }
            return {
                "llm_verdict":         "flag",
                "grotthuss_capable":   None,
                "grotthuss_reasoning": text[:200],
                "structural_concern":  "Could not parse JSON response",
                "llm_error":           "no_json",
            }
        except Exception as e:
            if attempt == 2:
                return {
                    "llm_verdict":         "flag",
                    "grotthuss_capable":   None,
                    "grotthuss_reasoning": f"API error: {str(e)[:100]}",
                    "structural_concern":  None,
                    "llm_error":           str(e)[:100],
                }
            time.sleep(2 ** attempt)

    return {
        "llm_verdict":       "flag",
        "grotthuss_capable": None,
        "grotthuss_reasoning": "max retries exceeded",
        "structural_concern":  None,
        "llm_error":           "max_retries",
    }


# ---------------------------------------------------------------------------
# Combined verdict
# ---------------------------------------------------------------------------

def combined_verdict(row: pd.Series) -> str:
    """
    Final verdict combining rule-based and LLM assessments.
    Rule reject always wins. LLM reject wins over rule accept.
    Flag from either source gives flag.
    """
    if row["rule_verdict"] == "reject":
        return "reject"
    llm = row.get("llm_verdict", "accept")
    if llm == "reject":
        return "reject"
    if row["rule_verdict"] == "flag" or llm == "flag":
        return "flag"
    return "accept"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(iteration: int,
         model: str,
         vertex_key_path: str | None,
         dry_run: bool,
         skip_llm: bool,
         delay: float) -> None:

    iter_dir = DATA_DIR / f"iter{iteration}"
    mol_path = iter_dir / "molecular_pa.parquet"
    out_path = iter_dir / "llm_verdicts.parquet"

    if not mol_path.exists():
        log.error(f"molecular_pa.parquet not found: {mol_path}")
        sys.exit(1)

    mol_df = pd.read_parquet(mol_path)
    log.info(f"Loaded {len(mol_df):,} molecules from {mol_path}")

    if dry_run:
        mol_df = mol_df.head(20)
        log.warning(f"DRY RUN — processing first {len(mol_df)} molecules only")

    # ── Step 1: Rule-based pre-screen (all molecules, fast) ───────────────
    log.info("Running rule-based pre-screen...")
    rule_results = [rule_based_check(row) for _, row in mol_df.iterrows()]
    rule_df      = pd.DataFrame(rule_results, index=mol_df.index)
    mol_df       = pd.concat([mol_df, rule_df], axis=1)

    n_rule_accept = (rule_df["rule_verdict"] == "accept").sum()
    n_rule_flag   = (rule_df["rule_verdict"] == "flag").sum()
    n_rule_reject = (rule_df["rule_verdict"] == "reject").sum()
    log.info(f"  Rule-based: {n_rule_accept} accept | "
             f"{n_rule_flag} flag | {n_rule_reject} hard reject")

    # ── Step 2: LLM verification (everything except hard rejects) ─────────
    if skip_llm:
        log.warning("Skipping LLM (--skip-llm). Using rule verdicts directly.")
        mol_df["llm_verdict"]         = mol_df["rule_verdict"]
        mol_df["grotthuss_capable"]   = None
        mol_df["grotthuss_reasoning"] = "LLM skipped"
        mol_df["structural_concern"]  = None
        mol_df["llm_error"]           = None

    else:
        # Load authentication
        sa_key     = None
        project_id = None
        api_key    = None

        if vertex_key_path:
            sa_key     = load_vertex_key(vertex_key_path)
            project_id = sa_key["project_id"]
            log.info(f"Using Vertex AI: project={project_id}, model={model}")
        else:
            api_key = load_gemini_api_key()
            if not api_key:
                log.error("No API key found. Use --vertex-key or set GEMINI_API_KEY.")
                sys.exit(1)
            log.info(f"Using Gemini API key, model={model}")

        # Send ALL non-rejected molecules to LLM
        # This is the key fix: no sampling — every molecule gets evaluated
        to_llm_mask  = mol_df["rule_verdict"] != "reject"
        to_llm       = mol_df[to_llm_mask].copy()
        hard_rejects = mol_df[~to_llm_mask].copy()

        n_to_llm = len(to_llm)
        est_min  = n_to_llm * delay / 60
        log.info(f"Sending {n_to_llm:,} molecules to LLM "
                 f"(~{est_min:.0f} min at {delay}s/call)...")
        log.info(f"  ({n_rule_reject} hard rejects skipped)")

        llm_rows    = to_llm.to_dict("records")
        llm_results = []

        for i, row in enumerate(llm_rows):
            if (i + 1) % 100 == 0 or i == 0:
                log.info(f"  [{i+1:4d}/{n_to_llm}] {row['smiles'][:50]}")

            result = query_llm(
                row, model, sa_key, project_id, api_key)
            llm_results.append(result)

            if result["llm_error"]:
                log.warning(f"  Error [{i+1}]: {result['llm_error']}")

            time.sleep(delay)

        llm_df = pd.DataFrame(llm_results, index=to_llm.index)
        to_llm = pd.concat([to_llm, llm_df], axis=1)

        # Hard rejects get null LLM fields
        hard_rejects = hard_rejects.copy()
        hard_rejects["llm_verdict"]         = "reject"
        hard_rejects["grotthuss_capable"]   = None
        hard_rejects["grotthuss_reasoning"] = "Hard reject — rule-based only"
        hard_rejects["structural_concern"]  = None
        hard_rejects["llm_error"]           = None

        mol_df = pd.concat([to_llm, hard_rejects]).sort_index()

    # ── Step 3: Combined verdict ───────────────────────────────────────────
    mol_df["final_verdict"] = mol_df.apply(combined_verdict, axis=1)

    # ── Save ──────────────────────────────────────────────────────────────
    mol_df.to_parquet(out_path, index=False)
    log.info(f"Saved verdicts -> {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    vc       = mol_df["final_verdict"].value_counts()
    accepted = mol_df[mol_df["final_verdict"] == "accept"]
    flagged  = mol_df[mol_df["final_verdict"] == "flag"]
    in_window = (
        (accepted["pa_pred_kcalmol"] >= 210) &
        (accepted["pa_pred_kcalmol"] <= 235)
    )
    errors = mol_df.get("llm_error", pd.Series()).notna().sum()

    log.info(f"\n{'='*50}")
    log.info(f"  LLM Verification — Iteration {iteration}")
    log.info(f"{'='*50}")
    log.info(f"  Final verdicts:")
    log.info(f"    Accept : {vc.get('accept', 0):,}")
    log.info(f"    Flag   : {vc.get('flag', 0):,}")
    log.info(f"    Reject : {vc.get('reject', 0):,}")
    log.info(f"  LLM errors/unparseable: {errors}")
    log.info(f"  Accepted in 210-235 window: {in_window.sum():,}")
    log.info(f"  Mean uncertainty (accepted): "
             f"{accepted['uncertainty'].mean():.1f} kcal/mol")

    if "grotthuss_capable" in mol_df.columns:
        gc = mol_df["grotthuss_capable"].value_counts(dropna=False)
        log.info(f"  Grotthuss-capable (LLM=True):  "
                 f"{gc.get(True, 0):,}")
        log.info(f"  Not capable (LLM=False):       "
                 f"{gc.get(False, 0):,}")
        log.info(f"  Not evaluated (LLM=None):      "
                 f"{gc.get(None, gc.get(pd.NA, 0)):,}")

    log.info(f"{'='*50}\n")
    log.info("Next step: python screening/scripts/07_pareto_select.py "
             f"--iter {iteration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Grotthuss verification of PA-screened candidates."
    )
    parser.add_argument(
        "--iter", type=int, default=1,
        help="Screening iteration number (default: 1)")
    parser.add_argument(
        "--model", default="gemini/gemini-2.5-flash",
        help="LiteLLM model string (default: gemini/gemini-2.5-flash). "
             "For Vertex AI use e.g. vertex_ai/gemini-2.0-pro-preview")
    parser.add_argument(
        "--vertex-key", default=None, metavar="PATH",
        help="Path to Vertex AI service account JSON key file. "
             "If not provided, uses GEMINI_API_KEY env var or api_keys.txt")
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between API calls (default: 2.0)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process first 20 molecules only")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM entirely, use rule-based verdicts only")
    args = parser.parse_args()

    main(
        iteration      = args.iter,
        model          = args.model,
        vertex_key_path= args.vertex_key,
        dry_run        = args.dry_run,
        skip_llm       = args.skip_llm,
        delay          = args.delay,
    )