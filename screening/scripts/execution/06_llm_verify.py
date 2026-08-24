#!/usr/bin/env python3
"""
06_llm_verify.py
================
Evaluate each non-rejected candidate for the stated proton-relay criterion
and structural-domain concerns. The paper run used Claude Opus 4.6 through
Vertex AI and retained a raw, versioned response record for every query.

Pipeline position:
  After : 05_predict_pa.py  (molecular_pa.parquet exists)
  Before: 07_pareto_select.py

Reads from:
    data/screening/iter{N}/molecular_pa.parquet

Writes to:
    data/screening/iter{N}/llm_verdicts.parquet

Production run (paper): Claude Opus 4.6 via Vertex AI, temperature=0, with all
non-hard-rejected candidates queried and raw responses checkpointed as JSONL.

Usage:
    # Claude Opus 4.6 via Vertex AI (used for paper):
    python screening/scripts/06_llm_verify.py --iter 1 \\
        --model vertex_ai/claude-opus-4-6@default \\
        --vertex-key /path/to/service-account.json

    # Dry run (first 20 molecules only):
    python screening/scripts/06_llm_verify.py --iter 1 --dry-run \\
        --model vertex_ai/claude-opus-4-6@default \\
        --vertex-key /path/to/service-account.json

    # Skip LLM entirely (rule-based only):
    python screening/scripts/06_llm_verify.py --iter 1 --skip-llm
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from pipeline_config import (
    DEFAULT_CONFIG_PATH,
    load_pipeline_config,
    load_training_reference_stats,
)

RDLogger.DisableLog('rdApp.*')

SCRIPT_DIR = Path(__file__).resolve().parent
SCREENING  = SCRIPT_DIR.parent.parent
PROJECT    = SCREENING.parent
DATA_ROOT  = next((p for p in (PROJECT / "data", PROJECT.parent / "data") if p.exists()), PROJECT / "data")
DATA_DIR   = DATA_ROOT / "screening"
CONFIG     = load_pipeline_config()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def load_gemini_api_key() -> str:
    """Return a Gemini key from the environment without reading local files."""
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


def grotthuss_motif_check(smiles: str) -> dict:
    """Deterministically test the manuscript's same-ring aromatic N criterion."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "motif_capable": False,
            "motif_reason": "invalid SMILES",
            "motif_ring_atoms": None,
        }

    for ring in mol.GetRingInfo().AtomRings():
        donors = []
        acceptors = []
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 7 or not atom.GetIsAromatic():
                continue
            if atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() <= 0:
                donors.append(idx)
            elif atom.GetTotalNumHs() == 0 and atom.GetFormalCharge() == 0:
                acceptors.append(idx)
        if donors and acceptors:
            return {
                "motif_capable": True,
                "motif_reason": (
                    "same aromatic ring contains N-H donor atom(s) "
                    f"{donors} and lone-pair N acceptor atom(s) {acceptors}"
                ),
                "motif_ring_atoms": list(ring),
            }

    return {
        "motif_capable": False,
        "motif_reason": (
            "no aromatic ring contains both an N-H donor and a distinct "
            "neutral aromatic N acceptor"
        ),
        "motif_ring_atoms": None,
    }


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

# ---------------------------------------------------------------------------
# Rule-based pre-screen
# ---------------------------------------------------------------------------

def rule_based_check(
    row: pd.Series,
    training_stats: dict[str, float],
    filter_config: dict,
) -> dict:
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
    motif       = grotthuss_motif_check(smiles)

    flags   = []
    verdict = "accept"

    if mol is None:
        flags.append("failure_mode:invalid_smiles")
        verdict = "reject"
    else:
        for name, pat in FAILURE_SMARTS.items():
            if pat and mol.HasSubstructMatch(pat):
                flags.append(f"failure_mode:{name}")
                verdict = "reject"
                break

    # Soft flags (LLM will still evaluate these)
    pa_domain_limit = (
        training_stats["pa_train_max_kcalmol"]
        + filter_config["pa_domain_buffer_kcalmol"]
    )
    if pa_pred > pa_domain_limit:
        flags.append(f"pa_above_domain:{pa_pred:.0f}")
        if verdict == "accept":
            verdict = "flag"

    if uncertainty > filter_config["uncertainty_max_kcalmol"]:
        flags.append(f"high_uncertainty:{uncertainty:.1f}")
        if verdict == "accept":
            verdict = "flag"

    # Flag extreme correction outliers relative to corrected training targets.
    z = abs(
        delta_pred - training_stats["correction_mean_kcalmol"]
    ) / training_stats["correction_std_kcalmol"]
    if z > filter_config["correction_z_max"]:
        flags.append(f"extreme_correction:{delta_pred:.1f}kcal({z:.1f}sigma)")
        if verdict == "accept":
            verdict = "flag"

    return {
        "rule_verdict":     verdict,
        "functional_group": fg,
        "rule_flags":       "; ".join(flags) if flags else "none",
        "n_flags":          len(flags),
        "motif_capable":    motif["motif_capable"],
        "motif_reason":     motif["motif_reason"],
        "motif_ring_atoms": motif["motif_ring_atoms"],
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

Note: Numerical checks and a deterministic same-ring aromatic N-H/N motif check \
have already been done by a rule-based filter. Do not repeat numerical checks; \
independently assess the chemistry from the supplied SMILES.

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

Input: SMILES=c1ncc[nH]1
Output: {"grotthuss_capable": true, "grotthuss_reasoning": "Pyridine-like N \
(position 2) accepts proton; pyrrole-like N-H (position 1) donates proton; \
both on aromatic 5-membered ring enabling fast reorientation.", \
"structural_concern": null, "verdict": "accept"}

Input: SMILES=N#Cc1ccc(N)cc1
Output: {"grotthuss_capable": false, "grotthuss_reasoning": "Nitrile N lone \
pair is part of the C-triple-N bond and too weakly basic; the exocyclic amino \
N-H is not an aromatic heterocyclic N-H relay site.", \
"structural_concern": null, "verdict": "reject"}

Input: SMILES=CP(C)(=O)c1cc[nH]c(=N)n1
Output: {"grotthuss_capable": true, "grotthuss_reasoning": "Imidazole-like \
ring provides N: acceptor and N-H donor sites for Grotthuss relay.", \
"structural_concern": "Phosphine-oxide substituent P(C)(=O) is unusual relative \
to N-heterocycle training data; model reliability uncertain.", "verdict": "flag"}\
"""


# ---------------------------------------------------------------------------
# LLM query (single molecule)
# ---------------------------------------------------------------------------

def validate_llm_payload(parsed: dict) -> None:
    """Reject malformed or internally inconsistent model decisions."""
    required = {
        "grotthuss_capable",
        "grotthuss_reasoning",
        "structural_concern",
        "verdict",
    }
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"missing JSON fields: {sorted(missing)}")
    if type(parsed["grotthuss_capable"]) is not bool:
        raise ValueError("grotthuss_capable must be a JSON boolean")
    if not isinstance(parsed["grotthuss_reasoning"], str) or not parsed["grotthuss_reasoning"].strip():
        raise ValueError("grotthuss_reasoning must be a non-empty string")
    concern = parsed["structural_concern"]
    if concern is not None and not isinstance(concern, str):
        raise ValueError("structural_concern must be null or a string")
    verdict = parsed["verdict"]
    if verdict not in {"accept", "flag", "reject"}:
        raise ValueError(f"invalid verdict: {verdict!r}")
    if not parsed["grotthuss_capable"] and verdict != "reject":
        raise ValueError("non-capable molecule must be rejected")
    if parsed["grotthuss_capable"] and verdict == "reject":
        raise ValueError("capable molecule cannot have reject verdict")
    if parsed["grotthuss_capable"] and concern and verdict != "flag":
        raise ValueError("capable molecule with structural concern must be flagged")


def parse_llm_json(text: str) -> dict:
    """Decode the first JSON object, tolerating fences or trailing commentary."""
    start = text.find("{")
    if start < 0:
        raise ValueError("response contained no JSON object")
    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def build_user_prompt(row: dict) -> str:
    """Build the deterministic per-molecule prompt used for checkpoint identity."""
    return (
        f"Assess this candidate molecule:\n\n"
        f"SMILES: {row['smiles']}\n"
        f"Molecular weight: {row.get('MW', 'N/A')} Da\n"
        f"Functional group class: {row['functional_group']}\n"
        f"PA_pred (ML corrected): {row['pa_pred_kcalmol']:.1f} kcal/mol\n"
        f"Number of protonation sites: {row.get('n_sites', 'N/A')}\n\n"
        f"Respond ONLY with valid JSON starting with {{"
    )


def query_llm(row: dict, model: str,
              sa_key: dict | None,
              project_id: str | None,
              api_key: str | None,
              llm_config: dict) -> dict:
    """Query LLM for one molecule. Returns parsed response dict with full metadata."""
    import litellm
    litellm.set_verbose = False

    prompt = build_user_prompt(row)

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(
        model=model,
        messages=base_messages,
        max_tokens=llm_config["max_tokens"],
        temperature=llm_config["temperature"],
    )

    # Authentication
    if model.startswith("vertex_ai/"):
        # LiteLLM uses the supplied key or Google application-default
        # credentials. The project can also be supplied through the standard
        # GOOGLE_CLOUD_PROJECT or VERTEXAI_PROJECT environment variable.
        if project_id:
            kwargs["vertex_project"] = project_id
        kwargs["vertex_location"] = llm_config["vertex_location"]
        if sa_key is not None:
            kwargs["vertex_credentials"] = sa_key
    elif api_key:
        # Gemini API key
        os.environ["GEMINI_API_KEY"] = api_key

    base_record = {
        "smiles":           row["smiles"],
        "mol_id":           row.get("mol_id"),
        "prompt":           prompt,
        "model_requested":  model,
    }

    last_text = None
    last_meta = {}

    for attempt in range(llm_config["max_attempts"]):
        t0 = time.time()
        attempt_text = None
        try:
            resp = litellm.completion(**kwargs)
            latency_s = time.time() - t0
            text = resp.choices[0].message.content.strip()
            attempt_text = text
            last_text = text

            usage = getattr(resp, "usage", None)
            token_info = {}
            if usage:
                token_info = {
                    "prompt_tokens":     getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens":      getattr(usage, "total_tokens", None),
                }

            meta = {
                "response_id":       getattr(resp, "id", None),
                "model_returned":    getattr(resp, "model", None),
                "created":           getattr(resp, "created", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "finish_reason":     resp.choices[0].finish_reason if resp.choices else None,
                "latency_s":         round(latency_s, 3),
                "attempt":           attempt + 1,
                **token_info,
            }
            last_meta = meta

            parsed = parse_llm_json(text)
            validate_llm_payload(parsed)
            return {
                "llm_verdict":           parsed["verdict"],
                "grotthuss_capable":     parsed["grotthuss_capable"],
                "grotthuss_reasoning":   parsed["grotthuss_reasoning"],
                "structural_concern":    parsed["structural_concern"],
                "llm_error":             None,
                "raw_response":          text,
                "parsed_json":           parsed,
                **base_record,
                **meta,
            }
        except Exception as e:
            latency_s = time.time() - t0
            if attempt == llm_config["max_attempts"] - 1:
                return {
                    **base_record,
                    **last_meta,
                    "llm_verdict":         "flag",
                    "grotthuss_capable":   None,
                    "grotthuss_reasoning": f"API error: {str(e)[:500]}",
                    "structural_concern":  None,
                    "llm_error":           str(e)[:500],
                    "raw_response":        last_text,
                    "parsed_json":         None,
                    "latency_s":           round(latency_s, 3),
                    "attempt":             attempt + 1,
                }
            if attempt_text is not None:
                kwargs["messages"] = base_messages + [
                    {"role": "assistant", "content": attempt_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed validation: "
                            f"{str(e)[:300]}. Return one corrected JSON object only. "
                            "Keep the chemical assessment unchanged unless changing it "
                            "is necessary to make grotthuss_capable, structural_concern, "
                            "and verdict logically consistent with the stated rules."
                        ),
                    },
                ]
            time.sleep(2 ** attempt)

    return {
        "llm_verdict":       "flag",
        "grotthuss_capable": None,
        "grotthuss_reasoning": "max retries exceeded",
        "structural_concern":  None,
        "llm_error":           "max_retries",
        "raw_response":        None,
        "parsed_json":         None,
        **base_record,
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
    if row["rule_verdict"] == "reject" or not bool(row.get("motif_capable", False)):
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
         delay: float,
         resume_jsonl: str | None = None) -> None:

    filter_config = CONFIG["rule_filter"]
    llm_config = CONFIG["llm"]
    pa_low = CONFIG["pa_window_kcalmol"]["low"]
    pa_high = CONFIG["pa_window_kcalmol"]["high"]
    training_stats = load_training_reference_stats()

    iter_dir = DATA_DIR / f"iter{iteration}"
    mol_path = iter_dir / "molecular_pa.parquet"
    out_name = "llm_verdicts_dry_run.parquet" if dry_run else "llm_verdicts.parquet"
    out_path = iter_dir / out_name
    log_dir  = PROJECT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_file   = log_dir / f"llm_verify_{run_ts}.log"
    jsonl_file = (
        Path(resume_jsonl).resolve()
        if resume_jsonl
        else iter_dir / f"llm_raw_responses_{run_ts}.jsonl"
    )
    manifest_file = iter_dir / f"llm_run_config_{run_ts}.json"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    # The module logger propagates to root, so attach once to avoid duplicate
    # records in the audit log.
    logging.getLogger().addHandler(fh)

    log.info(f"Run timestamp: {run_ts}")
    log.info(f"Log file: {log_file}")
    log.info(f"Raw response JSONL: {jsonl_file}")
    log.info(f"Run configuration: {manifest_file}")
    log.info(f"Model: {model}")
    log.info(f"Dry run: {dry_run}")
    log.info(f"Delay between calls: {delay}s")
    log.info(f"Screening config: {DEFAULT_CONFIG_PATH}")
    log.info(
        "Training-derived rule references: "
        f"PA max={training_stats['pa_train_max_kcalmol']:.1f}, "
        f"correction mean={training_stats['correction_mean_kcalmol']:.2f}, "
        f"correction std={training_stats['correction_std_kcalmol']:.2f} kcal/mol"
    )

    if not mol_path.exists():
        log.error(f"molecular_pa.parquet not found: {mol_path}")
        sys.exit(1)

    manifest = {
        "run_timestamp": run_ts,
        "iteration": iteration,
        "pipeline_config": CONFIG,
        "training_reference_stats": training_stats,
        "runtime": {
            "model": model,
            "delay_seconds": delay,
            "dry_run": dry_run,
            "skip_llm": skip_llm,
            "resume_jsonl": str(jsonl_file) if resume_jsonl else None,
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    mol_df = pd.read_parquet(mol_path)
    log.info(f"Loaded {len(mol_df):,} molecules from {mol_path}")

    if dry_run:
        mol_df = mol_df.head(20)
        log.warning(f"DRY RUN — processing first {len(mol_df)} molecules only")

    # ── Step 1: Rule-based pre-screen (all molecules, fast) ───────────────
    log.info("Running rule-based pre-screen...")
    rule_results = [
        rule_based_check(row, training_stats, filter_config)
        for _, row in mol_df.iterrows()
    ]
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
        elif model.startswith("vertex_ai/"):
            project_id = (
                os.environ.get("VERTEXAI_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            log.info(
                "Using Vertex AI application-default credentials"
                + (f" for project={project_id}" if project_id else "")
            )
        else:
            api_key = load_gemini_api_key()
            if not api_key:
                log.error("No API key found. Set GEMINI_API_KEY.")
                sys.exit(1)
            log.info(f"Using Gemini API key, model={model}")

        # Send every non-rejected molecule to the model. This is a defined
        # pipeline rule, not a sampled subset.
        to_llm_mask  = mol_df["rule_verdict"] != "reject"
        to_llm       = mol_df[to_llm_mask].copy()
        hard_rejects = mol_df[~to_llm_mask].copy()

        n_to_llm = len(to_llm)
        est_min  = n_to_llm * delay / 60
        log.info(f"Sending {n_to_llm:,} molecules to LLM "
                 f"(~{est_min:.0f} min at {delay}s/call)...")
        log.info(f"  ({n_rule_reject} hard rejects skipped)")

        llm_rows = to_llm.to_dict("records")
        resumed_by_smiles = {}
        if resume_jsonl:
            if not jsonl_file.exists():
                raise FileNotFoundError(f"Resume JSONL not found: {jsonl_file}")
            rows_by_smiles = {row["smiles"]: row for row in llm_rows}
            valid_smiles = set(rows_by_smiles)
            for line_number, line in enumerate(jsonl_file.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed resume JSONL line {line_number}: {exc}"
                    ) from exc
                smiles = record.get("smiles")
                if smiles not in valid_smiles:
                    raise ValueError(
                        "Resume JSONL contains a molecule absent from this run: "
                        f"{smiles}"
                    )
                if record.get("model_requested") != model:
                    raise ValueError(
                        f"Resume model mismatch for {smiles}: "
                        f"{record.get('model_requested')!r} != {model!r}"
                    )
                if record.get("prompt") != build_user_prompt(rows_by_smiles[smiles]):
                    raise ValueError(
                        f"Resume prompt/input mismatch for molecule: {smiles}"
                    )
                if record.get("llm_error") is None:
                    resumed_by_smiles[smiles] = record
            log.info(
                f"Resume: loaded {len(resumed_by_smiles):,} successful responses; "
                f"querying {n_to_llm - len(resumed_by_smiles):,} remaining"
            )

        llm_results = []
        jsonl_fh = open(jsonl_file, "a" if resume_jsonl else "w")
        run_start = time.time()

        for i, row in enumerate(llm_rows):
            if row["smiles"] in resumed_by_smiles:
                llm_results.append(resumed_by_smiles[row["smiles"]])
                continue
            log.info(f"  [{i+1:4d}/{n_to_llm}] smiles={row['smiles'][:60]}  "
                     f"fg={row.get('functional_group', '?')}  "
                     f"PA={row['pa_pred_kcalmol']:.1f}  "
                     f"unc={row.get('uncertainty', 0):.1f}  "
                     f"flags={row.get('rule_flags', 'none')}")

            result = query_llm(
                row, model, sa_key, project_id, api_key, llm_config)
            result["query_index"] = i
            result["wall_clock_cumulative_s"] = round(time.time() - run_start, 1)
            llm_results.append(result)

            jsonl_record = {k: v for k, v in result.items()}
            jsonl_fh.write(json.dumps(jsonl_record, default=str) + "\n")
            jsonl_fh.flush()

            log.info(f"         verdict={result.get('llm_verdict', '?')}  "
                     f"grotthuss={result.get('grotthuss_capable', '?')}  "
                     f"latency={result.get('latency_s', '?')}s  "
                     f"tokens={result.get('total_tokens', '?')}  "
                     f"attempt={result.get('attempt', '?')}  "
                     f"model_ret={result.get('model_returned', '?')}")

            if result.get("structural_concern"):
                log.info(f"         concern: {result['structural_concern']}")

            if result.get("llm_error"):
                log.warning(f"         ERROR: {result['llm_error']}")

            time.sleep(delay)

        jsonl_fh.close()
        log.info(f"Raw responses saved to {jsonl_file}")
        log.info(f"Total LLM wall time: {time.time() - run_start:.0f}s")

        parquet_cols = ["llm_verdict", "grotthuss_capable",
                       "grotthuss_reasoning", "structural_concern", "llm_error",
                       "latency_s", "prompt_tokens", "completion_tokens",
                       "total_tokens", "model_returned", "response_id",
                       "finish_reason", "attempt"]
        llm_df = pd.DataFrame(
            [{k: r.get(k) for k in parquet_cols} for r in llm_results],
            index=to_llm.index)
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
        (accepted["pa_pred_kcalmol"] >= pa_low) &
        (accepted["pa_pred_kcalmol"] <= pa_high)
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
    log.info(
        f"  Accepted in {pa_low:g}-{pa_high:g} window: {in_window.sum():,}"
    )
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

    if "latency_s" in mol_df.columns:
        lat = mol_df["latency_s"].dropna()
        if len(lat) > 0:
            log.info(f"  Latency (s): mean={lat.mean():.2f} "
                     f"min={lat.min():.2f} max={lat.max():.2f}")
    if "total_tokens" in mol_df.columns:
        tok = mol_df["total_tokens"].dropna()
        if len(tok) > 0:
            log.info(f"  Tokens: total={tok.sum():.0f} "
                     f"mean={tok.mean():.0f}/call")
    if "prompt_tokens" in mol_df.columns:
        pt = mol_df["prompt_tokens"].dropna()
        ct = mol_df["completion_tokens"].dropna()
        if len(pt) > 0:
            log.info(f"  Prompt tokens: total={pt.sum():.0f} "
                     f"mean={pt.mean():.0f}/call")
            log.info(f"  Completion tokens: total={ct.sum():.0f} "
                     f"mean={ct.mean():.0f}/call")
    if "model_returned" in mol_df.columns:
        models_used = mol_df["model_returned"].dropna().unique()
        log.info(f"  Models used: {list(models_used)}")
    if "finish_reason" in mol_df.columns:
        fr = mol_df["finish_reason"].value_counts(dropna=False)
        log.info(f"  Finish reasons: {dict(fr)}")

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
        "--model", default=CONFIG["llm"]["model"],
        help="LiteLLM model string (default: pipeline config). "
             "Paper production run used this model via Vertex AI at temperature=0.")
    parser.add_argument(
        "--vertex-key", default=None, metavar="PATH",
        help="Path to Vertex AI service account JSON key file. "
             "If omitted for a Vertex model, Google application-default "
             "credentials are used")
    parser.add_argument(
        "--delay", type=float, default=CONFIG["llm"]["delay_seconds"],
        help="Seconds between API calls (default: pipeline config)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process first 20 molecules only")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM entirely, use rule-based verdicts only")
    parser.add_argument(
        "--resume-jsonl", default=None, metavar="PATH",
        help="Resume a production run from its raw-response JSONL checkpoint")
    args = parser.parse_args()

    main(
        iteration      = args.iter,
        model          = args.model,
        vertex_key_path= args.vertex_key,
        dry_run        = args.dry_run,
        skip_llm       = args.skip_llm,
        delay          = args.delay,
        resume_jsonl    = args.resume_jsonl,
    )
