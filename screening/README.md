# `screening/` — prospective candidate screening pipeline

Seven-stage pipeline that screens a library of ~821 K ZINC molecules against
the target proton-affinity window (210–235 kcal/mol) and hands 30 Pareto-
selected candidates off to DFT validation.

All data for this pipeline lives under `../data/screening/` (NOT
`screening/data/` — consolidated in the repo reorg). Figures go to
`screening/figures/` and logs to `screening/logs/`.

Each stage resolves its paths from `Path(__file__)`, so you can run it from
anywhere.

---

## Pipeline stages (`scripts/execution/`)

Replace `{N}` with the iteration number (1 for the results reported in the paper).

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 01 | `01_build_index.py`        | `data/screening/zinc_raw/filtered_821k.csv`, Morgan-FP JSONs | `data/screening/processed/{zinc_metadata.parquet, zinc_fingerprints.npy, zinc_fp_keys.npy, zinc_index.faiss, build_report.json}` |
| 02 | `02_retrieve_candidates.py`| `data/screening/processed/*`, `data/screening/iter{N-1}/pareto_selected.csv` (if N>1) | `data/screening/iter{N}/candidates.parquet` |
| 03 | `03_run_pm7.py`            | `data/screening/iter{N}/candidates.parquet` | `data/screening/iter{N}/pm7_results.parquet`, `data/screening/iter{N}/pm7_raw/` |
| 04 | `04_featurize.py`          | `data/screening/iter{N}/pm7_results.parquet` | `data/screening/iter{N}/features.parquet` |
| 05 | `05_predict_pa.py`         | `data/screening/iter{N}/features.parquet`, `results/kmeans251/cv_results.json` | `data/screening/iter{N}/predictions.parquet`, `data/screening/iter{N}/molecular_pa.parquet` |
| 06 | `06_llm_verify.py`         | `data/screening/iter{N}/molecular_pa.parquet` | `data/screening/iter{N}/llm_verdicts.parquet` |
| 07 | `07_pareto_select.py`      | `data/screening/iter{N}/llm_verdicts.parquet`, `data/screening/processed/*` | `data/screening/iter{N}/pareto_selected.csv`, `data/screening/iter{N}/pareto_report.json` |
| 11 | `11_parse_dft_files.py`    | `data/screening/iter{N}/{pareto_selected.csv, pareto_dft_files/}` | `data/screening/iter{N}/dft_files_parsed.parquet`, `data/screening/iter{N}/dft_files_summary.csv` |

Helpers:
- `run_pm7_parallel.py` — parallel wrapper used by stage 03
- `mopac_calculator.py` — MOPAC invocation + PA extraction
- `sascorer.py`, `fpscores.pkl.gz` — Ertl & Schuffenhauer (2009) SA score

## Plotting (`scripts/plotting/`)

| Script | Purpose |
|---|---|
| `08_plot_results.py`      | Main screening-pipeline figures (PA distribution, etc.) |
| `plot_pareto.py`          | `iter{N}_pareto.pdf` — 3-panel Pareto selection summary |
| `plot_pa_parity_final.py` | PM7 vs DFT parity + bias annotations for the 30 validated candidates |
| `plot_si_candidates.py`   | `si_all30_candidates.pdf` — SI grid of all 30 validated candidates |

## Example run (iteration 1, matching paper results)

```bash
python screening/scripts/execution/01_build_index.py
python screening/scripts/execution/02_retrieve_candidates.py --iteration 1
python screening/scripts/execution/03_run_pm7.py            --iteration 1
python screening/scripts/execution/04_featurize.py          --iteration 1
python screening/scripts/execution/05_predict_pa.py         --iteration 1
python screening/scripts/execution/06_llm_verify.py         --iteration 1
python screening/scripts/execution/07_pareto_select.py      --iteration 1

# ...run DFT externally on the 30 pareto_selected.csv candidates...

python screening/scripts/execution/11_parse_dft_files.py    --iteration 1
python screening/scripts/plotting/plot_pareto.py            --iteration 1
python screening/scripts/plotting/plot_pa_parity_final.py   --iteration 1
python screening/scripts/plotting/plot_si_candidates.py
```
