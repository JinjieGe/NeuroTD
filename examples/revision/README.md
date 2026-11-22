# Revision evidence

The [evidence map](../../docs/revision-evidence-map.md) connects reviewer comments to the corresponding code. See the [run guide](../../docs/revision-running.md) for full experiments and [limitations](../../docs/revision-limitations.md) for scope. This bundle excludes Comment 2.5. Run the commands from the repository root.

```bash
python -m pip install -e .
python -m pip install -r examples/revision/requirements.txt
python examples/revision/run.py list
python examples/revision/run.py evidence --workspace revision_runs/my_run
```

The evidence command writes S1–S9, S11–S12 and Supplementary Data S1 to `revision_runs/my_run/supplement_tables/`. It recomputes the six regional comparisons and the matched-control and observed-record tests. Use `python examples/revision/verify_table1.py .` for the main benchmark table. Manuscript figure commands are in the [project README](../../README.md).

| Directory or file | Contents |
|---|---|
| `run.py`, `revision_runtime.py` | Launcher, path configuration, separate workspaces, and execution records |
| `studies/score_uncertainty/` | Known-delay selection and uncertainty across independent signals |
| `studies/dynamic_fixed/`, `studies/dynamic_selection/` | Matched interference controls, overlap and reconstruction studies, and constant-delay calibration including window selection |
| `studies/session_inference/` | Calibrated regional comparisons, independent validation, and earlier inference methods |
| `studies/dataset_reporting/` | NWB metadata and processed Patch-seq sweep counts |
| `studies/search_range/` | W/4 versus W/2 search-range ablation |
| `studies/snr_initial/`, `studies/snr_extension/` | Observed signal-quality proxies and known-shift noise experiments |
| `studies/figures_*/` | Response figures from saved results |
| `studies/audit_v1/` through `audit_v3/` | Historical statistical and electrode audits |
| `inputs/`, `evidence/` | Small staged inputs, known-truth simulations, and archived results |
| `source_manifest.csv`, `data_manifest.csv` | Provenance and SHA-256 hashes |

The launcher stages scripts and inputs in a workspace. When an existing staged copy differs, use a fresh workspace to keep results tied to the code that generated them.

## Two short examples

Count processed spike intervals and regenerate the four-panel figure for Comment 2.2:

```bash
python examples/revision/run.py run --workspace revision_runs/inventory --study dataset_reporting --script audit_inputs.py
python examples/revision/run.py figures --workspace revision_runs/figure22 --study figures_22
```

The figure is written to `revision_runs/figure22/figures_22/deliverables/`. Workspaces marked `ARCHIVED_FIGURES_ONLY` are reserved for figures generated from saved results. The estimation and uncertainty analyses use complete simulated signal pairs as their independent units.
