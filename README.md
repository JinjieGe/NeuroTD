# NeuroTD infers time varying delays in neural activities by adaptive sliding window alignment

NeuroTD estimates the local time shift between two neural recordings by sliding a window along the signals and minimizing the alignment error in each window. Simulations compare the estimated shifts with known delays. The real-data examples examine within-region electrode pairs in human intracranial recordings and associations between mouse Patch-seq firing features and gene expression.

<img src="./docs/assets/fig1.png" alt="Overlapping windows align two signals and estimate how their relative shift changes over time" width="700"/>

## Reproduce the revision evidence

After installing the package, run the following from the repository root:

```bash
python -m pip install -r examples/revision/requirements.txt
python examples/revision/run.py evidence --workspace revision_runs/current
```

This writes **Tables S1–S9** to `revision_runs/current/supplement_tables/`, recomputes the six regional comparisons from trial summaries, and applies the 72-test Holm correction within each panel of matched controls. The table inputs include archived simulation outcomes. To run the underlying experiments, including the independent Welch calibration, see the [run instructions](docs/revision-running.md). The [comment-to-code map](docs/revision-evidence-map.md) covers the other analyses and their scope. This branch excludes Comment 2.5.

## Installation Instructions (Ubuntu 24.04 LTS, WSL 2.6 in Windows 10/11 Pro)

First, clone and navigate to the repository.

```bash
git clone --branch revision/organized-evidence-20261108 https://github.com/JinjieGe/NeuroTD.git
cd NeuroTD
```

This process can take several minutes, depending on network speed.

Create and activate a virtual environment using python 3.12 with `conda`,

```bash
conda create -n neurotd python=3.12
conda activate neurotd
```

Install dependencies and the local library with `pip`. NeuroTD uses a modern `pyproject.toml`–based build system, with all runtime dependencies declared directly in `pyproject.toml`.
From the repository root, install the package with:

```bash
python -m pip install -e .
```

This process usually takes several minutes.

## Usage

### Validate window selection against known simulated delays

The original simulations already compute delay RMSE against generating delays.
An additional audit now records all four methods' unrounded delay trajectories,
compares NeuroTD's minimum-E window with a retrospective true-RMSE oracle, and
reports the excess delay RMSE. E is the existing `sqrt(mean(r_k))` alignment
diagnostic, not manuscript Equation (6)'s q or a calibrated confidence score.
Ground truth is used for evaluation, **not** to select the E-based result.

```bash
python examples/validate_known_delays.py --simulation both --smoke
# Full 11-noise x 6-window grids:
python examples/validate_known_delays.py --simulation both
```

Each command writes to a new `validation_runs/` directory without overwriting
earlier runs. See [the validation guide](docs/known_delay_validation.md) for
multi-seed runs, output definitions, shared-center checks, and the limitations
of the original continuous-delay truth convention. Estimators are unchanged.

### Example 1: Simulation study with discrete time-varying delays

The discrete-delay simulation evaluates NeuroTD under controlled conditions where the delay between two signals changes in a piecewise manner over time. Gaussian noise is added at multiple levels to assess robustness. NeuroTD is compared against Dynamic Time Warping (DTW), Canonical Time Warping (CTW), and FFT-based baselines using root-mean-square error (RMSE).

Run the discrete-delay simulation from the NeuroTD project directory:

```bash
python examples/run.py fig2 --headless
```
The script may also be executed in interactive environments (e.g., VS Code or Jupyter), which manage figure display automatically.

This script reproduces the results shown below:

- **(a)** simulated time-series data with corresponding ground-truth time-varying delays, with zoomed-in views of the first and last segments,
- **(b)** estimated delay trajectories at a fixed noise level, comparing NeuroTD with baseline methods, and
- **(c)** root-mean-square error (RMSE) as a function of noise level across methods.

<img src="./docs/assets/fig2.png" alt="" width="900"/>

### Example 2: Simulation study with continuous time-varying delays

The continuous-delay simulation evaluates NeuroTD under controlled conditions where the delay between two signals varies smoothly over time. This setting is designed to mimic gradual changes in neural timing. Gaussian noise is added at multiple levels to assess robustness. NeuroTD is evaluated by comparing estimated delay trajectories against known ground truth using root-mean-square error (RMSE).

Run the continuous-delay simulation from the NeuroTD project directory:

```bash
python examples/run.py fig3 --headless
```
The script may also be executed in interactive environments (e.g., VS Code or Jupyter), which manage figure display automatically.

This script reproduces three figures shown below:

- **(a)** simulated time-series data with corresponding ground-truth time-varying delays,
- **(b)** estimated delay trajectories at a fixed noise level, comparing NeuroTD with baseline methods, and
- **(c)** root-mean-square error (RMSE) as a function of noise level across methods.

<img src="./docs/assets/fig3.png" alt="" width="900"/>

## Repository layout

```text
src/neurotd/           Importable algorithms and reusable application helpers
examples/simulations/  Paper simulation and schematic entry points
examples/ieeg/         Intracranial-recording analysis entry point
examples/patchseq/     Patch-seq analysis and preprocessing entry points
examples/revision/     Categorized revision studies, inputs, and aggregate evidence
examples/legacy/       Preserved exploratory scripts (not supported examples)
data/processed/       Four previously published processed input files
docs/assets/          Images used in this README
tests/                Small tests and a continuous-simulation smoke harness
results/              Generated outputs (ignored by Git)
```

Use a Python 3.12+ environment and select the same interpreter in your editor.
Check `python -c "import neurotd; print(neurotd.__file__)"` to ensure you are using
this checkout rather than another editable installation.

The launcher saves outputs in `results/fig2/`, `results/fig3/`, etc. Omit
`--headless` to display interactive figures; `plt.show(block=True)` may wait until
the figure windows are closed. Existing files in the selected output directory
may be overwritten on rerun, so use `--output-dir` for a fresh run:

```bash
python examples/run.py fig2 --headless --output-dir results/discrete-run-2
```

The original algorithms, random seeds and full simulation grids are unchanged.
The continuous full grid can take substantially longer than the discrete grid.
Scripts can also be run directly, e.g.
`python examples/simulations/fig2_simulation_discrete_shift.py`; relative outputs
then go to the current working directory. Prefer the launcher to keep them separate.

## Real-data applications

See [data requirements](data/README.md) before running either application.
No raw recordings are automatically downloaded. For iEEG, point to the DANDI
000574 folder containing the `sub-*` directories:

```bash
python examples/run.py fig4 --headless --ieeg-dir /path/to/000574
```

Figure 5 uses the processed Patch-seq files shipped in `data/processed/`. First
extract the CSV from the ZIP as described in the data instructions, then run:

```bash
python examples/run.py fig5 --headless
```

This generates the gene list and correlation plot; enrichment analysis is a
separate external step described in the script, not an automated online upload.
Raw Patch-seq preprocessing is resource-intensive and is not part of the small
test suite. Refer to [example notes](examples/README.md).

For the optional editable schematic (Figure 1), install the extra dependency:

```bash
python -m pip install -e '.[examples]'
python examples/run.py fig1
```

## Tests

After installing the package, run the small, offline tests from the checkout:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

These check imports, path configuration, the repository layout, and basic delay
estimation. They do not certify all benchmark implementations or biological claims.
See [migration and validation notes](docs/repository-layout.md) for the scope of
the reorganization and known limitations.

<!-- ## Citations

If you use NeuroTD in your research, please cite our paper:

Xiang Huang, Noah Cohen Kalafut, Sayali Anil Alatkar, Athan Z. Li, Qiping Dong, Qiang Chang, and Daifeng Wang, NeuroTD infers time varying delays in neural activities by adaptive sliding window alignment. -->
