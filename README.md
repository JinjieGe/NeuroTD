# NeuroTD estimates time-varying waveform delays in neural recordings with residual-guided window selection

NeuroTD estimates a local shift between two recordings by minimizing a waveform alignment cost. For each candidate window length it aggregates the normalized, aligned-window residuals as `E = sqrt(mean(r_k))`. It selects the window with the smallest E. In simulations the known generating delays provide a separate assessment of delay accuracy. Applications examine within-region human intracranial recordings and spike-interval adaptation in mouse Patch-seq neurons.

## Install

Use Python 3.12 and run the commands from the checkout root. The [original software](https://github.com/daifengwanglab/NeuroTD) and the [archived revision analysis](https://github.com/JinjieGe/NeuroTD/tree/7d95c94a2e7ef4fc6bc476567c79f5f64e54bac6) provide the project history.

```bash
git clone --branch revision/organized-evidence-20261108 https://github.com/JinjieGe/NeuroTD.git
cd NeuroTD
conda create -n neurotd python=3.12
conda activate neurotd
python -m pip install -e .
python -m pip install -r examples/revision/requirements.txt
```

## Reproduce the manuscript results

```bash
python examples/revision/verify_table1.py .
python examples/revision/run.py evidence --workspace revision_runs/current
```

The first command regenerates the two seeded fixed-window examples using all four estimators and checks the 132 archived noise-by-window rows used in Table 1. Panel A uses the same evaluation centers for every method. Panel B uses each implementation's native centers.

The second command writes Tables S1–S9 and S11–S12 and the ranked gene list (Supplementary Data S1) to `revision_runs/current/supplement_tables/`. It uses archived simulation outcomes, Monte Carlo counts and trial summaries. Instructions for regenerating the underlying analyses and obtaining raw inputs are in the [run guide](docs/revision-running.md) and [data inventory](data/README.md). Table S10 summarizes the cross-trial reliability analysis supplied with the manuscript. Its computation is outside this code release.

| Manuscript item | Entry point or input |
|---|---|
| Fig. 1 | `python examples/revision/render_figure.py fig1 --output results/fig1_revised.pdf` renders the corrected shift and delay labels. |
| Figs. 2–3 and Tables S1–S2 | `python examples/run.py fig2 --headless` and `python examples/run.py fig3 --headless` run the original benchmark grids. After the evidence command, use `python examples/revision/render_figure.py fig2 --output results/fig2_revised.pdf` for the revised, linear-scale panel 2c. Fixed-window trajectories and across-window comparisons use different selection rules. |
| Tables S3–S5 | [Independent noise repetitions](examples/revision/studies/score_uncertainty) evaluate selection and uncertainty over 1,200 signal pairs. |
| Fig. 4 and Tables S6–S9 | `python examples/run.py fig4 --headless --ieeg-dir /path/to/000574` computes the original within-region delays for participant 01, sessions 01–03. The evidence command calculates the six calibrated regional contrasts from trial summaries. Afterward, `python examples/revision/render_figure.py fig4 --output results/fig4bc.pdf` draws panels b–c. Panel a reuses the original example trace. |
| Tables S11–S12 | [Matched-control and observed-record analyses](examples/revision/studies/dynamic_fixed) distinguish recovery of known delays from excursions in observed curves. S12 uses the largest of three reference-process p-values per endpoint and Holm adjustment across 72 endpoints. |
| Fig. 5 and Supplementary Data S1 | `python examples/run.py fig5 --headless` analyzes processed Patch-seq intervals and expression after [extracting the matrix](data/README.md). The gene list reports signed Spearman correlations in the cells expressing each gene. The paper retains the original enrichment illustration. |

For the Figure 4 display, 100-bin histograms and medians summarize a seeded sample of 200,000 nonzero windows per region. Statistical comparisons use trial summaries. To include panel a, add `--original-fig4 /path/to/fig4.png` using the 3,129 × 937 pixel artwork from the original manuscript project. The public data provide the source signals, but the exact trace excerpt was retained as artwork.

## Source and validation

- [Core package](src/neurotd), [figure and application scripts](examples), [revision evidence by reviewer comment](docs/revision-evidence-map.md), and [scope of the evidence](docs/revision-limitations.md).
- [Known-delay validation](docs/known_delay_validation.md) gives the independent score-versus-delay comparison and explains the generating-delay convention. For a small run, use `python examples/validate_known_delays.py --simulation both --smoke`.
- Run the offline tests with `MPLBACKEND=Agg python -m unittest discover -s tests -v`. The example launcher writes results under `results/`. For a separate run, supply `--output-dir`.

The original schematic and simulation illustrations are shown below. The revised rendering commands above reproduce the updated panels.

<img src="./docs/assets/fig1.png" alt="Original illustration of local sliding-window delay estimation" width="600"/>

<img src="./docs/assets/fig2.png" alt="Original illustration of the discrete-delay benchmark" width="700"/>

<img src="./docs/assets/fig3.png" alt="Continuous-delay simulation and window-selection comparisons" width="700"/>
