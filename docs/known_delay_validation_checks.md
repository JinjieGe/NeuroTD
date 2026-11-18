# Known-delay validation record, September 24, 2026

This audit added ground-truth comparisons to commit `1a6c4f23818bcfe3152f947c54c553a92edeca0c` while retaining the estimator, E, default signal models, and original table calculations.

## Checks performed

- Ten unit tests passed, including recovery of a +8-sample shift in an aperiodic signal, unit conversion, E-based selection independent of true RMSE, ties, zero denominators, and shared centers.
- The full discrete grid covered seeds 0, 1, and 2, eleven noise levels, and six windows, for 198 combinations.
- For each discrete seed, the audit's 264 method RMSEs and 66 E values matched the original script in the same run within a floating-point tolerance of 1e-12.
- The continuous smoke run used seed 2025, noise 0 and 0.02, and windows 65 and 513. Its 16 method RMSEs matched the original outputs. The comparison figures were inspected. This September audit covered four continuous conditions.

## Discrete results at native window centers

| Seed | Noise levels where automatic selection attained oracle RMSE | Maximum excess RMSE |
|---|---:|---:|
| 0 | 11/11 | 0 ms |
| 1 | 11/11 | 0 ms |
| 2 | 7/11 | 70.556315 ms |

For seed 2, four conditions produced large errors.

| Noise SD | E-selected W | Oracle W | Automatic delay RMSE | Oracle delay RMSE |
|---:|---:|---:|---:|---:|
| 0.04 | 1025 | 513 | 70.710723 ms | 0.154408 ms |
| 0.06 | 1025 | 513 | 70.710723 ms | 0.169146 ms |
| 0.08 | 1025 | 513 | 70.710723 ms | 0.182698 ms |
| 0.10 | 1025 | 513 | 70.710723 ms | 0.218366 ms |

The same estimator and candidate set can align repeated signals well while yielding a large delay error. All failures are retained. The four methods shared just six centers across the full discrete grid. The continuous smoke test compares only two candidate windows. The run manifest records software versions, status, and source SHA-256 hashes.

## Reproduce

```bash
python -m unittest discover -s tests -v
python examples/validate_known_delays.py --simulation discrete --seeds 0 1 2
python examples/validate_known_delays.py --simulation continuous --smoke
```

Tables, traces, and figures are written to fresh local output directories. The results describe the tested signals and seeds. See the [validation guide](known_delay_validation.md) for timing conventions and interpretation.
