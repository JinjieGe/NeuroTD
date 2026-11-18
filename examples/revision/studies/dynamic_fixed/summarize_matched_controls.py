"""Reproduce the 15 November 2026 matched controls from simulation record summaries.

Each public record has two Monte Carlo rank numerators (1 + number of reference
statistics >= the test statistic) for endpoints T and S. The corresponding
p-values divide by 3,000 because the matched frozen bank had 2,999 draws per
template. Dynamic controls also have a saved per-record delay RMSE in ms.
This entry runs Holm over 36 records x 2 endpoints within each test panel,
counts constant panels with any rejection, and averages dynamic record
detection fractions and mean record RMSEs across panels for each scenario.
"""
import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd
from scipy.stats import beta


SCENARIOS = ("stationary_white", "varying_gain", "varying_noise", "record_spectral_noise",
             "repeated_waveform", "common_transients", "observed_gain_envelope")
PANEL_KEYS = ["scenario_name", "composition", "replicate"]
RECORD_KEYS = PANEL_KEYS + ["record_index"]
NUMERATORS = ["rank_numerator_T", "rank_numerator_S"]
PANELS = 200
RECORDS = 36
REFERENCE_DRAWS = 2999


def original_records(results):
    """Project the original endpoint and error tables onto the public record columns."""
    endpoints = pd.read_csv(results / "control_rankp_holm72.csv.gz", float_precision="round_trip",
                            usecols=RECORD_KEYS + ["calibration", "scenario", "endpoint", "p_raw"])
    matched = endpoints.loc[(endpoints.calibration == "matched") & endpoints.scenario_name.isin(SCENARIOS)
                            & endpoints.composition.isin(["constant", "dynamic"])]
    wide = matched.pivot(index=RECORD_KEYS, columns="endpoint", values="p_raw")
    p = wide[["T", "S"]].to_numpy()
    denominator = REFERENCE_DRAWS + 1
    ranks = np.rint(p * denominator)
    if not np.allclose(p, ranks / denominator, rtol=1e-14, atol=0):
        raise ValueError("Original p-values do not match the frozen-bank rank grid")
    records = pd.DataFrame(ranks.astype(int), index=wide.index, columns=NUMERATORS).reset_index()

    stats = pd.read_csv(results / "control_statistics_all.csv.gz", float_precision="round_trip",
                        usecols=["scenario", "split", "dynamic", "replicate", "record_index", "rmse_ms"])
    names = matched[["scenario", "scenario_name"]].drop_duplicates()
    dynamic = stats.loc[(stats.split == 1) & stats.dynamic].merge(names, on="scenario", validate="many_to_one")
    dynamic["composition"] = "dynamic"
    return records.merge(dynamic[RECORD_KEYS + ["rmse_ms"]], on=RECORD_KEYS, how="left", validate="one_to_one")


def analyze_records(records):
    expected = pd.MultiIndex.from_product(
        [SCENARIOS, ("constant", "dynamic"), range(PANELS), range(RECORDS)], names=RECORD_KEYS)
    indexed = records.set_index(RECORD_KEYS)
    if len(indexed) != len(expected) or not indexed.index.is_unique or not expected.isin(indexed.index).all():
        raise ValueError("Expected 36 records in each of 200 constant and 200 dynamic panels per scenario")
    indexed = indexed.reindex(expected)
    ranks = indexed[NUMERATORS].to_numpy()
    denominator = REFERENCE_DRAWS + 1
    if not ((ranks >= 1) & (ranks <= denominator) & (ranks == np.floor(ranks))).all():
        raise ValueError(f"Each raw rank numerator must be an integer between 1 and {denominator}")

    # Holm step-down: stop at the first ordered p-value exceeding its cutoff.
    p = (ranks / denominator).reshape(-1, RECORDS * 2)
    order = np.argsort(p, axis=1)
    ordered_p = np.take_along_axis(p, order, axis=1)
    passed = np.logical_and.accumulate(ordered_p <= 0.05 / np.arange(p.shape[1], 0, -1), axis=1)
    rejected = np.take_along_axis(passed, np.argsort(order, axis=1), axis=1)

    panel_index = expected.droplevel("record_index").unique()
    dynamic = panel_index.get_level_values("composition") == "dynamic"
    errors = indexed.rmse_ms.to_numpy().reshape(-1, RECORDS)
    if not np.isnan(errors[~dynamic]).all() or not (np.isfinite(errors[dynamic]) & (errors[dynamic] >= 0)).all():
        raise ValueError("Expected one finite nonnegative RMSE for every dynamic record, blank for constant controls")
    panels = pd.DataFrame(index=panel_index)
    panels["any_rejection"] = rejected.any(axis=1)
    panels["fraction_records_detected"] = rejected.reshape(-1, RECORDS, 2).any(axis=2).mean(axis=1)
    panels["mean_rmse_ms"] = errors.mean(axis=1)
    return panels.sort_index()


def verify_panels(results, panels):
    """Compare all recomputed constant flags and dynamic outcomes to the historical panel tables."""
    inference = pd.read_csv(results / "control_panel_inference.csv", float_precision="round_trip",
                            dtype={"any_false_rejection": "boolean"},
                            usecols=PANEL_KEYS + ["calibration", "any_false_rejection", "power_record_any_endpoint"])
    matched = inference.loc[(inference.calibration == "matched") & inference.scenario_name.isin(SCENARIOS)
                            & inference.composition.isin(["constant", "dynamic"])]
    saved = matched.set_index(PANEL_KEYS).sort_index()
    if not saved.index.equals(panels.index):
        raise ValueError("Original inference panel identifiers differ from the reconstructed families")
    saved = saved.reindex(panels.index)
    constant = panels.xs("constant", level="composition")
    dynamic = panels.xs("dynamic", level="composition")
    np.testing.assert_array_equal(constant.any_rejection, saved.xs("constant", level="composition").any_false_rejection)
    np.testing.assert_allclose(dynamic.fraction_records_detected,
                               saved.xs("dynamic", level="composition").power_record_any_endpoint,
                               rtol=1e-14, atol=0, equal_nan=False)
    keys = ["scenario_name", "replicate"]
    tracking = pd.read_csv(results / "control_panel_tracking.csv", float_precision="round_trip",
                           usecols=keys + ["dynamic", "rmse_ms"])
    errors = tracking.loc[tracking.dynamic & tracking.scenario_name.isin(SCENARIOS)].set_index(keys)
    if not errors.sort_index().index.equals(dynamic.index):
        raise ValueError("Original dynamic tracking panel identifiers differ from the reconstructed panels")
    np.testing.assert_allclose(dynamic.mean_rmse_ms, errors.loc[dynamic.index, "rmse_ms"],
                               rtol=1e-14, atol=0, equal_nan=False)
    print(f"Verified {len(panels):,} Holm families and {len(dynamic):,} dynamic panel mean RMSEs "
          "against the original panel tables.")


def summarize(panels):
    scenarios = []
    for scenario in SCENARIOS:
        constant = panels.loc[(scenario, "constant")]
        dynamic = panels.loc[(scenario, "dynamic")]
        n = len(constant)
        count = int(constant.any_rejection.sum())
        upper = 1.0 if count == n else float(beta.ppf(0.95, count + 1, n - count))
        scenarios.append(dict(
            scenario=scenario, tier="supplementary" if scenario == "observed_gain_envelope" else "primary",
            constant_panels=n, panels_with_any_false_rejection=count, group_false_positive_rate=count / n,
            group_false_positive_rate_CP95_one_sided_upper=upper,
            dynamic_panels=len(dynamic), mean_dynamic_record_detection_fraction=fmean(dynamic.fraction_records_detected),
            mean_dynamic_record_rmse_ms=fmean(dynamic.mean_rmse_ms),
        ))
    return dict(reference_scope="Conditional on the frozen matched reference banks",
                interval_scope="One-sided 95% Clopper-Pearson upper bound for each individual scenario",
                reference_draws_per_template=REFERENCE_DRAWS, records_per_panel=RECORDS, holm_family_size=RECORDS * 2,
                scenarios=scenarios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path,
                        help="Read control_rankp_holm72.csv.gz and control_statistics_all.csv.gz "
                             "from this original results directory.")
    parser.add_argument("--verify-endpoints", action="store_true",
                        help="Compare the reconstructed outcomes against control_panel_inference.csv "
                             "and control_panel_tracking.csv in --source-results.")
    args = parser.parse_args()
    if args.verify_endpoints and not args.source_results:
        parser.error("--verify-endpoints requires --source-results")
    if args.source_results:
        records = original_records(args.source_results)
    else:
        records = pd.read_csv("matched_control_records.csv.gz", float_precision="round_trip")
    panels = analyze_records(records)
    if args.verify_endpoints:
        verify_panels(args.source_results, panels)
    summary = summarize(panels)
    destination = Path("outputs/matched_control_summary.json")
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    counts = [row["panels_with_any_false_rejection"] for row in summary["scenarios"]]
    low, high = min(counts), max(counts)
    print(f"Matched constant controls: {low}–{high}/{PANELS} panels with a false rejection "
          f"({low / PANELS:.1%}–{high / PANELS:.1%}), conditional on the frozen banks.")
    print("Scenario                 Null rejections   95% upper   Dynamic detection   Mean dynamic RMSE")
    for row in summary["scenarios"]:
        print(f"{row['scenario']:24} {row['panels_with_any_false_rejection']:2}/{PANELS:<10} "
              f"{row['group_false_positive_rate_CP95_one_sided_upper']:>8.2%} "
              f"{row['mean_dynamic_record_detection_fraction']:>18.2%} "
              f"{row['mean_dynamic_record_rmse_ms']:>16.3f} ms")
    print(destination)


if __name__ == "__main__":
    main()
