"""Repeat the observed-record reference tests using archived Monte Carlo counts."""

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


REFERENCES = ["common_phase", "envelope_phase", "residual_block"]
ENDPOINTS = ["maxdev", "variance"]


def observed_tests(counts, windows):
    record_ids = windows["record_id"]
    if record_ids.nunique() != 36 or len(windows) != 36:
        raise ValueError("Expected counts for 36 unique observed records")
    keys = ["record_id", "endpoint", "reference_model"]
    expected = pd.MultiIndex.from_product([record_ids, ENDPOINTS, REFERENCES], names=keys)
    data = counts.set_index(keys).sort_index()
    if not data.index.equals(expected.sort_values()):
        raise ValueError("Expected three reference processes for both endpoints of every record")
    if not data.draws.eq(2999).all() or not data.exceedances.between(0, data.draws).all():
        raise ValueError("Invalid Monte Carlo counts")
    support = data.index.get_level_values("reference_model").map({
        "common_phase": 508, "envelope_phase": 508, "residual_block": 504,
    })
    if not data.n_windows.eq(support).all():
        raise ValueError("Unexpected reference-process support")
    p_values = (data.exceedances + 1) / (data.draws + 1)
    result = p_values.unstack("reference_model")[REFERENCES]
    result["largest_p"] = result.max(axis=1)
    result["holm_72"] = multipletests(result.largest_p, method="holm")[1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    windows = pd.read_csv("observed_window_counts.csv")
    result = observed_tests(pd.read_csv("observed_reference_counts.csv"), windows)
    if not windows.boundary_windows.between(0, windows.windows).all():
        raise ValueError("Invalid window counts")
    print(f"{windows.boundary_windows.sum()} of {windows.windows.sum()} observed windows reached the search boundary")
    args.table_dir.mkdir(parents=True, exist_ok=True)
    focal = result.loc[[("ses-03_t27_Amyg_p000", "maxdev")]]
    focal.to_csv(args.table_dir / "Table_S12.csv")
    output = Path("outputs/observed_departures_holm72.csv")
    output.parent.mkdir(exist_ok=True)
    result.to_csv(output)
    print(f"{result.holm_72.le(0.05).sum()} of {len(result)} endpoints passed Holm correction; see {output}")


if __name__ == "__main__":
    main()
