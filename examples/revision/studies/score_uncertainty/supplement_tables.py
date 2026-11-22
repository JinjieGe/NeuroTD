"""Rebuild supplementary CSV tables from the archived benchmarks and revision evidence."""

import argparse
import csv
from collections import Counter, defaultdict
from io import StringIO
from itertools import combinations, product
from pathlib import Path
from statistics import mean

from revision_runtime import resolve_path


MS_PER_SAMPLE = 1000 / 5120
WINDOWS = (33, 65, 129, 257, 513, 1025)
REPEATED_NOISE = (0.02, 0.10, 0.20)
# These locations were marked as revised in the manuscript. All values,
# including the six marked entries, are read from the benchmark run.
S1_REVISED_CELLS = {
    (0.0, 65, "dtw"), (0.0, 513, "neurotd"), (0.0, 1025, "neurotd"),
    (0.02, 65, "dtw"), (0.06, 257, "dtw"), (0.20, 65, "dtw"),
}


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def check_grid(cells, expected, label):
    counts = Counter(cells)
    expected = set(expected)
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError(f"{label}: expected exactly one row for each of {len(expected)} conditions.")


def benchmark_table(records, discrete):
    cells = ((float(r["noise_std"]), int(r["window_size"])) for r in records)
    noise_levels = [percent / 100 for percent in range(0, 21, 2)]
    check_grid(cells, product(noise_levels, WINDOWS), "Discrete benchmark" if discrete else "Continuous benchmark")
    rows = [["Noise SD", "Window (samples, ms)", "E", "NeuroTD", "DTW", "CTW", "FFT"]]
    for record in sorted(records, key=lambda r: (float(r["noise_std"]), int(r["window_size"]))):
        noise, window = float(record["noise_std"]), int(record["window_size"])
        noise_label = f"{noise:.2f}" if discrete else (f"{noise:g}" if noise else "0.000")
        row = [noise_label, f"{window} ({window * MS_PER_SAMPLE:.1f})",
               f"{float(record['quality_rmse_neurotd']):.3f}"]
        for method in ("neurotd", "dtw", "ctw", "fft"):
            delay_rmse = float(record["rmse_" + method]) * MS_PER_SAMPLE
            revised = discrete and (round(noise, 2), window, method) in S1_REVISED_CELLS
            row.append(f"{delay_rmse:.3f}" + ("†" if revised else ""))
        rows.append(row)
    return rows


def selection_tables(repetitions, continuous_benchmark):
    conditions = defaultdict(list)
    for record in repetitions:
        noise = float(record["noise_std"])
        if noise != 0:
            conditions[record["model"], noise].append(record)

    s3 = [["Model", "Noise SD", "Automatic\nmean RMSE\n(ms)", "Oracle\nmean RMSE\n(ms)",
           "Mean excess\nRMSE\n(ms)", "Same window\nn / 200"]]
    # Retain the manuscript order, including the two higher-noise discrete conditions.
    order = [("discrete", 0.02), ("continuous", 0.02), ("continuous", 0.10),
             ("continuous", 0.20), ("discrete", 0.10), ("discrete", 0.20)]
    if set(conditions) != set(order):
        raise ValueError("Window selection requires exactly two models at noise SD 0.02, 0.10, and 0.20.")
    for model, noise in order:
        trials = conditions[model, noise]
        if len(trials) != 200 or len({r["replicate_id"] for r in trials}) != 200:
            raise ValueError(f"Expected 200 distinct signal pairs for {model}, SD {noise}.")
        means = [mean(float(r[col]) for r in trials)
                 for col in ("selected_rmse_ms", "oracle_rmse_ms", "selection_loss_ms")]
        same = sum(r["selected_W"] == r["oracle_W"] for r in trials)
        s3.append([model.title(), f"{noise:.2f}", *(f"{v:.3f}" for v in means),
                   f"{same}/{len(trials)}"])

    noiseless = [r for r in continuous_benchmark if float(r["noise_std"]) == 0]
    baseline = min(noiseless, key=lambda r: float(r["quality_rmse_neurotd"]))
    window = int(baseline["window_size"])
    s4 = [["Noise SD", "Independent\nnoise repetitions", "Selected W\n(samples)",
           "Window\n(ms)", "Selection\ncount"],
          ["0.00", "—", window, f"{window * MS_PER_SAMPLE:.1f}", "Single baseline"]]
    for noise in REPEATED_NOISE:
        trials = conditions["continuous", noise]
        counts = Counter(int(r["selected_W"]) for r in trials)
        for window, count in sorted(counts.items()):
            frequency = f"{count}/{len(trials)}\n({100 * count / len(trials):g}%)"
            s4.append([f"{noise:.2f}", len(trials), window,
                       f"{window * MS_PER_SAMPLE:.1f}", frequency])
    return s3, s4


def uncertainty_table(records):
    rows = [["Model", "Noise SD", "Window W\n(samples)", "Mean E", "SD of E",
             "Relative variability\nCV (%)"]]
    positive = [r for r in records if float(r["noise_std"]) != 0]
    cells = ((r["model"], float(r["noise_std"]), int(r["window_size"])) for r in positive)
    check_grid(cells, product(("discrete", "continuous"), REPEATED_NOISE, WINDOWS), "E uncertainty")
    if any(int(r["B"]) != 200 or r["independent_noise_repeats"] != "True" for r in positive):
        raise ValueError("Every positive-noise E summary must contain 200 independent repetitions.")
    for r in sorted(positive, key=lambda r: (r["model"] != "discrete", float(r["noise_std"]),
                                               int(r["window_size"]))):
        avg, sd = float(r["E_mean"]), float(r["E_repeat_sd"])
        rows.append([r["model"].title(), f"{float(r['noise_std']):.2f}", r["window_size"],
                     f"{avg:.6f}", f"{sd:.6f}", f"{100 * sd / avg:.3f}"])
    return rows


def validation_table(records):
    rows = [["Familywise level α", "Maximum observed\nFWER (%)",
             "Maximum upper 95%\nconfidence limit (%)", "Configurations\npassing"]]
    for alpha in (0.05, 0.01, 0.001):
        cases = [r for r in records if r["method"] == "null_calibrated_trace_Welch"
                 and r["stage"] == "2" and float(r["alpha"]) == alpha]
        configurations = {(r["scenario"], r["null_type"]) for r in cases}
        if len(cases) != 30 or len(configurations) != 30 or any(int(r["n"]) != 20000 for r in cases):
            raise ValueError(f"Expected 30 independent-validation configurations at α={alpha:g}.")
        rates = [int(r["false_rejections"]) / int(r["n"]) for r in cases]
        if any(rate != float(r["fwer"]) for rate, r in zip(rates, cases)):
            raise ValueError(f"Rejection counts disagree with stored FWER at α={alpha:g}.")
        upper = [float(r["fwer_ci_high"]) for r in cases]
        passed = sum(bound <= 1.25 * alpha and (alpha != 0.05 or float(r["coverage_ci_low"]) >= 0.94)
                     for bound, r in zip(upper, cases))
        rows.append([f"{alpha:g}", f"{100 * max(rates):.3f}", f"{100 * max(upper):.4f}",
                     f"{passed}/{len(cases)}"])
    return rows


def noise_table(records):
    rows = [["True shift (ms)", "Added-noise SNR (dB)", "Mean absolute estimated delay (ms)",
             "Mean delay RMSE (ms)"]]
    for shift in (0.0, 12.5):
        for snr in (20, -20):
            record, = [r for r in records if r["noise"] == "white" and r["mode"] == "both"
                       and int(r["delay_samples"]) >= 0 and float(r["truth_abs_ms"]) == shift
                       and float(r["snr_db"]) == snr]
            rows.append([f"{shift:.3f}", str(snr).replace("-", "−"),
                         f"{float(record['mean_abs_ms']):.3f}", f"{float(record['mean_rmse_ms']):.3f}"])
    return rows


def composition_table(original, extension):
    regions = ["Amygdala", "Hippocampus", "Middle Temporal Gyrus", "Superior Temporal Gyrus"]
    original_sessions, extended_sessions = defaultdict(set), defaultdict(set)
    for r in original:
        if int(r["N_nonzero"]) > 0:
            original_sessions[r["region"]].add(r["session"])
    for r in extension:
        if r["median_keep_artifact_ms"]:
            extended_sessions[r["region"]].add(r["session"])
    rows = [["Region comparison", "Original analysis\nSubjects / sessions",
             "Extended analysis\nSubjects / sessions"]]
    for a, b in combinations(regions, 2):
        fixed = original_sessions[a] & original_sessions[b]  # Original archive is one participant.
        shared = extended_sessions[a] & extended_sessions[b]
        subjects = {s.split("_", 1)[0] for s in shared}
        rows.append([f"{a}–{b}", f"{int(bool(fixed))} / {len(fixed)}", f"{len(subjects)} / {len(shared)}"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/supplement_tables"),
                        help="Destination directory for the supplementary CSV files.")
    args = parser.parse_args()
    # The launcher stages original benchmark tables and source inputs. The
    # archived evidence is read in place because it is not copied by 'run'.
    evidence = Path(resolve_path("repo", "examples/revision/evidence"))
    benchmark = Path(resolve_path("baseline_tables"))
    discrete = read_rows(benchmark / "simulation_results_discrete.csv")
    continuous = read_rows(benchmark / "simulation_results_continuous.csv")
    scores = evidence / "score_uncertainty"
    s3, s4 = selection_tables(read_rows(scores / "selection_by_repeat.csv"), continuous)
    tables = {
        1: benchmark_table(discrete, discrete=True), 2: benchmark_table(continuous, discrete=False),
        3: s3, 4: s4, 5: uncertainty_table(read_rows(scores / "uncertainty_summary.csv")),
        7: validation_table(read_rows(resolve_path("session_inference", "inputs/welch_validation.csv"))),
        8: noise_table(read_rows(evidence / "snr_extension" / "summary.csv")),
        9: composition_table(read_rows(resolve_path("audit_v1", "tables/trial_region_summary.csv")),
                             read_rows(resolve_path("evidence", "reviewer5/artifact_sensitivity_by_region_session.csv"))),
    }
    # Preflight every destination before creating any files. Reusing an output
    # directory is safe when its tables already match the requested results.
    pending = {}
    for number, table in tables.items():
        destination = args.output_dir / f"Table_S{number}.csv"
        buffer = StringIO(newline="")
        csv.writer(buffer).writerows(table)
        content = buffer.getvalue().encode("utf-8")
        if destination.exists():
            if destination.read_bytes() != content:
                parser.error(f"Existing table differs: {destination}. Choose a different output directory.")
        else:
            pending[destination] = content
    genes = read_rows(resolve_path("evidence", "reviewer5/correlation_new_vs_existing.csv"))
    if len(genes) != 200 or len({row["gene"] for row in genes}) != 200:
        parser.error("Expected 200 distinct ranked gene identifiers")
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["gene", "correlation"])
    writer.writerows((row["gene"], row["corr_score"]) for row in genes)
    destination = args.output_dir / "Supplementary_Data_S1_genes_top200.csv"
    content = buffer.getvalue().encode("utf-8")
    if destination.exists() and destination.read_bytes() != content:
        parser.error(f"Existing gene table differs: {destination}. Choose a different output directory.")
    if not destination.exists():
        pending[destination] = content
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for destination, content in pending.items():
        with destination.open("xb") as output:
            output.write(content)
    print("Supplementary tables " + ", ".join(f"S{n}" for n in tables) + f" in {args.output_dir}")


if __name__ == "__main__":
    main()
