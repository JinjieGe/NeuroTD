"""Ground-truth audit of the existing simulations; never changes an estimator.

E = sqrt(mean(r_k)) is the existing implementation's alignment diagnostic.
Delay RMSE uses known simulated delays, not alignment residuals. Oracle selection
is a retrospective benchmark, not a deployable ground-truth-free selector.
"""

from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("NeuroTD", "DTW", "CTW", "FFT")


def delay_metrics(estimated, truth, sampling_rate):
    """Evaluate same-center delays in samples; fail rather than drop invalid data."""
    estimated, truth = np.asarray(estimated, float), np.asarray(truth, float)
    if estimated.ndim != 1 or estimated.size == 0 or estimated.shape != truth.shape:
        raise ValueError("Estimated and true delays must be nonempty equal-length vectors")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(truth)):
        raise ValueError("Nonfinite delay estimates or truth")
    error = estimated - truth
    factor = 1000.0 / sampling_rate
    return {
        "n_centers": estimated.size,
        "delay_rmse_ms": float(np.sqrt(np.mean(error**2)) * factor),
        "delay_mae_ms": float(np.mean(np.abs(error)) * factor),
        "delay_bias_ms": float(np.mean(error) * factor),
    }


class KnownDelayAudit:
    """Collect native-center trajectories, then independently score window selection."""

    def __init__(self, simulation, seed, sampling_rate):
        self.simulation = simulation
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.frames = []
        self.rows = []

    def record(self, noise, window, hop, alignment_error_E, traces):
        """traces maps method to (centers, estimated_samples, true_samples)."""
        if set(traces) != set(METHODS):
            raise ValueError("All four methods must be supplied")
        if not np.isfinite(alignment_error_E) or alignment_error_E < 0:
            raise ValueError("E must be finite and nonnegative")
        if not np.isfinite(noise) or noise < 0 or window < 3 or window % 2 != 1 or hop < 1:
            raise ValueError("Invalid noise, odd window size, or hop")
        if any(r["noise_std"] == noise and r["window_size"] == window for r in self.rows):
            raise ValueError("Duplicate noise/window combination")
        for method in METHODS:
            centers, estimated, truth = (np.asarray(v) for v in traces[method])
            metrics = delay_metrics(estimated, truth, self.sampling_rate)
            if (centers.shape != estimated.shape or not np.all(np.isfinite(centers))
                    or np.any(np.diff(centers) <= 0) or np.any(centers != centers.astype(int))):
                raise ValueError("Centers must be unique increasing integer indices")
            common = {
                "simulation": self.simulation, "seed": self.seed,
                "noise_std": float(noise), "window_size": int(window),
                "hop_size": int(hop), "method": method,
            }
            self.frames.append(pd.DataFrame({
                **common, "center_index": centers.astype(int),
                "center_time_s": centers / self.sampling_rate,
                "estimated_delay_samples": estimated, "true_delay_samples": truth,
                "error_samples": estimated - truth,
            }))
            self.rows.append({
                **common, **metrics,
                "alignment_error_E": float(alignment_error_E) if method == "NeuroTD" else np.nan,
                "true_outside_neurotd_search_fraction": (
                    float(np.mean(np.abs(truth) > window // 2)) if method == "NeuroTD" else np.nan
                ),
            })

    def tables(self):
        if not self.rows:
            raise ValueError("No simulation results recorded")
        candidates = pd.DataFrame(self.rows)
        traces = pd.concat(self.frames, ignore_index=True)
        # Secondary audit: exactly the same center indices for every W and method.
        # Do not interpolate, extrapolate, or pretend unequal native centers are equal.
        candidates["n_shared_centers"] = 0
        candidates["delay_rmse_shared_ms"] = np.nan
        for noise, group in traces.groupby("noise_std"):
            groups = list(group.groupby(["window_size", "method"]))
            shared = set(groups[0][1].center_index)
            for _, frame in groups[1:]:
                shared.intersection_update(frame.center_index)
            for (window, method), frame in groups:
                mask = ((candidates.noise_std == noise) & (candidates.window_size == window)
                        & (candidates.method == method))
                candidates.loc[mask, "n_shared_centers"] = len(shared)
                if shared:
                    aligned = frame[frame.center_index.isin(shared)]
                    metrics = delay_metrics(aligned.estimated_delay_samples,
                                            aligned.true_delay_samples, self.sampling_rate)
                    candidates.loc[mask, "delay_rmse_shared_ms"] = metrics["delay_rmse_ms"]
        selection = compare_selection(candidates)
        return candidates, traces, selection

    def write(self, output_directory):
        output = Path(output_directory)
        # Fresh directory prevents stale or overwritten evidence.
        output.mkdir(parents=True, exist_ok=False)
        candidates, traces, selection = self.tables()
        candidates.to_csv(output / "candidate_metrics.csv", index=False)
        traces.to_csv(output / "delay_traces.csv.gz", index=False, compression="gzip")
        selection.to_csv(output / "window_selection.csv", index=False)
        self._plot(candidates, selection, output)
        native = selection[selection.evaluation == "native_centers"]
        report = (
            f"# Known-delay validation: {self.simulation}, seed {self.seed}\n\n"
            f"Sampling rate: {self.sampling_rate} Hz. "
            f"Noise levels: {candidates.noise_std.nunique()}; "
            f"candidate windows: {sorted(candidates.window_size.unique().tolist())}.\n\n"
            f"E-selected window is RMSE-optimal (including numerical ties) at "
            f"{int(native.E_choice_is_rmse_optimal.sum())}/{len(native)} noise levels. "
            f"Maximum excess delay RMSE: {native.excess_rmse_ms.max():.9g} ms.\n\n"
            "E is sqrt(mean(r_k)), NOT manuscript q and NOT delay RMSE. "
            "E selection uses no ground truth; the oracle and delay errors do. "
            "All four methods are evaluated at every fixed candidate W. "
            "Only NeuroTD has E-based selection in this audit.\n\n"
            "Native-center scores preserve the original scripts' evaluation. "
            "Shared-center scores use the exact intersection across ALL candidate "
            "windows and methods; check n_shared_centers, which can be small or zero. "
            "This does not equalize the estimators' search context or fix baselines.\n\n"
            "Ground truth is the original scripts' convention: nearest segment "
            "center for discrete delays; generating delay sampled at the reported "
            "center for continuous delays. For y(t)=x(t-Delta(t)), this latter "
            "quantity is NOT generally identical to the inverse-map displacement "
            "at a reference-x time. This audit preserves rather than resolves "
            "that modeling convention.\n\n"
            "Exact minima use unrounded values; ties choose the smallest W. "
            "RMSE-optimality flag uses 1e-12 ms absolute / 1e-10 relative tolerance. "
            "Undefined ratios (zero oracle RMSE) are blank, not infinity. "
            "Noise levels and overlapping centers are not independent replicates; "
            "no p-values or confidence intervals are claimed. Repeat with independent "
            "seeds and predeclared signal settings before generalizing. "
            "Known simulation truth is an external error target, not an independent "
            "held-out dataset or evidence of biological validity.\n"
        )
        (output / "summary.md").write_text(report, encoding="utf-8")
        print(f"Known-delay validation saved to {output.resolve()}")
        return selection

    @staticmethod
    def _plot(candidates, selection, output):
        from matplotlib import pyplot as plt

        # Original figure scripts set large global fonts; keep audit plots legible.
        with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                             "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.titlesize": 12}):
            KnownDelayAudit._plot_figures(candidates, selection, output)

    @staticmethod
    def _plot_figures(candidates, selection, output):
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for evaluation, linestyle in (("native_centers", "-"), ("shared_centers", "--")):
            frame = selection[selection.evaluation == evaluation]
            axes[0].plot(frame.noise_std, frame.E_selected_rmse_ms, linestyle,
                         marker="o", label=f"E-selected ({evaluation})")
            axes[0].plot(frame.noise_std, frame.oracle_rmse_ms, linestyle,
                         marker="x", label=f"Truth oracle ({evaluation})")
            axes[1].plot(frame.noise_std, frame.excess_rmse_ms, linestyle,
                         marker="o", label=evaluation)
        axes[0].set_ylabel("Delay RMSE against known truth (ms)")
        axes[1].set_ylabel("Excess RMSE over truth oracle (ms)")
        for ax in axes:
            ax.set_xlabel("Noise standard deviation")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("NeuroTD window-selection validation (not a confidence score)")
        fig.savefig(output / "selection_vs_truth.png", dpi=160)
        plt.close(fig)
        windows = sorted(candidates.window_size.unique())
        fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4),
                                 squeeze=False, constrained_layout=True)
        for ax, window in zip(axes[0], windows):
            for method in METHODS:
                frame = candidates[(candidates.window_size == window) & (candidates.method == method)]
                ax.plot(frame.noise_std, frame.delay_rmse_ms, marker=".", label=method)
            ax.set(title=f"Fixed W={window}", xlabel="Noise standard deviation",
                   ylabel="Delay RMSE (ms; native centers)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.savefig(output / "four_methods_known_delay_rmse.png", dpi=160)
        plt.close(fig)


def compare_selection(candidates):
    """Choose W from E ONLY, then evaluate it against known-delay oracle scores."""
    results = []
    neurotd = candidates[candidates.method == "NeuroTD"]
    for noise, frame in neurotd.groupby("noise_std", sort=True):
        frame = frame.sort_values("window_size")
        # Deliberately no RMSE tie-breaker in E selection: no truth leakage.
        selected = frame.loc[frame.alignment_error_E.idxmin()]
        for evaluation, column in (("native_centers", "delay_rmse_ms"),
                                   ("shared_centers", "delay_rmse_shared_ms")):
            if frame[column].isna().all():
                continue
            oracle = frame.loc[frame[column].idxmin()]
            chosen, best = float(selected[column]), float(oracle[column])
            results.append({
                "simulation": selected.simulation, "seed": selected.seed,
                "noise_std": noise, "evaluation": evaluation,
                "E_selected_window": int(selected.window_size),
                "E_selected_value": selected.alignment_error_E,
                "oracle_window": int(oracle.window_size),
                "E_selected_rmse_ms": chosen, "oracle_rmse_ms": best,
                "excess_rmse_ms": chosen - best,
                "rmse_ratio": chosen / best if best > 0 else np.nan,
                "same_window": bool(selected.window_size == oracle.window_size),
                "E_choice_is_rmse_optimal": bool(np.isclose(chosen, best, rtol=1e-10, atol=1e-12)),
                "selected_n_centers": int(selected.n_centers if evaluation == "native_centers"
                                          else selected.n_shared_centers),
                "oracle_n_centers": int(oracle.n_centers if evaluation == "native_centers"
                                        else oracle.n_shared_centers),
            })
    return pd.DataFrame(results)
