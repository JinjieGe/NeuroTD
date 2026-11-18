"""Reproduce the final R2.4.2 fixed-session, calibrated Welch-type contrasts.

The default recomputes the six comparisons from the packaged trial-region
sums and counts with the frozen SE multiplier. Pass --repeat-calibration to
refit the multiplier in 200,000 null datasets, validate it in 600,000 independent
datasets and run two additional 20,000-dataset challenges. The sampling target
conditions on this participant's three recorded sessions and contacts. Report
adjusted p < 0.001 as a threshold, the smallest family level validated.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import stats

from genmodels import SCENARIOS, estimate_params
from infer import COMP_NAMES, N_COMP, REGIONS, influence

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ('ses-01', 'ses-02', 'ses-03')
TRIALS = 50
PHI_GRID = np.linspace(-0.8, 0.98, 1781)
FROZEN_MULTIPLIER = 1.7519988157717314
ALPHAS = np.array([0.001, 0.01, 0.05])
SEED = 202611120242
DRAWS = 20_000
BATCH = 500  # Part of the original RNG stream layout.


def read_trials(path):
    frame = pd.read_csv(path)
    keys = ['session', 'trial_pos', 'region']
    required = set(keys + ['S_nonzero_ms', 'N_nonzero', 'mean_nonzero_ms'])
    if missing := required.difference(frame.columns):
        raise ValueError(f'Missing trial table columns: {sorted(missing)}')
    expected = pd.MultiIndex.from_product([SESSIONS, range(TRIALS), REGIONS], names=keys)
    observed = frame.set_index(keys).index
    if not observed.is_unique or len(observed) != len(expected) or len(expected.difference(observed)):
        raise ValueError('Expected one row per region at trial positions 0–49 in ses-01, ses-02 and ses-03.')
    aligned = frame.set_index(keys).reindex(expected)
    sums = aligned.S_nonzero_ms.to_numpy(float).reshape(150, 4)
    counts = aligned.N_nonzero.to_numpy(float).reshape(150, 4)
    if not (np.isfinite(sums).all() and np.isfinite(counts).all() and (sums > 0).all()
            and (counts > 0).all() and np.equal(counts, np.floor(counts)).all()):
        raise ValueError('Expected positive finite absolute-delay sums and positive integer nonzero counts.')
    means = aligned.mean_nonzero_ms.to_numpy(float).reshape(150, 4)
    if not np.allclose(means, sums / counts, rtol=1e-12, atol=1e-14):
        raise ValueError('Trial-region means are inconsistent with their sums and counts.')
    return frame, sums, counts


@lru_cache(maxsize=1)
def trace_table():
    """Trace moments of the centered Bartlett quadratic form on the frozen AR(1) grid."""
    lag = np.abs(np.arange(TRIALS)[:, None] - np.arange(TRIALS)[None, :])
    center = np.eye(TRIALS) - np.ones((TRIALS, TRIALS)) / TRIALS
    quadratic = center @ (1 - lag / TRIALS) @ center
    table = []
    for phi in PHI_GRID:
        correlation = phi ** lag
        product = quadratic @ correlation
        table.append((np.trace(product), np.sum(product * product.T), correlation.sum(),
                      np.trace(center @ correlation) / (TRIALS - 1)))
    return np.array(table)


def fit(sums, counts):
    """Return contrast, analytic SE and effective df for each pooled-ratio comparison."""
    trial_influence, difference = influence(sums, counts, np.ones(len(SESSIONS) * TRIALS))
    by_session = trial_influence.reshape((*difference.shape[:-1], len(SESSIONS), TRIALS, N_COMP))
    residuals = by_session - by_session.mean(axis=-2, keepdims=True)
    squares = (residuals * residuals).sum(axis=-2)
    energy = squares.sum(axis=-2)
    if np.any(energy <= 0):
        raise ValueError('A contrast has no within-session variation to estimate its uncertainty.')
    lag_one = (residuals[..., 1:, :] * residuals[..., :-1, :]).sum(axis=(-3, -2))
    raw_phi = lag_one / energy
    phi = np.clip(raw_phi + (1 + 3 * raw_phi) / TRIALS, PHI_GRID[0], PHI_GRID[-1])
    first, second, total, centered = (np.interp(phi, PHI_GRID, column) for column in trace_table().T)
    scales = squares / (TRIALS - 1) / centered[..., None, :]
    effective_df = (first ** 2 / second) * (scales.sum(axis=-2) ** 2 / (scales ** 2).sum(axis=-2))
    # Full-bandwidth Bartlett variance equals 2/n times squared within-session partial sums.
    raw_variance = (2 / TRIALS) * (np.cumsum(residuals, axis=-2) ** 2).sum(axis=(-3, -2))
    standard_error = np.sqrt((total / first) * raw_variance)
    if not (np.isfinite(standard_error).all() and (standard_error > 0).all()
            and np.isfinite(effective_df).all() and (effective_df > 0).all()):
        raise ValueError('Invalid Welch-type variance or effective degrees of freedom.')
    return difference, standard_error, effective_df


def comparisons(sums, counts, multiplier):
    difference, analytic_se, effective_df = fit(sums, counts)
    calibrated_se = multiplier * analytic_se
    two_sided_family = 2 * N_COMP
    half_width = stats.t.isf(0.05 / two_sided_family, effective_df) * calibrated_se
    adjusted_p = np.minimum(1, two_sided_family * stats.t.sf(np.abs(difference) / calibrated_se, effective_df))
    return pd.DataFrame(dict(comparison=COMP_NAMES, difference_ms=difference, se_analytic_ms=analytic_se,
                             effective_df=effective_df, se_factor_null_calibration=multiplier,
                             se_calibrated_ms=calibrated_se, p_adj_calibrated=adjusted_p,
                             simultaneous95_low_ms=difference - half_width,
                             simultaneous95_high_ms=difference + half_width))


def write_table_s6(computed, path):
    """Format the six recomputed contrasts in the supplement's order and directions."""
    comparisons_by_name = computed.set_index('comparison')
    order = ['HIP-AMY', 'HIP-MTG', 'HIP-STG', 'AMY-MTG', 'AMY-STG', 'MTG-STG']
    rows = [['Comparison', 'Mean difference\n(ms)', 'Calibrated SE\n(ms)', 'Approx.\ndf',
             'Simultaneous\n95% interval (ms)', 'Adjusted p']]
    for name in order:
        # The statistical output uses AMY-HIP. Reverse both endpoints for HIP-AMY.
        key = 'AMY-HIP' if name == 'HIP-AMY' else name
        row = comparisons_by_name.loc[key]
        difference = row.difference_ms
        low, high = row.simultaneous95_low_ms, row.simultaneous95_high_ms
        if key != name:
            difference, low, high = -difference, -high, -low
        p = row.p_adj_calibrated
        rows.append([name.replace('-', '–'), f'{difference:.3f}', f'{row.se_calibrated_ms:.4f}',
                     f'{row.effective_df:.2f}', f'[{low:.3f}, {high:.3f}]',
                     '<0.001' if p < 0.001 else f'{p:#.3g}'])

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        generated = Path(directory) / path.name
        with generated.open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerows(rows)
        try:
            path.hardlink_to(generated)
        except FileExistsError:
            if path.read_bytes() != generated.read_bytes():
                raise FileExistsError(f'Existing table differs: {path}. Choose a different output directory.') from None


def simulation_batches(stage, scenario, null_type, parameters):
    # The published earlier study provides the same ten generators and the two challenge models.
    from run_analysis import sample

    scenarios = [*SCENARIOS, 'E1', 'E2']
    null_types = ['G', 'P1', 'P2']
    rng = np.random.default_rng([SEED, stage, scenarios.index(scenario), null_types.index(null_type)])
    for _ in range(DRAWS // BATCH):
        sums, counts, truth = sample(scenario, null_type, rng, BATCH, parameters)
        difference, standard_error, effective_df = fit(sums, counts)
        critical = stats.t.isf(ALPHAS[None, :, None] / (2 * N_COMP), effective_df[:, None, :])
        yield difference, truth, standard_error, critical


def repeat_calibration(frame, outdir):
    parameters = estimate_params(frame)
    factors = []
    for scenario in SCENARIOS:
        samples = []
        for difference, truth, se, critical in simulation_batches(1, scenario, 'G', parameters):
            error = np.abs((difference - truth) / se)
            samples.append(np.max(error[:, None, :] / critical, axis=-1))
        ordered = np.sort(np.concatenate(samples), axis=0)
        for index, alpha in enumerate(ALPHAS):
            rank = int(stats.binom.ppf(0.95, DRAWS, 1 - alpha)) + 1
            factors.append(dict(scenario=scenario, alpha=alpha, n=DRAWS, order_k=rank,
                                se_factor=ordered[rank - 1, index]))
    calibration = pd.DataFrame(factors)
    multiplier = max(1.0, calibration.se_factor.max())
    calibration.to_csv(outdir / 'welch_calibration_factors.csv', index=False)

    from run_analysis import cp

    validation = []
    for scenario in [*SCENARIOS, 'E1', 'E2']:
        stage = 2 if scenario in SCENARIOS else 3
        for null_type in (['G', 'P1', 'P2'] if stage == 2 else ['G']):
            false_rejections = np.zeros(len(ALPHAS), dtype=int)
            covered = np.zeros(len(ALPHAS), dtype=int)
            for difference, truth, se, critical in simulation_batches(stage, scenario, null_type, parameters):
                null = np.isclose(truth, 0, atol=1e-12)
                threshold = multiplier * critical
                tested = np.abs(difference / se)[:, None, :]
                error = np.abs((difference - truth) / se)[:, None, :]
                false_rejections += (tested[:, :, null] > threshold[:, :, null]).any(axis=-1).sum(axis=0)
                covered += (error <= threshold).all(axis=-1).sum(axis=0)
            for alpha, rejected, coverage in zip(ALPHAS, false_rejections, covered):
                low, high = cp(int(rejected), DRAWS)
                validation.append(dict(scenario=scenario, null_type=null_type, stage=stage, alpha=alpha,
                                       n=DRAWS, false_rejections=rejected, fwer=rejected / DRAWS,
                                       fwer_ci_low=low, fwer_ci_high=high, fwer_gate=high <= 1.25 * alpha,
                                       simultaneous_coverage=coverage / DRAWS))
    pd.DataFrame(validation).to_csv(outdir / 'welch_validation_and_challenges.csv', index=False)
    return multiplier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat-calibration', action='store_true', help='Run all 840,000 simulated datasets.')
    parser.add_argument('--table-dir', type=Path, default=ROOT / 'outputs', help='Directory for Table_S6.csv.')
    args = parser.parse_args()
    frame, sums, counts = read_trials(ROOT / 'inputs/trial_region_summary.csv')
    outdir = ROOT / 'outputs'
    outdir.mkdir(exist_ok=True)
    multiplier = repeat_calibration(frame, outdir) if args.repeat_calibration else FROZEN_MULTIPLIER
    source = 'refitted' if args.repeat_calibration else 'frozen'
    table = comparisons(sums, counts, multiplier)
    destination = outdir / 'calibrated_welch_six_comparisons.csv'
    table_path = args.table_dir / 'Table_S6.csv'
    try:
        write_table_s6(table, table_path)
    except FileExistsError as error:
        parser.error(str(error))
    table.to_csv(destination, index=False)
    print(f'Six fixed-session comparisons with {source} multiplier {multiplier:.16g}: {destination}')
    print(f'Supplementary Table S6: {table_path}')


if __name__ == '__main__':
    main()
