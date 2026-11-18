"""Null and partial-null generating models for the trial-level pooled ratio comparison.

Every model generates trial-level region means y_tr and sets S_tr = N_tr * y_tr. The comparison
target is sum E(S) / sum E(N) per region; all session offsets and trends are centered with the
N weights of the pooled ratio, so the target differences equal the injected effects exactly.
Observed differences are never forced to zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infer import REGIONS, pairwise

N_PER = 50
SESSIONS = [(0, 50), (50, 100), (100, 150)]
BASE = 9.0

SCENARIOS = {
    "A": "Independent Gaussian trials, equal means and four different regional variances",
    "B1": "Short-range AR(1) phi=0.3 with observed cross-region correlations",
    "B2": "Short-range AR(1) phi=0.6 with observed cross-region correlations",
    "C1": "Session offsets, session-specific SDs, diagonal VAR(1), observed correlations and observed N",
    "C2": "C1 with random N resampled by row and associated with the signal, preserving the ratio estimand",
    "D1": "Stress: longer dependence at phi=0.85 with C1 structure",
    "D2": "Stress: approximate long memory, AR(0.5) plus slow AR(0.97) (40% of variance)",
    "D3": "Stress: within-session linear trend at observed slope scale, centred with N weights",
    "D4": "Stress: t(3) innovations and shared trial-level heteroscedastic scale",
    "D5": "Stress: two outlying trials per session with independent ±6 SD regional deviations",
}
CORE = ["A", "B1", "B2", "C1", "C2"]
NULL_TYPES = {
    "G": ("Global null", np.zeros(4)),
    "P1": ("Partial null: hippocampus +0.7 ms", np.array([0.0, 0.7, 0.0, 0.0])),
    "P2": ("Partial null: MTG=STG=+0.3 ms, AMY=HIP", np.array([0.0, 0.0, 0.3, 0.3])),
}


def true_null_mask(ntype):
    d = pairwise(NULL_TYPES[ntype][1])
    return np.isclose(d, 0.0)


def estimate_params(trs: pd.DataFrame) -> dict:
    """Generating-model parameters from the real trial x region table."""
    y = trs.pivot_table(index=["session", "trial_pos"], columns="region", values="mean_nonzero_ms")[REGIONS]
    N = trs.pivot_table(index=["session", "trial_pos"], columns="region", values="N_nonzero")[REGIONS]
    ses = sorted(y.index.get_level_values(0).unique())
    m = np.stack([y.loc[s].mean().to_numpy() for s in ses])
    sd = np.stack([y.loc[s].std(ddof=1).to_numpy() for s in ses])
    res = y - y.groupby(level=0).transform("mean")
    R = res.corr().to_numpy()
    phi = []
    for r in REGIONS:
        num = den = 0.0
        for s in ses:
            x = res.loc[s][r].to_numpy()
            num += (x[1:] * x[:-1]).sum()
            den += (x * x).sum()
        phi.append(num / den)
    phi = np.clip(np.array(phi), -0.5, 0.9)
    slope_N, trend = [], np.zeros((3, 4))
    for j, r in enumerate(REGIONS):
        num = den = 0.0
        for i, s in enumerate(ses):
            yn = res.loc[s][r].to_numpy()
            nn = N.loc[s][r].to_numpy() - N.loc[s][r].mean()
            num += (yn * nn).sum()
            den += (nn * nn).sum()
            t = np.arange(N_PER) - (N_PER - 1) / 2
            trend[i, j] = (yn * t).sum() / (t * t).sum()
        slope_N.append(num / den if den > 0 else 0.0)
    return dict(m=m, sd=sd, R=R, phi=phi, slope_N=np.array(slope_N), trend=trend,
                N_real=N.to_numpy().astype(float), sessions=ses)


def _ar(rng, nsim, n, phi, chol, innov="normal"):
    """Unit-variance stationary AR(1) per region with correlated innovations -> (nsim, n, 4)."""
    k = chol.shape[0]
    if innov == "t3":
        z = rng.standard_t(3, size=(nsim, n, k)) / np.sqrt(3.0)
    else:
        z = rng.standard_normal((nsim, n, k))
    eta = z @ chol.T
    e = np.empty_like(eta)
    e[:, 0] = eta[:, 0]
    a = np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        e[:, t] = phi * e[:, t - 1] + a * eta[:, t]
    return e


def _center(o, wts):
    """Remove the weighted session mean per region so sum_s wts_sr * o_sr = 0."""
    return o - (wts * o).sum(0, keepdims=True) / wts.sum(0, keepdims=True)


def generate(scen: str, ntype: str, rng: np.random.Generator, nsim: int, P: dict):
    """Returns S, N (nsim, 150, 4) and the true differences (6,)."""
    delta = NULL_TYPES[ntype][1]
    truth = pairwise(delta)
    Nr = P["N_real"]
    T = Nr.shape[0]
    R = P["R"] if scen != "A" else np.eye(4)
    chol = np.linalg.cholesky(R)
    sd_mean = P["sd"].mean(0)
    y = np.empty((nsim, T, 4))
    Nout = np.broadcast_to(Nr, (nsim, T, 4)).copy()
    Ntot = np.stack([Nr[s0:s1].sum(0) for s0, s1 in SESSIONS])  # (3,4)
    if scen in ("A", "B1", "B2"):
        phi = {"A": 0.0, "B1": 0.3, "B2": 0.6}[scen] * np.ones(4)
        for s0, s1 in SESSIONS:
            e = _ar(rng, nsim, s1 - s0, phi, chol)
            y[:, s0:s1] = BASE + delta + sd_mean * e
        return y * Nout, Nout, truth
    # C/D family
    if scen == "C2":
        EN = np.stack([Nr[s0:s1].mean(0) for s0, s1 in SESSIONS])
        VN = np.stack([Nr[s0:s1].var(0) for s0, s1 in SESSIONS])
        o = _center(P["m"] - P["m"].mean(0), EN)
        c = P["slope_N"]
        for i, (s0, s1) in enumerate(SESSIONS):
            rows = rng.integers(s0, s1, size=(nsim, s1 - s0))
            Nout[:, s0:s1] = Nr[rows]
            sd_eps = np.sqrt(np.maximum(P["sd"][i] ** 2 - c ** 2 * VN[i], 0.25 * P["sd"][i] ** 2))
            e = _ar(rng, nsim, s1 - s0, P["phi"], chol)
            Ns = Nout[:, s0:s1]
            y[:, s0:s1] = (BASE + delta + o[i] + sd_eps * e + c * (Ns - EN[i]) - c * VN[i] / EN[i])
        return y * Nout, Nout, truth
    o = _center(P["m"] - P["m"].mean(0), Ntot)
    for i, (s0, s1) in enumerate(SESSIONS):
        n = s1 - s0
        sd = P["sd"][i]
        if scen == "D1":
            e = _ar(rng, nsim, n, 0.85 * np.ones(4), chol)
        elif scen == "D2":
            e = np.sqrt(0.6) * _ar(rng, nsim, n, 0.5 * np.ones(4), chol) + \
                np.sqrt(0.4) * _ar(rng, nsim, n, 0.97 * np.ones(4), chol)
        elif scen == "D4":
            e = _ar(rng, nsim, n, P["phi"], chol, innov="t3")
            scale = np.exp(0.5 * rng.standard_normal((nsim, n, 1))) / np.exp(0.25)
            e = e * scale
        else:  # C1, D3, D5
            e = _ar(rng, nsim, n, P["phi"], chol)
        yy = BASE + delta + o[i] + sd * e
        if scen == "D3":
            t = np.arange(n, dtype=float)
            Ns = Nr[s0:s1]
            tbar = (Ns * t[:, None]).sum(0) / Ns.sum(0)
            yy = yy + P["trend"][i] * (t[:, None] - tbar)
        if scen == "D5":
            for k in range(nsim):
                tt = rng.choice(n, size=2, replace=False)
                sign = rng.choice([-1.0, 1.0], size=(2, 4))
                yy[k, tt] += 6 * sd * sign
        y[:, s0:s1] = yy
    return y * Nout, Nout, truth
