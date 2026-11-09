"""Inference for differences of pooled ratio means with trial-level dependence.

Data convention: S, N have shape (..., T, 4); w has shape (T,) (inclusion weight 0/1);
``sessions`` is a list of (start, stop) slices of the trial axis. Sessions are fixed strata.

Methods
- M0  centered circular block bootstrap, absolute-error test (previous round).
- M2  studentized circular block bootstrap (bootstrap-t).
- M1  Bartlett HAC of linearized influence values, finite-sample fixed-b reference.
- M3  as M1, but the long-run variance uses within-session detrended influence values.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

REGIONS = ["Amygdala", "Hippocampus", "Middle Temporal Gyrus", "Superior Temporal Gyrus"]
SHORT = {"Amygdala": "AMY", "Hippocampus": "HIP", "Middle Temporal Gyrus": "MTG", "Superior Temporal Gyrus": "STG"}
COMPARISONS = list(combinations(range(4), 2))
COMP_NAMES = [f"{SHORT[REGIONS[a]]}-{SHORT[REGIONS[b]]}" for a, b in COMPARISONS]
N_COMP = 6
B_VALUES = [0.1, 0.2, 0.3, 0.5, 1.0]
LS = [1, 2, 5, 10]
ALPHAS = [0.05, 0.01, 0.001]
MAX_LAG = 50


def analytic_method_names():
    return [f"M1_HAC_fixedb_b{b:g}" for b in B_VALUES] + [f"M3_HACdetrend_fixedb_b{b:g}" for b in B_VALUES]


def boot_method_names():
    return [f"M0_CBB_abs_L{L}" for L in LS] + [f"M2_CBB_studentized_L{L}" for L in LS]


def pairwise(x):
    return np.stack([x[..., a] - x[..., b] for a, b in COMPARISONS], axis=-1)


# ============================================================================ estimator & influence
def pooled(S, N, w):
    """Pooled ratio per region. S, N (..., T, 4)."""
    Sw = (S * w[:, None]).sum(-2)
    Nw = (N * w[:, None]).sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = Sw / Nw
    return mu, Nw


def influence(S, N, w):
    """Linearized influence values u (..., T, 6) of the six pooled-ratio differences and delta-hat (..., 6).

    z_tr = w_t (S_tr - mu_r N_tr) / sum_t w_t N_tr ;  u_tk = z_ta - z_tb.
    Numerator, denominator and the covariance of the two regions in the same trial are
    propagated together because both regions of a trial enter one u_t.
    """
    mu, Nw = pooled(S, N, w)
    z = w[:, None] * (S - mu[..., None, :] * N) / Nw[..., None, :]
    return pairwise(z), pairwise(mu)


def _residualize(u, sessions, w, detrend):
    """Session-wise demeaned (or intercept+trend residualized) influence values; excluded trials -> 0."""
    r = np.zeros_like(u)
    for s0, s1 in sessions:
        ws = w[s0:s1] > 0
        us = u[..., s0:s1, :]
        if detrend:
            t = np.arange(s1 - s0, dtype=float)
            X = np.column_stack([np.ones(ws.sum()), t[ws]])
            P = X @ np.linalg.pinv(X)  # projection on included trials
            ui = us[..., ws, :]
            fit = np.einsum("ij,...jk->...ik", P, ui)
            rs = np.zeros_like(us)
            rs[..., ws, :] = ui - fit
        else:
            m = (us * ws[:, None]).sum(-2, keepdims=True) / max(ws.sum(), 1)
            rs = (us - m) * ws[:, None]
        r[..., s0:s1, :] = rs
    return r


def autocov_sums(r, sessions, max_lag=MAX_LAG):
    """gamma[..., j, k] = sum over sessions and t of r_t r_{t+j} (non-circular), j = 0..max_lag-1."""
    shape = r.shape[:-2] + (max_lag, r.shape[-1])
    g = np.zeros(shape)
    for s0, s1 in sessions:
        rs = r[..., s0:s1, :]
        n = s1 - s0
        for j in range(min(max_lag, n)):
            g[..., j, :] += (rs[..., : n - j, :] * rs[..., j:, :]).sum(-2)
    return g


def bartlett_weights(M):
    j = np.arange(MAX_LAG)
    wts = np.where(j < M, 1 - j / M, 0.0)
    wts[1:] *= 2
    return wts


def hac_variances(r, sessions_n, b_values=B_VALUES, sessions=None):
    """Long-run variance of the total for each b (list) -> (..., len(b), 6). Bandwidth M = b * n_session.

    All sessions in one scope have the same n here (50), so a single M per b is used.
    """
    g = autocov_sums(r, sessions)
    out = []
    for b in b_values:
        M = max(1, int(round(b * sessions_n)))
        out.append(np.einsum("j,...jk->...k", bartlett_weights(M), g))
    return np.stack(out, axis=-2)


# ============================================================================ fixed-b reference
class FixedBRef:
    """Finite-sample fixed-b reference distribution of |t| for one design.

    Stores the 100,000 body quantiles and the exact top 1% tail; p = (1 + #ref >= |t|) / (R + 1).
    """

    def __init__(self, body, tail, R):
        self.body = np.asarray(body, dtype=np.float64)  # ascending, body[i] = sorted[i * step]
        self.tail = np.asarray(tail, dtype=np.float64)  # ascending exact top values
        self.R = int(R)
        self.step = (self.R - self.tail.size) / (self.body.size)

    def count_ge(self, x):
        x = np.asarray(x, dtype=np.float64)
        n_tail = self.tail.size
        xs = np.where(np.isfinite(x), x, 0.0)
        c_tail = n_tail - np.searchsorted(self.tail, xs, side="left")
        # body: count of stored body quantiles >= x times the rank step (resolution: one step = R/1e5 draws)
        c_body = (self.body.size - np.searchsorted(self.body, xs, side="left")) * self.step + n_tail
        c = np.where(xs >= self.tail[0], c_tail, c_body).astype(np.float64)
        return np.where(np.isfinite(x), c, np.nan)

    def pvalue(self, abs_t):
        c = self.count_ge(abs_t)
        return (1.0 + c) / (self.R + 1.0)

    def quantile(self, q):
        """Quantile of |t| (q in (0,1))."""
        rank = q * self.R
        n_tail = self.tail.size
        if rank >= self.R - n_tail:
            k = int(np.clip(round(rank - (self.R - n_tail)), 0, n_tail - 1))
            return float(self.tail[k])
        k = int(np.clip(round(rank / self.step), 0, self.body.size - 1))
        return float(self.body[k])


def simulate_reference_chunk(seed, n_sessions, n_per, detrend, n_draws, b_values=B_VALUES):
    """|t| draws for i.i.d. Gaussian influence values under the given design, all b at once -> (n_draws, len(b))."""
    rng = np.random.default_rng(seed)
    T = n_sessions * n_per
    sessions = [(i * n_per, (i + 1) * n_per) for i in range(n_sessions)]
    e = rng.standard_normal((n_draws, T, 1))
    num = e.sum(-2)[..., 0]
    r = _residualize(e, sessions, np.ones(T), detrend)
    V = hac_variances(r, n_per, b_values, sessions)[..., 0]  # (n_draws, len(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.abs(num[:, None]) / np.sqrt(V)
    return t.astype(np.float64)


def build_reference(all_abs_t, n_body=100_000, tail_frac=0.01):
    x = np.sort(all_abs_t[np.isfinite(all_abs_t)])
    R = x.size
    n_tail = int(R * tail_frac)
    body_part = x[: R - n_tail]
    step = body_part.size / n_body
    idx = np.minimum((np.arange(n_body) * step).astype(int), body_part.size - 1)
    return FixedBRef(body_part[idx], x[R - n_tail:], R)


# ============================================================================ analytic methods (vectorized over sims)
def analytic_inference(S, N, w, sessions, refs, b_values=B_VALUES):
    """M1 and M3 for a batch. Returns dict with delta (..., 6), se (..., 10, 6), p (..., 10, 6),
    half95 (..., 10, 6), halfB (..., 10, 6). Method order = analytic_method_names().

    refs[(detrend, b)] -> FixedBRef for this design.
    """
    n_per = sessions[0][1] - sessions[0][0]
    u, delta = influence(S, N, w)
    ses, ps, h95, hB = [], [], [], []
    for detrend in (False, True):
        r = _residualize(u, sessions, w, detrend)
        V = hac_variances(r, n_per, b_values, sessions)  # (..., nb, 6)
        se = np.sqrt(np.maximum(V, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.abs(delta[..., None, :]) / se
        t = np.where(delta[..., None, :] == 0, 0.0, t)  # identical regions: 0/0 -> t = 0, p = 1
        for i, b in enumerate(b_values):
            ref = refs[(detrend, b)]
            ses.append(se[..., i, :])
            ps.append(ref.pvalue(t[..., i, :]))
            h95.append(ref.quantile(0.95) * se[..., i, :])
            hB.append(ref.quantile(1 - 0.05 / N_COMP) * se[..., i, :])
    st = lambda a: np.stack(a, axis=-2)  # noqa: E731
    return dict(delta=delta, se=st(ses), p=st(ps), half95=st(h95), halfB=st(hB))


# ============================================================================ bootstrap methods (one data set)
def circular_block_positions(rng, n, L, B):
    nb = -(-n // L)
    starts = rng.integers(0, n, size=(B, nb))
    pos = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    return pos.reshape(B, nb * L)[:, :n]


def stratified_indices(rng, sessions, L, B):
    return np.concatenate([s0 + circular_block_positions(rng, s1 - s0, L, B) for s0, s1 in sessions], axis=1)


def circular_bartlett_total(r, sessions, L):
    """Sum over sessions of sum_{|j|<L} (1-|j|/L) sum_t r_t r_{(t+j) mod n}; equals the CBB variance of the total."""
    V = np.zeros(r.shape[-1])
    for s0, s1 in sessions:
        rs = r[s0:s1]
        V += (rs * rs).sum(0)
        for j in range(1, L):
            V += 2 * (1 - j / L) * (rs * np.roll(rs, -j, axis=0)).sum(0)
    return V


def _circular_block_sums(X, L):
    """(n, k) -> (n, k): sum of L consecutive rows starting at each position, wrapping at n."""
    out = X.copy()
    for i in range(1, L):
        out = out + np.roll(X, -i, axis=0)
    return out


def _nanq(x, q):
    ok = np.isfinite(x)
    return np.array([np.quantile(x[ok[:, k], k], q) if ok[:, k].any() else np.nan for k in range(x.shape[1])])


def boot_inference(S, N, w, sessions, rng, B, Ls=LS, chunk=None):
    """M0 and M2 for one data set; returns dict of arrays (8, 6): p, half95, halfB, plus delta (6,), n_valid (8,6).

    Resampling draws ceil(n/L) circular block starts per session (identical draws to
    ``stratified_indices``); because every L used here divides n, a resample is the
    concatenation of whole blocks, so block sums of S and N are precomputed per start.
    The studentizing variance of a resample is computed from its own block sums of the
    linearized influence values: sum_s k/(k-1) * sum_j (T_j - mean T)^2.
    """
    Sw = S * w[:, None]
    Nw = N * w[:, None]
    u, delta = influence(S, N, w)
    p0, h0, hb0, p2, h2, hb2, nv0, nv2 = [], [], [], [], [], [], [], []
    chunk = chunk or B
    r = _residualize(u, sessions, w, False)
    for L in Ls:
        V = circular_bartlett_total(r, sessions, L)
        se = np.sqrt(np.maximum(V, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            t_obs = np.abs(delta) / se
        t_obs = np.where(delta == 0, 0.0, t_obs)
        bsS = [_circular_block_sums(Sw[s0:s1], L) for s0, s1 in sessions]
        bsN = [_circular_block_sums(Nw[s0:s1], L) for s0, s1 in sessions]
        E, TS = [], []
        done = 0
        while done < B:
            b = min(chunk, B - done)
            blocksS, blocksN, ks = [], [], []
            for i, (s0, s1) in enumerate(sessions):
                n = s1 - s0
                nb = -(-n // L)
                assert nb * L == n, "block length must divide the session length"
                starts = rng.integers(0, n, size=(b, nb))
                blocksS.append(bsS[i][starts])  # (b, nb, 4)
                blocksN.append(bsN[i][starts])
                ks.append(nb)
            totS = sum(x.sum(1) for x in blocksS)
            totN = sum(x.sum(1) for x in blocksN)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = totS / totN
                d = pairwise(mu)
                Vs = np.zeros((b, 6))
                for bS, bN, k in zip(blocksS, blocksN, ks):
                    zb = pairwise((bS - mu[:, None, :] * bN) / totN[:, None, :])  # (b, k, 6)
                    c = zb - zb.mean(1, keepdims=True)
                    Vs += (c * c).sum(1) * (k / (k - 1) if k > 1 else np.nan)
                ts = (d - delta) / np.sqrt(Vs)
                ts = np.where(d - delta == 0, 0.0, ts)
            E.append(d - delta)
            TS.append(ts)
            done += b
        e = np.concatenate(E)
        ts = np.concatenate(TS)
        ae = np.abs(e)
        ok0 = np.isfinite(ae)
        n0 = ok0.sum(0)
        cnt0 = (np.where(ok0, ae, -np.inf) >= np.abs(delta)).sum(0)
        p0.append((1 + cnt0) / (n0 + 1))
        h0.append(_nanq(ae, 0.95))
        hb0.append(_nanq(ae, 1 - 0.05 / N_COMP))
        nv0.append(n0)
        at = np.abs(ts)
        ok2 = np.isfinite(at)
        n2 = ok2.sum(0)
        cnt2 = (np.where(ok2, at, -np.inf) >= t_obs).sum(0)
        p2.append(np.where(np.isfinite(t_obs), (1 + cnt2) / (n2 + 1), np.nan))
        h2.append(_nanq(at, 0.95) * se)
        hb2.append(_nanq(at, 1 - 0.05 / N_COMP) * se)
        nv2.append(n2)
    st = np.stack
    return dict(delta=delta, p=st(p0 + p2), half95=st(h0 + h2), halfB=st(hb0 + hb2), n_valid=st(nv0 + nv2))
