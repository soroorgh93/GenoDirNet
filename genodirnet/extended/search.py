"""Architecture searches over the design space of the sequence network (extended schedule).

Shared by scripts/genodirnet2/search_first.py and search_second.py:
  ArchitecturePolicy   independent categorical distributions, one per design decision (learned sampler)
  zero_cost_proxies    Fisher information of the loss with respect to the stage activations, gradient norm, SNIP,
                       SynFlow and NASWOT of the untrained network on two fixed mini-batches
  short_train          two-epoch training on a fixed subset of the training transcripts; objective = mean of the
                       validation AUROC on complete transcripts and on one 300-nucleotide window per transcript,
                       with its DeLong standard error
  info_proxies         quantities of the Fisher geometry from the Gram matrix of per-example margin gradients on
                       160 probe windows, at initialisation and after 50 training steps
  GraphGP              Gaussian process over the design graph and the proxies, used with expected improvement

Notation of the Fisher geometry. For n probe examples let a_i be the gradient of the logit margin z_i with respect
to all parameters, K = A A^T the n x n Gram matrix, p_i = sigmoid(z_i), w_i = p_i (1 - p_i) and r_i = p_i - y_i.
Fisher information F = (1/n) sum_i w_i a_i a_i^T; empirical Fisher J = (1/n) sum_i r_i^2 a_i a_i^T; Takeuchi trace
tr(J (F + delta I)^-1). All quantities reduce to n x n algebra on K. One damped natural-gradient step fitted on one
half of the probes and read out on the other half (fk_ngd) is the Fisher-kernel classifier; ntk_gd is the plain
gradient step.
"""
import math
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg, optimize, stats
from torch.utils.data import DataLoader

from ..metrics import delong_variance
from .data import TranscriptDataset, make_test_windows, worker_init
from .models import SPACE, build_two_branch, count_macs_reader, count_parameters, spec_name, vector_to_spec

PROXIES = ["fisher", "grad_norm", "snip", "synflow", "naswot", "params"]
BASE = PROXIES + ["mmac"]
GEO_INIT = ["fk_ngd", "ntk_gd", "lga", "fisher_tr", "fisher_pr", "fisher_logdet", "ntk_cond", "fisher_effdim"]
GEO_WARM = ["warm_fk_ngd", "warm_ntk_gd", "warm_lga", "warm_tic", "warm_godambe_ratio", "warm_fisher_effdim", "warm_fisher_tr"]
WARM = ["warm_auroc", "tse"]
FEATS = BASE + GEO_INIT + GEO_WARM + WARM
ALL_Q = FEATS + ["tic", "godambe_ratio", "warm_fisher_pr", "warm_fisher_logdet", "warm_ntk_cond"]
DEC = list(SPACE.keys())


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- learned sampler
class ArchitecturePolicy(nn.Module):
    """Independent categorical distributions, one per decision of the design space."""

    def __init__(self):
        super().__init__()
        self.logits = nn.ParameterList([nn.Parameter(torch.zeros(len(v))) for v in SPACE.values()])

    def sample(self, n, rng):
        specs, vecs = [], []
        with torch.no_grad():
            probs = [torch.softmax(l.double(), 0).numpy() for l in self.logits]
        probs = [p / p.sum() for p in probs]
        for _ in range(n):
            vec = [int(rng.choice(len(p), p=p)) for p in probs]
            vecs.append(vec); specs.append(vector_to_spec(vec))
        return specs, vecs

    def log_prob(self, vec):
        return sum(F.log_softmax(l, 0)[i] for l, i in zip(self.logits, vec))

    def entropy(self):
        return sum(torch.distributions.Categorical(logits=l).entropy() for l in self.logits)

    def marginals(self):
        return {k: dict(zip(map(str, v), torch.softmax(l, 0).tolist())) for (k, v), l in zip(SPACE.items(), self.logits)}


def within_budget(spec, max_params, max_mmac):
    m = build_two_branch(spec)
    p = count_parameters(m); mm = count_macs_reader(spec) / 1e6
    return (p <= max_params and mm <= max_mmac), p, mm


# ----------------------------------------------------------------------------- proxies at initialisation
def binary_codes(acts):
    return torch.cat([(a > 0).flatten(1).float() for a in acts], dim=1)


def zero_cost_proxies(spec, batches, dev):
    """Training-free scores of an architecture at random initialisation (mean over the batches)."""
    torch.manual_seed(0)
    model = build_two_branch(spec).to(dev); model.train()
    enc = model.seq.encoder; enc.record_acts = True
    out = {k: 0.0 for k in PROXIES}
    out["params"] = float(count_parameters(model))
    for x, m, f, y in batches:
        x, m, f, y = x.to(dev), m.to(dev), f.to(dev), y.to(dev)
        model.zero_grad(set_to_none=True)
        logits = model(x, f, m)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        fisher = 0.0
        for a in enc._acts:
            if a.grad is None: continue
            fisher += 0.5 * ((a * a.grad).sum(2) ** 2).sum(1).mean().item()      # per-channel Fisher (Theis), summed
        out["fisher"] += fisher
        gn, snip = 0.0, 0.0
        for p in model.parameters():
            if p.grad is not None:
                gn += p.grad.pow(2).sum().item(); snip += (p * p.grad).abs().sum().item()
        out["grad_norm"] += math.sqrt(gn); out["snip"] += snip
        # NASWOT: log-determinant of the Hamming kernel of binary activation codes
        with torch.no_grad():
            codes = binary_codes(enc._acts)
            K = codes @ codes.t() + (1 - codes) @ (1 - codes).t()
            out["naswot"] += torch.slogdet(K.double() + 1e-3 * torch.eye(K.shape[0], device=dev, dtype=torch.double))[1].item()
    for k in ("fisher", "grad_norm", "snip", "naswot"):
        out[k] /= len(batches)
    # SynFlow: all-ones input, |weights|, sum of theta * dR/dtheta (data-free; eval mode so that BN is the identity)
    sf = build_two_branch(spec).to(dev).double().eval()
    with torch.no_grad():
        for p in sf.parameters(): p.abs_()
    x1 = torch.ones(1, 4, batches[0][0].shape[2], device=dev, dtype=torch.double); f1 = torch.ones(1, 5, device=dev, dtype=torch.double)
    sf.zero_grad(); sf(x1, f1).sum().backward()
    out["synflow"] = float(sum((p * p.grad).sum().item() for p in sf.parameters() if p.grad is not None))
    out["synflow"] = math.log(max(out["synflow"], 1e-300))
    enc.record_acts = False
    del model, sf
    return out


# ----------------------------------------------------------------------------- short training
def make_val_sets(val, n_val, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(val))[:n_val]; sub = [val[i] for i in idx]
    w = make_test_windows(sub, lengths=(300,), half=False, seed=seed)["w300"]
    return sub, w


@torch.no_grad()
def score_rows(model, rows, table, mean, std, dev, workers):
    ds = TranscriptDataset(rows, table, mean, std, fragment_aware=False, error_aware=False)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=workers)
    out, ys = [], []
    model.eval()
    for x, m, f, y, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x.to(dev), f.to(dev), m.to(dev)).float()
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(out), np.concatenate(ys)


def short_train(spec, train_sub, val_full, val_w300, table, mean, std, dev, epochs, workers, seed=0):
    torch.manual_seed(seed)
    model = build_two_branch(spec).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ds = TranscriptDataset(train_sub, table, mean, std, fragment_aware=True, error_aware=False, seed=seed)
    gen = torch.Generator(); gen.manual_seed(seed)                 # same batches for every candidate
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=workers, drop_last=True, pin_memory=(dev == "cuda"),
                    generator=gen, worker_init_fn=worker_init if workers > 0 else None)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, m, f, y, _ in dl:
            x, m, f, y = (t.to(dev, non_blocking=True) for t in (x, m, f, y))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lg = model(x, f, m)
            loss = F.cross_entropy(lg.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    s_full, y_full = score_rows(model, val_full, table, mean, std, dev, workers)
    s_w, y_w = score_rows(model, val_w300, table, mean, std, dev, workers)
    a1, v1 = delong_variance(y_full, s_full); a2, v2 = delong_variance(y_w, s_w)
    rec = {"val_auroc_full": 100 * a1, "val_auroc_w300": 100 * a2, "objective": 50 * (a1 + a2),
           "se": 50 * math.sqrt(v1 + v2), "train_seconds": time.time() - t0}
    return rec, (s_full, y_full, s_w, y_w)


# ----------------------------------------------------------------------------- Fisher geometry
def rank_auroc(scores, labels):
    """AUROC of scores for labels (ties count one half); NaN when a class is missing or a score is not finite."""
    s = np.asarray(scores, float); y = np.asarray(labels).astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or not np.all(np.isfinite(s)):
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _eig_psd(M):
    return np.clip(np.linalg.eigvalsh(0.5 * (M + M.T)), 0.0, None)


def _inv_wk_plus_c(K, w, c):
    """(W K + c I)^{-1} = W^{1/2} (W^{1/2} K W^{1/2} + c I)^{-1} W^{-1/2}  (symmetric, Cholesky-stable)."""
    sw = np.sqrt(w)
    B = sw[:, None] * K * sw[None, :] + c * np.eye(len(w))
    Binv = linalg.cho_solve(linalg.cho_factor(B, lower=True), np.eye(len(w)))
    return sw[:, None] * Binv / sw[None, :]


def gram_quantities(K, z, y, splits, damping=0.1):
    """All scalars from one Gram matrix.  splits = list of (idx_fit, idx_eval) index arrays."""
    K = np.asarray(K, float); z = np.asarray(z, float); y = np.asarray(y, float)
    n = len(y)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    w = np.clip(p * (1 - p), 1e-6, None)
    r = p - y
    out = {}
    sw = np.sqrt(w)
    lam = _eig_psd(sw[:, None] * K * sw[None, :]) / n          # nonzero spectrum of the Fisher F
    tr = float(lam.sum())
    out["fisher_tr"] = math.log(max(tr, 1e-300))
    out["fisher_pr"] = float(tr ** 2 / max(float((lam ** 2).sum()), 1e-300))      # participation ratio
    delta = damping * tr / n                                    # damping = 0.1 x mean nonzero eigenvalue
    out["fisher_logdet"] = float(np.sum(np.log1p(lam / delta)))
    mu = _eig_psd(K)
    out["ntk_cond"] = math.log(max(mu.max(), 1e-300) / max(mu.min(), 1e-12 * mu.max(), 1e-300))
    Hc = np.eye(n) - 1.0 / n
    Kc = Hc @ K @ Hc; yc = y - y.mean()
    out["lga"] = float(yc @ Kc @ yc / (np.linalg.norm(Kc) * (yc @ yc) + 1e-300))   # centred kernel-target alignment
    # Godambe / Takeuchi: tr(J (H + delta I)^{-1}) = tr(R^2 K (W K + n delta I)^{-1}); well-specified value: r^2 -> w
    KM = K @ _inv_wk_plus_c(K, w, n * delta)
    dKM = np.diag(KM)
    tic = float(np.sum(r ** 2 * dKM)); eff = float(np.sum(w * dKM))
    out["tic"] = tic
    out["fisher_effdim"] = eff                                   # = sum lambda / (lambda + delta)
    out["godambe_ratio"] = tic / max(eff, 1e-300)                # 1 when the empirical and model Fisher agree
    ng, gd = [], []
    for i1, i2 in splits:
        i1 = np.asarray(i1); i2 = np.asarray(i2)
        K11 = K[np.ix_(i1, i1)]; K21 = K[np.ix_(i2, i1)]; n1 = len(i1)
        w1 = w[i1]; res = y[i1] - p[i1]
        gd.append(rank_auroc(K21 @ res, y[i2]))
        d1 = damping * float(np.sum(w1 * np.diag(K11))) / n1 ** 2
        ng.append(rank_auroc(K21 @ (_inv_wk_plus_c(K11, w1, n1 * d1) @ res), y[i2]))
    out["fk_ngd"] = float(np.nanmean(ng)) if ng else float("nan")
    out["ntk_gd"] = float(np.nanmean(gd)) if gd else float("nan")
    return out


def balanced_splits(groups, seed=0):
    """Two complementary halves, each half taking half of every group (class x window type)."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g); rng.shuffle(idx)
        h = len(idx) // 2
        a += list(idx[:h]); b += list(idx[h:])
    a, b = np.array(sorted(a)), np.array(sorted(b))
    return [(a, b), (b, a)]


def gaussianize(cols):
    """Rank -> normal scores, column by column (robust to heavy tails); NaN -> 0."""
    X = np.asarray(cols, float)
    if X.ndim == 1:
        X = X[:, None]
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        v = X[:, j]; ok = np.isfinite(v)
        if ok.sum() >= 2 and np.ptp(v[ok]) > 0:
            rk = stats.rankdata(v[ok])
            out[ok, j] = stats.norm.ppf((rk - 0.5) / ok.sum())
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok])[0])


class Context:
    pass


def build_context(train, train_sub_idx, table, mean, std, n_probe, warm_steps, dev, seed=0):
    """Probe set (balanced: class x {fragment-aware window, 300-nt window}) drawn from training
    transcripts NOT used for the short training, and fixed warm-up batches from the short-training set."""
    ctx = Context()
    rng = np.random.default_rng(seed + 12345)
    used = set(int(i) for i in train_sub_idx)
    rest = [r for i, r in enumerate(train) if i not in used]
    if len(rest) < n_probe:
        rest = list(train)
    q = max(n_probe // 4, 2)
    items, groups = [], []
    for cls in (1, 0):
        pool = [r for r in rest if r["label"] == cls]
        pick = rng.choice(len(pool), size=min(2 * q, len(pool)), replace=False)
        sel = [pool[i] for i in pick]
        frag, fix = sel[:q], make_test_windows(sel[q:], lengths=(300,), half=False, seed=seed)["w300"]
        for rows, fa, g in ((frag, True, 2 * cls), (fix, False, 2 * cls + 1)):
            ds = TranscriptDataset(rows, table, mean, std, fragment_aware=fa, error_aware=False, seed=seed + 7)
            for i in range(len(ds)):
                x, m, f, y, _ = ds[i]
                items.append((x, m, f, int(y))); groups.append(g)
    ctx.items = items
    ctx.y = np.array([it[3] for it in items], float)
    ctx.splits = balanced_splits(np.array(groups), seed=seed)
    # fixed warm-up batches (identical for every architecture)
    sub = [train[i] for i in train_sub_idx]
    ds = TranscriptDataset(sub, table, mean, std, fragment_aware=True, error_aware=False, seed=seed + 3)
    g = torch.Generator(); g.manual_seed(seed + 3)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True, generator=g)
    ctx.warm = []
    for x, m, f, y, _ in dl:
        ctx.warm.append((x, m, f, y))
        if len(ctx.warm) >= warm_steps: break
    ctx.dev = dev
    return ctx


def margin_gram(model, items, dev):
    """Gram matrix of the per-example gradients of the logit margin (eval mode, fp32)."""
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params); n = len(items)
    A = torch.zeros(n, P, device=dev, dtype=torch.float32)
    z = np.zeros(n)
    model.eval()
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False          # inputs are cropped to a few lengths; avoid re-tuning per shape
    for i, (x, m, f, _) in enumerate(items):
        L = int(m.sum().item()); Lc = min(x.shape[-1], max(256, 1 << max(L - 1, 1).bit_length()))   # next power of two
        xi = x[:, :Lc].unsqueeze(0).to(dev); mi = m[:Lc].unsqueeze(0).to(dev); fi = f.unsqueeze(0).to(dev)
        lg = model(xi, fi, mi)[0]
        zi = lg[1] - lg[0]
        gs = torch.autograd.grad(zi, params, allow_unused=True)
        off = 0
        for gr, p in zip(gs, params):
            k = p.numel()
            if gr is not None:
                A[i, off:off + k] = gr.reshape(-1).float()
            off += k
        z[i] = float(zi.item())
    torch.backends.cudnn.benchmark = bench
    K = (A @ A.t()).double().cpu().numpy()
    del A
    return K, z


def warm_up(model, batches, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train(); losses = []
    for x, m, f, y in batches:
        x, m, f, y = x.to(dev), m.to(dev), f.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, f, m)
        loss = F.cross_entropy(lg.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def info_proxies(spec, ctx):
    out = {}
    try:
        torch.manual_seed(0)
        model = build_two_branch(spec).to(ctx.dev)
        K, z = margin_gram(model, ctx.items, ctx.dev)
        out.update(gram_quantities(K, z, ctx.y, ctx.splits))
        out["tse"] = warm_up(model, ctx.warm, ctx.dev)
        K2, z2 = margin_gram(model, ctx.items, ctx.dev)
        out.update({"warm_" + k: v for k, v in gram_quantities(K2, z2, ctx.y, ctx.splits).items()})
        out["warm_auroc"] = rank_auroc(z2, ctx.y)
        del model
    except Exception as e:                       # a failure of one architecture must not stop the search
        log(f"    WARNING: information proxies failed for {spec_name(spec)}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
    for k in ALL_Q:
        if k not in BASE:
            v = out.get(k, float("nan"))
            out[k] = float(v) if v is not None and np.isfinite(v) else float("nan")
    if ctx.dev == "cuda":
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- surrogate
class GraphGP:
    """Gaussian process over architectures.

    k(a, b) = s^2 exp(-sum_d beta_d [c_d(a) != c_d(b)]) exp(-1/2 sum_j (q_j(a) - q_j(b))^2 / l_j^2)

    The first factor is the diffusion kernel on the Cartesian product of complete graphs, one per design
    decision (Kondor and Lafferty 2002); the second is an automatic-relevance kernel over the rank-normalised
    proxies. The noise variance of an observation is the DeLong variance of its validation objective plus a
    learned term. Hyperparameters: maximum a posteriori with weak log-normal priors (L-BFGS-B, several starts)."""

    def __init__(self, n_cat, n_cont):
        self.D, self.J = n_cat, n_cont
        self.mu0 = np.r_[0.0, math.log(0.1), np.full(n_cat, math.log(0.3)), np.full(n_cont, math.log(0.05))]
        self.sd0 = np.r_[1.0, 1.5, np.full(n_cat, 1.5), np.full(n_cont, 1.5)]

    def _K(self, th, M, Dq):
        beta = np.exp(th[2:2 + self.D]); il2 = np.exp(th[2 + self.D:])
        e = np.zeros(M.shape[:2])
        if self.D: e = e - M @ beta
        if self.J: e = e - 0.5 * (Dq @ il2)
        return math.exp(th[0]) * np.exp(e)

    @staticmethod
    def _pair(Ca, Qa, Cb, Qb):
        M = (Ca[:, None, :] != Cb[None, :, :]).astype(float)
        Dq = (Qa[:, None, :] - Qb[None, :, :]) ** 2
        return M, Dq

    def fit(self, C, Q, y, noise_var, restarts=3, seed=0, maxiter=200):
        C = np.asarray(C, float).reshape(len(y), self.D); Q = np.asarray(Q, float).reshape(len(y), self.J)
        y = np.asarray(y, float)
        self.ym = float(y.mean()); self.ys = float(y.std()) if len(y) > 1 and y.std() > 0 else 1.0
        yn = (y - self.ym) / self.ys; nv = np.asarray(noise_var, float) / self.ys ** 2
        M, Dq = self._pair(C, Q, C, Q)
        n = len(y)

        def nlp(th):
            Kx = self._K(th, M, Dq) + np.diag(nv + math.exp(th[1])) + 1e-8 * np.eye(n)
            try:
                L = linalg.cholesky(Kx, lower=True)
            except linalg.LinAlgError:
                return 1e10
            a = linalg.cho_solve((L, True), yn)
            return 0.5 * yn @ a + np.log(np.diag(L)).sum() + 0.5 * np.sum(((th - self.mu0) / self.sd0) ** 2)

        rng = np.random.default_rng(seed)
        best = None
        for s in range(restarts):
            th0 = self.mu0 + (0 if s == 0 else rng.normal(0, 0.7, len(self.mu0)))
            try:
                res = optimize.minimize(nlp, th0, method="L-BFGS-B", bounds=[(-10, 5)] * len(th0), options={"maxiter": maxiter})
                if best is None or res.fun < best.fun: best = res
            except Exception:
                continue
        self.th = best.x if best is not None else self.mu0.copy()
        self.C, self.Q, self.yn, self.nv = C, Q, yn, nv
        self._factor()
        return self

    def _factor(self):
        M, Dq = self._pair(self.C, self.Q, self.C, self.Q)
        Kx = self._K(self.th, M, Dq) + np.diag(self.nv + math.exp(self.th[1])) + 1e-8 * np.eye(len(self.yn))
        self.L = linalg.cholesky(Kx, lower=True)
        self.alpha = linalg.cho_solve((self.L, True), self.yn)

    def condition(self, c, q, y_new, noise_new):
        """Same hyperparameters, one more (fantasy) observation (kriging believer)."""
        g = GraphGP(self.D, self.J); g.mu0, g.sd0, g.th = self.mu0, self.sd0, self.th
        g.ym, g.ys = self.ym, self.ys
        g.C = np.vstack([self.C, np.asarray(c, float).reshape(1, self.D)])
        g.Q = np.vstack([self.Q, np.asarray(q, float).reshape(1, self.J)])
        g.yn = np.r_[self.yn, (y_new - self.ym) / self.ys]; g.nv = np.r_[self.nv, noise_new / self.ys ** 2]
        g._factor()
        return g

    def predict(self, Cs, Qs):
        Cs = np.asarray(Cs, float); ns = Cs.shape[0] if Cs.ndim > 1 else 1
        Cs = Cs.reshape(ns, self.D); Qs = np.asarray(Qs, float).reshape(ns, self.J)
        M, Dq = self._pair(Cs, Qs, self.C, self.Q)
        Ks = self._K(self.th, M, Dq)
        mu = Ks @ self.alpha
        v = linalg.solve_triangular(self.L, Ks.T, lower=True)
        var = np.clip(math.exp(self.th[0]) - (v ** 2).sum(0), 1e-12, None)
        return mu * self.ys + self.ym, var * self.ys ** 2

    def relevance(self):
        return np.exp(self.th[2:2 + self.D]), np.exp(self.th[2 + self.D:])


def expected_improvement(mu, var, best, xi=0.0):
    s = np.sqrt(np.clip(var, 1e-12, None)); d = mu - best - xi; u = d / s
    return d * stats.norm.cdf(u) + s * stats.norm.pdf(u)
