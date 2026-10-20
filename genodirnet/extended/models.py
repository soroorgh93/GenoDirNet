"""Networks of the extended schedule (PyTorch).

  SequenceEncoder       strided one-dimensional convolutions built from a design specification. The default
                        specification is the sequence network of GenoDirNet: six convolutions of kernel 9 and
                        stride 2 with 32, 64, 96, 128, 192 and 256 filters, batch normalisation and ReLU. The
                        other values span the design space of the architecture searches: depth, width, kernel,
                        codon-periodic branch, residual pairs, GELU, group normalisation, squeeze-and-excitation,
                        dilation in the last two stages and the pooling rule.
  SequenceReader        encoder, pooling and a linear head.
  ORFBranch             perceptron on the five ORF features (eight with the likelihood-ratio features).
  TwoBranchNetwork      sequence network and ORF branch, concatenated, dropout, linear layer (two logits).
  CausalConvLM          autoregressive model of a nucleotide sequence (causal dilated convolutions with gated
                        activations, residual and skip connections).
  LLRFeatures           frozen pair of CausalConvLMs giving the mean, the maximum mean over 300-nucleotide
                        windows and the standard deviation of the per-nucleotide log-likelihood ratio (bits).
  MaskedDenoisingModel  encoder and a light transposed-convolution decoder: four outputs per position (masked
                        denoising) or one (replaced-nucleotide detection).
  SmallGenerator        generator of replaced-nucleotide detection.
  LengthWeightedLoss    loss weighted across window-length bins.

Every module takes one-hot input of shape B x 4 x L (padding = all-zero columns) and a mask B x L (1 = sequence).
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 8192

# ---------------------------------------------------------------------------------
#  design specification (one point of the design space of the architecture searches)
# ---------------------------------------------------------------------------------
DEFAULT_SPEC = dict(depth=6, width=32, kernel=9, codon_branch=0, block="plain", act="relu",
                    norm="bn", se=0, dilate_last=1, pool="meanmax")
SPACE = {                                   # decision -> admissible values
    "depth": [5, 6],
    "width": [24, 32, 48],
    "kernel": [5, 9, 15],
    "codon_branch": [0, 1],
    "block": ["plain", "respair"],
    "act": ["relu", "gelu"],
    "norm": ["bn", "gn"],
    "se": [0, 1],
    "dilate_last": [1, 2],
    "pool": ["meanmax", "attnmax", "last4"],
}
CHANNEL_PATTERN = {5: [1, 2, 4, 6, 8], 6: [1, 2, 3, 4, 6, 8]}


def spec_channels(spec):
    return [spec["width"] * m for m in CHANNEL_PATTERN[spec["depth"]]]


def load_spec(path_or_name):
    """'default' or None -> DEFAULT_SPEC; otherwise a JSON file with the specification (extra keys ignored)."""
    if path_or_name in (None, "", "default"):
        return dict(DEFAULT_SPEC)
    d = json.load(open(path_or_name))
    d = d.get("spec", d)
    spec = dict(DEFAULT_SPEC)
    spec.update({k: d[k] for k in DEFAULT_SPEC if k in d})
    return spec


def spec_to_vector(spec):
    return [SPACE[k].index(spec[k]) for k in SPACE]


def vector_to_spec(vec):
    return {k: SPACE[k][int(i)] for k, i in zip(SPACE, vec)}


def spec_name(spec):
    return "d{depth}w{width}k{kernel}c{codon_branch}{block}-{act}-{norm}-se{se}-dl{dilate_last}-{pool}".format(**spec)


def make_act(name):
    return nn.GELU() if name == "gelu" else nn.ReLU(inplace=True)


def make_norm(name, c):
    return nn.GroupNorm(8, c) if name == "gn" else nn.BatchNorm1d(c)


# ---------------------------------------------------------------------------------
#  building blocks
# ---------------------------------------------------------------------------------
class RepConv1d(nn.Module):
    """Re-parameterised one-dimensional convolution. The kernel is the sum of a k-tap kernel, a 3-tap kernel, a
    1-tap kernel and, optionally, a codon-periodic branch with taps at offsets -3, 0 and +3 and a learned 1 x 1
    projection. The branches are added into one kernel before the convolution, and fuse() returns an ordinary
    nn.Conv1d for deployment."""

    def __init__(self, c_in, c_out, k, stride=1, dilation=1, codon_branch=False, bias=False):
        super().__init__()
        self.k = max(k, 7) if codon_branch else k
        self.stride, self.dilation, self.codon_branch = stride, dilation, codon_branch
        self.c_in, self.c_out = c_in, c_out
        self.w_main = nn.Parameter(torch.empty(c_out, c_in, k))
        self.w3 = nn.Parameter(torch.empty(c_out, c_in, 3))
        self.w1 = nn.Parameter(torch.empty(c_out, c_in, 1))
        fan_in = c_in * k
        nn.init.trunc_normal_(self.w_main, std=math.sqrt(2.0 / fan_in))
        nn.init.trunc_normal_(self.w3, std=0.5 * math.sqrt(2.0 / (c_in * 3)))
        nn.init.trunc_normal_(self.w1, std=0.5 * math.sqrt(2.0 / c_in))
        if codon_branch:
            self.w_codon = nn.Parameter(torch.empty(c_out, c_in, 1))
            nn.init.trunc_normal_(self.w_codon, std=0.5 * math.sqrt(2.0 / (c_in * 3)))
            pattern = torch.zeros(self.k); c = self.k // 2
            pattern[[c - 3, c, c + 3]] = 1.0 / 3.0
            self.register_buffer("codon_pattern", pattern)
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None

    @staticmethod
    def _pad_to(w, k):
        p = (k - w.shape[-1]) // 2
        return F.pad(w, (p, k - w.shape[-1] - p))

    def kernel(self):
        w = self._pad_to(self.w_main, self.k) + self._pad_to(self.w3, self.k) + self._pad_to(self.w1, self.k)
        if self.codon_branch:
            w = w + self.w_codon * self.codon_pattern.view(1, 1, -1)
        return w

    def forward(self, x):
        pad = self.dilation * (self.k - 1) // 2
        return F.conv1d(x, self.kernel(), self.bias, stride=self.stride, padding=pad, dilation=self.dilation)

    @torch.no_grad()
    def fuse(self):
        conv = nn.Conv1d(self.c_in, self.c_out, self.k, stride=self.stride, padding=self.dilation * (self.k - 1) // 2,
                         dilation=self.dilation, bias=self.bias is not None)
        conv.weight.copy_(self.kernel())
        if self.bias is not None:
            conv.bias.copy_(self.bias)
        return conv


class SEBlock1d(nn.Module):
    """Squeeze-and-excitation over channels (masked global average)."""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(c // r, 4)); self.fc2 = nn.Linear(max(c // r, 4), c)

    def forward(self, x, m=None):
        if m is None:
            s = x.mean(2)
        else:
            s = (x * m).sum(2) / m.sum(2).clamp(min=1.0)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(2)


class ResidualPair1d(nn.Module):
    """x + 0.1 * conv_b(act(conv_a(x))), kernel 3, stride 1, no normalisation."""

    def __init__(self, c, act="relu", scale=0.1):
        super().__init__()
        self.a = nn.Conv1d(c, c, 3, padding=1); self.b = nn.Conv1d(c, c, 3, padding=1)
        self.act = make_act(act); self.scale = scale
        nn.init.trunc_normal_(self.a.weight, std=0.35 * math.sqrt(2.0 / (3 * c))); nn.init.zeros_(self.a.bias)
        nn.init.trunc_normal_(self.b.weight, std=0.35 * math.sqrt(2.0 / (3 * c))); nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.scale * self.b(self.act(self.a(x)))


class Stage(nn.Module):
    """One strided stage: convolution, normalisation, activation, optional squeeze-and-excitation and optional
    residual pair. With spec["codon_branch"] = 1 every stage uses the re-parameterised convolution; the
    codon-periodic taps are added only in the first stage, the only one that reads single nucleotides."""

    def __init__(self, c_in, c_out, spec, dilation, first=False):
        super().__init__()
        k = spec["kernel"]
        if spec["codon_branch"]:
            self.conv = RepConv1d(c_in, c_out, k, stride=2, dilation=dilation, codon_branch=first)
        else:
            self.conv = nn.Conv1d(c_in, c_out, k, stride=2, padding=dilation * (k - 1) // 2, dilation=dilation, bias=False)
        self.norm = make_norm(spec["norm"], c_out)
        self.act = make_act(spec["act"])
        self.se = SEBlock1d(c_out) if spec["se"] else None
        self.res = ResidualPair1d(c_out, spec["act"]) if spec["block"] == "respair" else None

    def forward(self, x, m=None):
        h = self.act(self.norm(self.conv(x)))
        if self.se is not None:
            h = self.se(h, m)
        if self.res is not None:
            h = self.res(h)
        return h


# ---------------------------------------------------------------------------------
#  sequence reader
# ---------------------------------------------------------------------------------
class SequenceEncoder(nn.Module):
    """Stack of strided stages; forward returns the last feature map, stages() all of them."""

    def __init__(self, spec=None, in_ch=4):
        super().__init__()
        self.spec = dict(DEFAULT_SPEC) if spec is None else dict(spec)
        chans = spec_channels(self.spec)
        stages, c_in = [], in_ch
        for i, c in enumerate(chans):
            dil = self.spec["dilate_last"] if i >= len(chans) - 2 else 1
            stages.append(Stage(c_in, c, self.spec, dil, first=(i == 0))); c_in = c
        self.stages = nn.ModuleList(stages)
        self.channels = chans
        self.out_channels = c_in
        self.downsample = 2 ** len(chans)
        self.record_acts = False            # True keeps every stage output with its gradient (search proxies)
        self._acts = []

    def pooled_mask(self, mask, T, level):
        """Mask pooled to the resolution of stage `level` (1-based), shape B x 1 x T."""
        if mask is None:
            return None
        f = 2 ** level
        return F.max_pool1d(mask.float().unsqueeze(1), kernel_size=f, stride=f, ceil_mode=True)[:, :, :T]

    def stages_forward(self, x, mask=None):
        outs, h = [], x
        self._acts = []
        for i, st in enumerate(self.stages):
            m = None
            if st.se is not None and mask is not None:
                T_next = (h.shape[2] + 1) // 2
                m = self.pooled_mask(mask, T_next, i + 1)
            h = st(h, m)
            outs.append(h)
            if self.record_acts:
                h.retain_grad(); self._acts.append(h)
        return outs

    def forward(self, x, mask=None):
        return self.stages_forward(x, mask)[-1]

    @torch.no_grad()
    def fuse(self):
        """Replace re-parameterised convolutions by plain ones (deployment)."""
        for st in self.stages:
            if isinstance(st.conv, RepConv1d):
                st.conv = st.conv.fuse()
        return self


class AttentionPool1d(nn.Module):
    def __init__(self, c, hidden=64):
        super().__init__()
        self.w = nn.Linear(c, hidden); self.v = nn.Linear(hidden, 1)

    def forward(self, h, m=None):                 # h: B x C x T
        s = self.v(torch.tanh(self.w(h.transpose(1, 2)))).squeeze(2)   # B x T
        if m is not None:
            s = s.masked_fill(m.squeeze(1) <= 0, -1e4)
        a = torch.softmax(s, dim=1)
        return (h * a.unsqueeze(1)).sum(2)


def masked_mean_max(h, m):
    if m is None:
        return h.mean(2), h.max(2).values
    mean = (h * m).sum(2) / m.sum(2).clamp(min=1.0)
    mx = (h + (m - 1) * 1e4).max(2).values
    return mean, mx


class SequenceReader(nn.Module):
    """Encoder + pooling (meanmax | attnmax | last4) + linear head."""

    def __init__(self, encoder=None, spec=None):
        super().__init__()
        self.encoder = encoder or SequenceEncoder(spec)
        spec = self.encoder.spec
        self.pool = spec["pool"]
        C = self.encoder.out_channels
        if self.pool == "meanmax":
            self.feat_dim = 2 * C
        elif self.pool == "attnmax":
            self.attn = AttentionPool1d(C); self.feat_dim = 2 * C
        elif self.pool == "last4":
            self.feat_dim = sum(self.encoder.channels[-4:]) + C
        else:
            raise ValueError(self.pool)
        self.head = nn.Linear(self.feat_dim, 2)

    def features(self, x, mask=None):
        outs = self.encoder.stages_forward(x, mask)
        h = outs[-1]
        m = self.encoder.pooled_mask(mask, h.shape[2], len(outs))
        if self.pool == "meanmax":
            mean, mx = masked_mean_max(h, m)
            return torch.cat([mean, mx], dim=1)
        if self.pool == "attnmax":
            _, mx = masked_mean_max(h, m)
            return torch.cat([self.attn(h, m), mx], dim=1)
        feats = []
        for lvl, hh in enumerate(outs[-4:], start=len(outs) - 3):
            mm = self.encoder.pooled_mask(mask, hh.shape[2], lvl)
            feats.append(masked_mean_max(hh, mm)[0])
        feats.append(masked_mean_max(h, m)[1])
        return torch.cat(feats, dim=1)

    def forward(self, x, mask=None):
        return self.head(self.features(x, mask))


# ---------------------------------------------------------------------------------
#  ORF branch, generative likelihood-ratio features, fusion
# ---------------------------------------------------------------------------------
class ORFBranch(nn.Module):
    def __init__(self, n_in=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.feat_dim = hidden

    def forward(self, f):
        return self.net(f)


class CausalConvLM(nn.Module):
    """Autoregressive model p(x_t | x_<t) of a nucleotide sequence: causal dilated convolutions with gated
    activations and residual and skip connections. Default: 128 channels, 20 blocks in two cycles of dilations
    1 to 512, kernel 3, so that the logits at position t depend on the 4,093 preceding nucleotides.
    forward(x) returns logits B x 4 x L, where logits[:, :, t] depends on x[:, :, :t] only."""

    def __init__(self, channels=128, n_blocks=20, kernel=3, skip_channels=128, cycle=10):
        super().__init__()
        self.channels, self.kernel, self.n_blocks, self.cycle = channels, kernel, n_blocks, cycle
        self.inp = nn.Conv1d(5, channels, 1)                     # 4 nucleotides + "start/pad" channel
        self.filters = nn.ModuleList(); self.gates = nn.ModuleList()
        self.res = nn.ModuleList(); self.skips = nn.ModuleList()
        self.dilations = [2 ** (i % cycle) for i in range(n_blocks)]
        for d in self.dilations:
            self.filters.append(nn.Conv1d(channels, channels, kernel, dilation=d))
            self.gates.append(nn.Conv1d(channels, channels, kernel, dilation=d))
            self.res.append(nn.Conv1d(channels, channels, 1))
            self.skips.append(nn.Conv1d(channels, skip_channels, 1))
        self.out1 = nn.Conv1d(skip_channels, skip_channels, 1); self.out2 = nn.Conv1d(skip_channels, 4, 1)
        self.receptive_field = 1 + (kernel - 1) * sum(self.dilations)

    def forward(self, x):
        B, _, L = x.shape
        start = torch.zeros(B, 1, L, dtype=x.dtype, device=x.device); start[:, 0, 0] = 1.0
        shifted = F.pad(x, (1, 0))[:, :, :L]                     # x_{t-1} at position t, zeros at t = 0
        h = self.inp(torch.cat([shifted, start], dim=1))
        skip = 0
        for f, g, r, s, d in zip(self.filters, self.gates, self.res, self.skips, self.dilations):
            pad = (self.kernel - 1) * d
            hp = F.pad(h, (pad, 0))                              # left padding = causal
            z = torch.tanh(f(hp)) * torch.sigmoid(g(hp))
            skip = skip + s(z)
            h = h + r(z)
        return self.out2(F.relu(self.out1(F.relu(skip))))

    def log_prob(self, x, mask=None):
        """Per-position log p(x_t | x_<t) (B x L); zero at padding."""
        lp = F.log_softmax(self.forward(x).float(), dim=1)
        ll = (lp * x.float()).sum(1)
        if mask is not None:
            ll = ll * mask.float()
        return ll

    @torch.no_grad()
    def sample(self, n, length, device, temperature=1.0, top_p=0.95, context=None, prefix=None):
        """Autoregressive sampling with temperature and nucleus (top-p) sampling. context=None conditions on the
        whole receptive field; a smaller context (the synthetic windows use 512 nucleotides) is faster."""
        context = self.receptive_field if context is None else context
        x = torch.zeros(n, 4, length, device=device)
        t0 = 0
        if prefix is not None:
            t0 = prefix.shape[2]; x[:, :, :t0] = prefix
        on_gpu = torch.device(device).type == "cuda"
        for t in range(t0, length):
            a = max(0, t - context)
            ctx = torch.cat([x[:, :, a:t], torch.zeros(n, 4, 1, device=device)], dim=2)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_gpu):
                out = self.forward(ctx)[:, :, -1]
            logits = out.float() / max(temperature, 1e-6)
            probs = torch.softmax(logits.float(), dim=1)
            if top_p < 1.0:
                sp, si = probs.sort(dim=1, descending=True)
                cum = sp.cumsum(1); keep = (cum - sp) < top_p
                sp = sp * keep; sp = sp / sp.sum(1, keepdim=True)
                idx = si.gather(1, torch.multinomial(sp, 1))
            else:
                idx = torch.multinomial(probs, 1)
            x[torch.arange(n, device=device), idx.squeeze(1), t] = 1.0
        return x


class LLRFeatures(nn.Module):
    """Frozen pair (coding, noncoding) of CausalConvLMs. raw() returns, per window, the mean, the maximum mean over
    300-nucleotide windows and the standard deviation of the per-nucleotide log-likelihood ratio in bits (B x 3);
    forward() standardises them with batch normalisation."""

    def __init__(self, lm_coding, lm_noncoding, window=300):
        super().__init__()
        self.lm_c, self.lm_n, self.window = lm_coding, lm_noncoding, window
        for p in self.parameters():
            p.requires_grad_(False)
        self.norm = nn.BatchNorm1d(3)

    @torch.no_grad()
    def raw(self, x, mask):
        self.lm_c.eval(); self.lm_n.eval()
        d = (self.lm_c.log_prob(x, mask) - self.lm_n.log_prob(x, mask)) / math.log(2.0)   # bits, B x L
        m = mask.float()
        n = m.sum(1).clamp(min=1.0)
        mean = d.sum(1) / n
        w = min(self.window, x.shape[2])
        num = F.avg_pool1d(d.unsqueeze(1), w, stride=1).squeeze(1) * w
        den = F.avg_pool1d(m.unsqueeze(1), w, stride=1).squeeze(1) * w
        valid = den >= n.unsqueeze(1).clamp(max=float(w)) * 0.999
        win = (num / den.clamp(min=1.0)).masked_fill(~valid, -1e4)
        mx = win.max(1).values
        mx = torch.where(mx < -1e3, mean, mx)
        var = ((d - mean.unsqueeze(1)) ** 2 * m).sum(1) / n
        return torch.stack([mean, mx, var.sqrt()], dim=1)

    def forward(self, x, mask):
        return self.norm(self.raw(x, mask))


class TwoBranchNetwork(nn.Module):
    """Sequence network and ORF branch (with the likelihood-ratio features when llr is given) -> two logits."""

    def __init__(self, encoder=None, n_orf=5, dropout=0.2, spec=None, llr=None):
        super().__init__()
        self.seq = SequenceReader(encoder, spec)
        self.llr = llr
        self.orf = ORFBranch(n_orf + (3 if llr is not None else 0))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.seq.feat_dim + self.orf.feat_dim, 2)

    def forward(self, x, orf_feats, mask=None):
        f = orf_feats
        if self.llr is not None:
            f = torch.cat([f, self.llr(x, mask if mask is not None else torch.ones(x.shape[0], x.shape[2], device=x.device))], dim=1)
        z = torch.cat([self.seq.features(x, mask), self.orf(f)], dim=1)
        return self.head(self.drop(z))

    def state_dict_without_llr(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("llr.lm_")}


# ---------------------------------------------------------------------------------
#  pretraining heads
# ---------------------------------------------------------------------------------
class MaskedDenoisingModel(nn.Module):
    """Encoder + light transposed-convolution decoder; `out_dim` 4 (nucleotide logits) or 1 (replaced-token logit)."""

    def __init__(self, encoder=None, spec=None, out_dim=4):
        super().__init__()
        self.encoder = encoder or SequenceEncoder(spec)
        k = 9
        rev = list(reversed(self.encoder.channels))
        layers, c_in = [], rev[0]
        for c in rev[1:] + [32]:
            layers += [nn.ConvTranspose1d(c_in, c, k, stride=2, padding=k // 2, output_padding=1, bias=False),
                       nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            c_in = c
        self.decoder = nn.Sequential(*layers)
        self.out = nn.Conv1d(c_in, out_dim, 1)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)[:, :, :x.shape[2]]
        if y.shape[2] < x.shape[2]:
            y = F.pad(y, (0, x.shape[2] - y.shape[2]))
        return self.out(y)


class SmallGenerator(nn.Module):
    """Generator of replaced-nucleotide detection: three convolutions of 64 filters that predict hidden nucleotides."""

    def __init__(self, width=64, n_layers=3, k=9):
        super().__init__()
        layers, c_in = [], 4
        for _ in range(n_layers):
            layers += [nn.Conv1d(c_in, width, k, padding=k // 2), nn.BatchNorm1d(width), nn.ReLU(inplace=True)]; c_in = width
        self.net = nn.Sequential(*layers); self.out = nn.Conv1d(width, 4, 1)

    def forward(self, x):
        return self.out(self.net(x))


# ---------------------------------------------------------------------------------
#  utilities
# ---------------------------------------------------------------------------------
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_macs_reader(spec, L=SEQ_LEN):
    """Multiply-accumulate operations of the sequence network for an input of L nucleotides."""
    chans = spec_channels(spec)
    macs, c_in, T = 0, 4, L
    for i, c in enumerate(chans):
        k = max(spec["kernel"], 7) if (spec["codon_branch"] and i == 0) else spec["kernel"]
        T = (T + 1) // 2
        macs += T * c * c_in * k
        if spec["block"] == "respair":
            macs += 2 * T * c * c * 3
        if spec["se"]:
            macs += c * max(c // 8, 4) * 2
        c_in = c
    return macs


def build_two_branch(spec=None, llr=None, n_orf=5):
    return TwoBranchNetwork(SequenceEncoder(spec), n_orf=n_orf, spec=spec, llr=llr)


def load_llr(lm_dir, device="cpu"):
    """The likelihood-ratio features from the two autoregressive models written by scripts/genodirnet2/autoregressive.py."""
    lms = []
    for name in ("lm_coding.pt", "lm_noncoding.pt"):
        ck = torch.load(os.path.join(lm_dir, name), map_location=device)
        lm = CausalConvLM(**ck["config"]); lm.load_state_dict(ck["state_dict"]); lm.eval()
        lms.append(lm.to(device))
    return LLRFeatures(lms[0], lms[1]).to(device)


class LengthWeightedLoss(nn.Module):
    """Loss weighted across window-length bins. The mean loss of each bin is tracked by an exponential moving
    average (decay `ema`); bin g has weight w_g = softmax_g(ell_g / tau). In a mini-batch, the weights of the bins
    present are renormalised to sum to one and each bin contributes w_g times the mean loss of its examples. The
    weights are constants for the gradient."""

    def __init__(self, n_groups, tau=0.05, ema=0.98):
        super().__init__()
        self.n_groups, self.tau, self.ema = n_groups, float(tau), ema
        self.register_buffer("xi0", torch.ones(n_groups) / n_groups)
        self.register_buffer("ell", torch.zeros(n_groups))
        self.register_buffer("seen", torch.zeros(n_groups))

    @torch.no_grad()
    def _update(self, per_example_loss, group_ids):
        for g in range(self.n_groups):
            m = group_ids == g
            if m.any():
                v = per_example_loss[m].mean()
                self.ell[g] = v if self.seen[g] == 0 else self.ema * self.ell[g] + (1 - self.ema) * v
                self.seen[g] += 1

    def weights(self):
        return torch.softmax(torch.log(self.xi0 + 1e-12) + self.ell / self.tau, dim=0)

    def forward(self, per_example_loss, group_ids):
        self._update(per_example_loss.detach(), group_ids)
        w = self.weights()
        counts = torch.bincount(group_ids, minlength=self.n_groups)
        w_present = w * (counts > 0).float()
        w_present = w_present / w_present.sum().clamp(min=1e-12)
        per_w = (w_present / counts.clamp(min=1).float())[group_ids]
        return (per_w * per_example_loss).sum()
