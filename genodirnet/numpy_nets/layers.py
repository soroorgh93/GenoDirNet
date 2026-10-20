"""Layers with explicit forward and backward passes (channels last, float32)."""
from __future__ import annotations
import math
import numpy as np
from . import lib, ptr, uptr, c32

class Layer:
    def __init__(self):
        self.params, self.grads, self.buffers, self.frozen = {}, {}, {}, set()
        self.training = True

    def modules(self):
        yield self
        for v in self.__dict__.values():
            if isinstance(v, Layer):
                yield from v.modules()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Layer):
                        yield from m.modules()

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def named_params(self, prefix=''):
        for i, m in enumerate(self.modules()):
            for k, v in m.params.items():
                yield f'{prefix}{i}.{k}', m, k

    def state_dict(self):
        d = {}
        for name, m, k in self.named_params():
            d[name] = m.params[k]
        for i, m in enumerate(self.modules()):
            for k, v in m.buffers.items():
                d[f'{i}.buf.{k}'] = v
        return d

    def load_state_dict(self, d):
        for name, m, k in self.named_params():
            m.params[k] = d[name].copy()
        for i, m in enumerate(self.modules()):
            for k in m.buffers:
                m.buffers[k] = d[f'{i}.buf.{k}'].copy()

    def n_params(self):
        return sum(v.size for _, m, k in self.named_params() for v in [m.params[k]])

    def zero_grad(self):
        for m in self.modules():
            m.grads = {}

    def free(self):
        """Drop the activations kept for the backward pass (after an inference batch)."""
        for m in self.modules():
            m.cache = None

def kaiming_out(shape, fan_out, rng):
    return (rng.standard_normal(shape) * math.sqrt(2.0 / fan_out)).astype(np.float32)

class Conv(Layer):
    """Dense convolution with kernel (kh, kw), stride s, 'same' padding (k // 2) unless pad is given."""

    def __init__(self, cin, cout, k, s=1, rng=None, pad=None):
        super().__init__()
        self.kh, self.kw = (k, k) if isinstance(k, int) else k
        self.s, self.cin, self.cout = s, cin, cout
        self.ph, self.pw = (self.kh // 2, self.kw // 2) if pad is None else ((pad, pad) if isinstance(pad, int) else pad)
        self.params['w'] = kaiming_out((self.kh, self.kw, cin, cout), cout * self.kh * self.kw, rng)
        self.macs = 0

    def forward(self, x):
        B, H, W, C = x.shape
        Ho = (H + 2 * self.ph - self.kh) // self.s + 1; Wo = (W + 2 * self.pw - self.kw) // self.s + 1
        if self.kh == 1 and self.kw == 1 and self.s == 1:
            cols = x.reshape(-1, C)
        else:
            cols = np.empty((B * Ho * Wo, self.kh * self.kw * C), dtype=np.float32)
            lib().im2col(ptr(x), ptr(cols), B, H, W, C, self.kh, self.kw, self.s, self.ph, self.pw, Ho, Wo)
        y = cols @ self.params['w'].reshape(-1, self.cout)
        self.cache = (cols, x.shape, Ho, Wo)
        self.macs = cols.shape[0] * cols.shape[1] * self.cout // B
        return y.reshape(B, Ho, Wo, self.cout)

    def backward(self, dy):
        cols, xshape, Ho, Wo = self.cache
        B, H, W, C = xshape
        dy2 = c32(dy).reshape(-1, self.cout)
        self.grads['w'] = (cols.T @ dy2).reshape(self.params['w'].shape)
        dcols = dy2 @ self.params['w'].reshape(-1, self.cout).T
        if self.kh == 1 and self.kw == 1 and self.s == 1:
            return dcols.reshape(xshape)
        dx = np.empty(xshape, dtype=np.float32)
        lib().col2im(ptr(c32(dcols)), ptr(dx), B, H, W, C, self.kh, self.kw, self.s, self.ph, self.pw, Ho, Wo)
        return dx

class DWConv(Layer):
    """Depthwise convolution with kernel (kh, kw) and 'same' padding."""

    def __init__(self, c, k, s=1, rng=None, fixed=None):
        super().__init__()
        self.kh, self.kw = (k, k) if isinstance(k, int) else k
        self.c, self.s, self.ph, self.pw = c, s, self.kh // 2, self.kw // 2
        if fixed is not None:
            self.params['w'] = np.repeat(np.asarray(fixed, dtype=np.float32).reshape(self.kh, self.kw, 1), c, axis=2)
            self.frozen.add('w')
        else:
            self.params['w'] = kaiming_out((self.kh, self.kw, c), self.kh * self.kw, rng)
        self.macs = 0

    def forward(self, x):
        B, H, W, C = x.shape
        Ho = (H + 2 * self.ph - self.kh) // self.s + 1; Wo = (W + 2 * self.pw - self.kw) // self.s + 1
        y = np.empty((B, Ho, Wo, C), dtype=np.float32)
        lib().dw_fwd(ptr(x), ptr(self.params['w']), ptr(y), B, H, W, C, self.kh, self.kw, self.s, self.ph, self.pw, Ho, Wo)
        self.cache = (x, Ho, Wo)
        self.macs = Ho * Wo * C * self.kh * self.kw
        return y

    def backward(self, dy):
        x, Ho, Wo = self.cache
        B, H, W, C = x.shape
        dy = c32(dy)
        if 'w' not in self.frozen:
            dw = np.empty_like(self.params['w'])
            lib().dw_bwd_w(ptr(x), ptr(dy), ptr(dw), B, H, W, C, self.kh, self.kw, self.s, self.ph, self.pw, Ho, Wo)
            self.grads['w'] = dw
        dx = np.empty_like(x)
        lib().dw_bwd_x(ptr(dy), ptr(self.params['w']), ptr(dx), B, H, W, C, self.kh, self.kw, self.s, self.ph, self.pw, Ho, Wo)
        return dx

class BN(Layer):
    def __init__(self, c, eps=1e-5, momentum=0.1):
        super().__init__()
        self.params['g'] = np.ones(c, dtype=np.float32); self.params['b'] = np.zeros(c, dtype=np.float32)
        self.buffers['mean'] = np.zeros(c, dtype=np.float32); self.buffers['var'] = np.ones(c, dtype=np.float32)
        self.eps, self.momentum, self.c = eps, momentum, c

    def forward(self, x):
        x = c32(x); C = x.shape[-1]; N = x.size // C
        xhat = np.empty_like(x); y = np.empty_like(x)
        if self.training:
            mean = np.empty(C, np.float32); inv = np.empty(C, np.float32)
            lib().bn_fwd(ptr(x), ptr(self.params['g']), ptr(self.params['b']), ptr(xhat), ptr(y), ptr(mean), ptr(inv), N, C, 1, self.eps)
            var = 1.0 / (inv * inv) - self.eps
            self.buffers['mean'] = (1 - self.momentum) * self.buffers['mean'] + self.momentum * mean
            self.buffers['var'] = ((1 - self.momentum) * self.buffers['var'] + self.momentum * var * N / max(1, N - 1)).astype(np.float32)
        else:
            mean = self.buffers['mean']; inv = (1.0 / np.sqrt(self.buffers['var'] + self.eps)).astype(np.float32)
            lib().bn_fwd(ptr(x), ptr(self.params['g']), ptr(self.params['b']), ptr(xhat), ptr(y), ptr(mean), ptr(inv), N, C, 0, self.eps)
        self.cache = (xhat, inv)
        return y

    def backward(self, dy):
        xhat, inv = self.cache
        dy = c32(dy); C = dy.shape[-1]; N = dy.size // C
        dx = np.empty_like(dy); dg = np.empty(C, np.float32); db = np.empty(C, np.float32)
        lib().bn_bwd(ptr(dy), ptr(xhat), ptr(self.params['g']), ptr(inv), ptr(dx), ptr(dg), ptr(db), N, C, 1 if self.training else 0)
        self.grads['g'] = dg; self.grads['b'] = db
        return dx

class ReLU(Layer):
    def forward(self, x):
        x = c32(x); y = np.empty_like(x); m = np.empty(x.shape, np.uint8)
        lib().relu_fwd(ptr(x), ptr(y), uptr(m), x.size); self.cache = m; return y

    def backward(self, dy):
        dy = c32(dy); dx = np.empty_like(dy)
        lib().relu_bwd(ptr(dy), uptr(self.cache), ptr(dx), dy.size); return dx

class BNReLU(BN):
    """Batch normalisation followed by ReLU in one pass over the tensor."""

    def forward(self, x):
        x = c32(x); C = x.shape[-1]; N = x.size // C
        xhat = np.empty_like(x); y = np.empty_like(x); mask = np.empty(x.shape, np.uint8)
        if self.training:
            mean = np.empty(C, np.float32); inv = np.empty(C, np.float32)
            lib().bn_relu_fwd(ptr(x), ptr(self.params['g']), ptr(self.params['b']), ptr(xhat), ptr(y), uptr(mask), ptr(mean), ptr(inv), N, C, 1, self.eps)
            var = 1.0 / (inv * inv) - self.eps
            self.buffers['mean'] = (1 - self.momentum) * self.buffers['mean'] + self.momentum * mean
            self.buffers['var'] = ((1 - self.momentum) * self.buffers['var'] + self.momentum * var * N / max(1, N - 1)).astype(np.float32)
        else:
            mean = self.buffers['mean']; inv = (1.0 / np.sqrt(self.buffers['var'] + self.eps)).astype(np.float32)
            lib().bn_relu_fwd(ptr(x), ptr(self.params['g']), ptr(self.params['b']), ptr(xhat), ptr(y), uptr(mask), ptr(mean), ptr(inv), N, C, 0, self.eps)
        self.cache = (xhat, inv, mask)
        return y

    def backward(self, dy):
        xhat, inv, mask = self.cache
        dy = c32(dy); d = np.empty_like(dy)
        lib().relu_bwd(ptr(dy), uptr(mask), ptr(d), dy.size)
        self.cache = (xhat, inv)
        return BN.backward(self, d)

class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__(); self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dy):
        for l in reversed(self.layers):
            dy = l.backward(dy)
        return dy

def conv_bn(cin, cout, k, s, rng, relu=True):
    return Sequential(Conv(cin, cout, k, s, rng), BNReLU(cout) if relu else BN(cout))

class DropPath(Layer):
    def __init__(self, p, rng):
        super().__init__(); self.p, self.rng = p, rng

    def forward(self, x):
        if not self.training or self.p == 0:
            self.cache = None; return x
        keep = 1 - self.p
        m = (self.rng.random((x.shape[0],) + (1,) * (x.ndim - 1)) < keep).astype(np.float32) / keep
        self.cache = m; return x * m

    def backward(self, dy):
        return dy if self.cache is None else dy * self.cache

class Dropout(Layer):
    def __init__(self, p, rng):
        super().__init__(); self.p, self.rng = p, rng

    def forward(self, x):
        if not self.training or self.p == 0:
            self.cache = None; return x
        m = (self.rng.random(x.shape) >= self.p).astype(np.float32) / (1 - self.p)
        self.cache = m; return x * m

    def backward(self, dy):
        return dy if self.cache is None else dy * self.cache

class Linear(Layer):
    def __init__(self, cin, cout, rng, std=0.02):
        super().__init__()
        w = rng.standard_normal((cin, cout)) * std
        self.params['w'] = np.clip(w, -2 * std, 2 * std).astype(np.float32)
        self.params['b'] = np.zeros(cout, dtype=np.float32)

    def forward(self, x):
        self.cache = x; return x @ self.params['w'] + self.params['b']

    def backward(self, dy):
        self.grads['w'] = self.cache.T @ dy; self.grads['b'] = dy.sum(0)
        return dy @ self.params['w'].T

class GlobalAvgPool(Layer):
    def forward(self, x):
        self.cache = x.shape; return x.mean((1, 2))

    def backward(self, dy):
        B, H, W, C = self.cache
        return np.broadcast_to(dy[:, None, None, :] / (H * W), self.cache).astype(np.float32)

class MaxPool2(Layer):
    def forward(self, x):
        B, H, W, C = x.shape
        y = np.empty((B, H // 2, W // 2, C), dtype=np.float32); arg = np.empty(y.shape, dtype=np.uint8)
        lib().maxpool2_fwd(ptr(x), ptr(y), uptr(arg), B, H, W, C)
        self.cache = (arg, x.shape); return y

    def backward(self, dy):
        arg, shape = self.cache
        dx = np.empty(shape, dtype=np.float32)
        lib().maxpool2_bwd(ptr(c32(dy)), uptr(arg), ptr(dx), *shape)
        return dx

class DirectionalMixerBase(Layer):
    pass

class DirectionalMixer(DirectionalMixerBase):
    """Five depthwise paths (k x k, 1 x k, k x 1, 1 x 1, identity) summed, then one batch normalisation (shared)."""

    def __init__(self, c, k, rng):
        super().__init__()
        self.k, self.c = k, c
        self.square = DWConv(c, k, rng=rng)
        self.horizontal = DWConv(c, (1, k), rng=rng)
        self.vertical = DWConv(c, (k, 1), rng=rng)
        self.point = DWConv(c, 1, rng=rng)
        self.bn = BNReLU(c)
        self.macs = 0

    def equivalent_kernel(self):
        k = self.k
        w = self.square.params['w'].copy()
        w[k // 2, :, :] += self.horizontal.params['w'][0]
        w[:, k // 2, :] += self.vertical.params['w'][:, 0]
        w[k // 2, k // 2, :] += self.point.params['w'][0, 0] + 1.0
        return w

    def forward(self, x):
        x = c32(x); B, H, W, C = x.shape; k = self.k
        self.w = self.equivalent_kernel(); y = np.empty_like(x)
        lib().dw_fwd(ptr(x), ptr(self.w), ptr(y), B, H, W, C, k, k, 1, k // 2, k // 2, H, W)
        self.cache = x; self.macs = H * W * C * k * k
        return self.bn.forward(y)

    def backward(self, dy):
        d = c32(self.bn.backward(dy)); x = self.cache; B, H, W, C = x.shape; k = self.k
        dw = np.empty_like(self.w); dx = np.empty_like(x)
        lib().dw_bwd_w(ptr(x), ptr(d), ptr(dw), B, H, W, C, k, k, 1, k // 2, k // 2, H, W)
        lib().dw_bwd_x(ptr(d), ptr(self.w), ptr(dx), B, H, W, C, k, k, 1, k // 2, k // 2, H, W)
        self.square.grads['w'] = dw
        self.horizontal.grads['w'] = dw[k // 2:k // 2 + 1, :, :].copy()
        self.vertical.grads['w'] = dw[:, k // 2:k // 2 + 1, :].copy()
        self.point.grads['w'] = dw[k // 2:k // 2 + 1, k // 2:k // 2 + 1, :].copy()
        return dx

class PlainDepthwiseMixer(DirectionalMixerBase):
    """Variant of the directional mixer: one k x k depthwise convolution trained directly, without the line, point and
    identity paths."""

    def __init__(self, c, k, rng):
        super().__init__()
        self.k, self.c = k, c
        self.square = DWConv(c, k, rng=rng)
        self.bn = BNReLU(c)
        self.macs = 0

    def forward(self, x):
        y = self.square.forward(c32(x)); self.macs = self.square.macs
        return self.bn.forward(y)

    def backward(self, dy):
        return self.square.backward(c32(self.bn.backward(dy)))

class DirectionalBlock(Layer):
    def __init__(self, c, expansion, k, drop_path, rng, plain=False):
        super().__init__()
        hidden = int(math.ceil(c * expansion / 8.0) * 8)
        self.expand = conv_bn(c, hidden, 1, 1, rng)
        self.mixer = (PlainDepthwiseMixer if plain else DirectionalMixer)(hidden, k, rng)
        self.project = conv_bn(hidden, c, 1, 1, rng, relu=False)
        self.project.layers[1].params['g'][:] = 0.0
        self.drop = DropPath(drop_path, rng)
        self.out = ReLU()

    def forward(self, x):
        y = self.mixer.forward(self.expand.forward(x))
        y = self.drop.forward(self.project.forward(y))
        return self.out.forward(x + y)

    def backward(self, dy):
        d = self.out.backward(dy)
        dres = d
        d = self.expand.backward(self.mixer.backward(self.project.backward(self.drop.backward(d))))
        return d + dres

class CompassBlock(Layer):
    AX = [[0, .25, 0], [.25, 0, .25], [0, .25, 0]]
    DG = [[.25, 0, .25], [0, 0, 0], [.25, 0, .25]]

    def __init__(self, c, rng):
        super().__init__()
        self.axial = DWConv(c, 3, fixed=self.AX); self.diagonal = DWConv(c, 3, fixed=self.DG)
        self.pa = Conv(c, c, 1, 1, rng); self.pd = Conv(c, c, 1, 1, rng)
        self.bn = BN(c); self.bn.params['g'][:] = 0.0
        self.act = ReLU()

    def forward(self, x):
        m = self.pa.forward(self.axial.forward(x)) + self.pd.forward(self.diagonal.forward(x))
        return self.act.forward(x + self.bn.forward(m))

    def backward(self, dy):
        d = self.act.backward(dy)
        dm = self.bn.backward(d)
        return d + self.axial.backward(self.pa.backward(dm)) + self.diagonal.backward(self.pd.backward(dm))

class PyramidHead(Layer):
    def __init__(self, channels, head_width, num_classes, dropout, rng):
        super().__init__()
        self.proj = [conv_bn(c, head_width, 1, 1, rng) for c in channels]
        self.pools = [GlobalAvgPool() for _ in channels]
        self.drop = Dropout(dropout, rng)
        self.fc = Linear(head_width * len(channels), num_classes, rng)
        self.hw = head_width

    def forward(self, feats):
        v = [p.forward(pr.forward(f)) for p, pr, f in zip(self.pools, self.proj, feats)]
        return self.fc.forward(self.drop.forward(np.concatenate(v, 1)))

    def backward(self, dy):
        d = self.drop.backward(self.fc.backward(dy))
        return [pr.backward(p.backward(d[:, i * self.hw:(i + 1) * self.hw])) for i, (p, pr) in enumerate(zip(self.pools, self.proj))]

class AuxHead(Layer):
    def __init__(self, c, num_classes, rng):
        super().__init__(); self.pool = GlobalAvgPool(); self.fc = Linear(c, num_classes, rng)

    def forward(self, x):
        return self.fc.forward(self.pool.forward(x))

    def backward(self, dy):
        return self.pool.backward(self.fc.backward(dy))
