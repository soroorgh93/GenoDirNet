"""The networks of the standard schedule in the NumPy framework (channels last): the FR-CGR network and its
variants, the plain convolutional reference, the sequence network and the sequence-ORF network."""

import numpy as np

from .layers import (Layer, BN, ReLU, Sequential, conv_bn, Conv, Linear, GlobalAvgPool, Dropout,
                     DirectionalMixer, PlainDepthwiseMixer, DirectionalBlock, CompassBlock, PyramidHead, AuxHead)

# identifier -> (word length k, control of the input); see genodirnet.frcgr.network_input
FRCGR_INPUTS = {'frcgr_k5': (5, None), 'frcgr_k6': (6, None), 'frcgr_k7': (7, None), 'frcgr_k5_summed': (5, 'summed'),
                'frcgr_k5_partition': (5, 'partition'), 'frcgr_k5_permuted': (5, 'permuted'), 'frcgr_k5_onekernel': (5, None),
                'frcgr_k5_noneighbour': (5, None), 'plaincnn_k5': (5, None)}
SEQUENCE_MODELS = ('seq_net', 'seq_orf_net')


class FRCGRNet(Layer):
    """FR-CGR network: a stem, five stages of inverted residual blocks whose depthwise kernel is the sum of five paths,
    neighbourhood blocks after stages 4 and 5, a head on the pooled outputs of stages 3 to 5 and two auxiliary heads."""

    def __init__(self, rng, num_classes=2, stem_width=16, widths=(24, 40, 72, 112, 160), depths=(1, 2, 2, 2, 1),
                 expansions=(2.0, 3.0, 3.0, 3.0, 2.0), stage_kernels=((3,), (5, 3), (3, 5), (5, 3), (5,)),
                 neighbour_after_stages=(3, 4), head_width=96, dropout=0.2, drop_path_rate=0.08, plain=False):
        super().__init__()
        self.stem = conv_bn(3, stem_width, 3, 2, rng)
        rates = np.linspace(0.0, drop_path_rate, sum(depths)).tolist(); idx = 0
        self.stages = []; cin = stem_width
        for si, (w, d, e, ks) in enumerate(zip(widths, depths, expansions, stage_kernels)):
            stride = 1 if si == 0 else 2
            mods = [conv_bn(cin, w, 1 if stride == 1 else 3, stride, rng)]
            for bi in range(d):
                mods.append(DirectionalBlock(w, e, ks[bi], rates[idx], rng, plain)); idx += 1
            if si in neighbour_after_stages:
                mods.append(CompassBlock(w, rng))
            self.stages.append(Sequential(*mods)); cin = w
        self.head = PyramidHead(widths[-3:], head_width, num_classes, dropout, rng)
        self.aux3 = AuxHead(widths[-3], num_classes, rng); self.aux4 = AuxHead(widths[-2], num_classes, rng)
        self.widths = widths

    def forward(self, x, aux=None):
        aux = self.training if aux is None else aux
        x = self.stem.forward(x); feats = []
        for st in self.stages:
            x = st.forward(x); feats.append(x)
        f3, f4, f5 = feats[-3:]
        logits = self.head.forward((f3, f4, f5))
        self.cache_aux = aux
        if aux:
            return logits, self.aux3.forward(f3), self.aux4.forward(f4)
        return logits

    def backward(self, dlogits, da3=None, da4=None):
        d3, d4, d5 = self.head.backward(dlogits)
        if self.cache_aux:
            d3 = d3 + self.aux3.backward(da3); d4 = d4 + self.aux4.backward(da4)
        d = self.stages[4].backward(d5) + d4
        d = self.stages[3].backward(d) + d3
        for st in (self.stages[2], self.stages[1], self.stages[0]):
            d = st.backward(d)
        return self.stem.backward(d)

    def deploy_params(self):
        """Parameters after merging the five paths of every depthwise kernel, without the auxiliary heads."""
        aux = {id(m) for a in (self.aux3, self.aux4) for m in a.modules()}
        mixers = [m for m in self.modules() if isinstance(m, DirectionalMixer)]
        paths = {id(p) for mm in mixers for p in (mm.square, mm.horizontal, mm.vertical, mm.point)}
        mixer_bn = {id(mm.bn) for mm in mixers}
        n = 0
        for m in self.modules():
            if id(m) in aux or isinstance(m, PlainDepthwiseMixer) or id(m) in paths or id(m) in mixer_bn:
                continue
            if isinstance(m, DirectionalMixer):
                n += m.square.params['w'].size + 2 * m.bn.params['g'].size
            else:
                n += sum(v.size for v in m.params.values())
        return int(n)


class PlainCNN(Layer):
    """Reference network of matched size: a stem, four stages of two 3 x 3 convolutions, a 1 x 1 head, pooling, linear."""

    def __init__(self, rng, num_classes=2, stem=24, widths=(32, 64, 112, 176), head=192):
        super().__init__()
        layers = [conv_bn(3, stem, 3, 2, rng)]; c = stem
        for w in widths:
            layers += [conv_bn(c, w, 3, 2, rng), conv_bn(w, w, 3, 1, rng)]; c = w
        layers.append(conv_bn(c, head, 1, 1, rng))
        self.features = Sequential(*layers); self.pool = GlobalAvgPool(); self.fc = Linear(head, num_classes, rng)

    def forward(self, x, aux=None):
        return self.fc.forward(self.pool.forward(self.features.forward(x)))

    def backward(self, dy, *_):
        return self.features.backward(self.pool.backward(self.fc.backward(dy)))


class SeqNet(Layer):
    """Sequence network: six strided one-dimensional convolutions (kernel 9) on the one-hot sequence, mean and maximum
    pooling over the 8,192 positions of the zero-padded input, linear classifier."""

    def __init__(self, rng, num_classes=2, widths=(32, 64, 96, 128, 192, 256), kernel=9):
        super().__init__()
        layers, c = [], 4
        for w in widths:
            layers += [Conv(c, w, (kernel, 1), 2, rng, pad=(kernel // 2, 0)), BN(w), ReLU()]; c = w
        self.features = Sequential(*layers); self.fc = Linear(2 * c, num_classes, rng); self.c = c

    def forward(self, x, aux=None):
        h = self.features.forward(x)
        self.cache = h
        return self.fc.forward(np.concatenate([h.mean((1, 2)), h.max((1, 2))], 1))

    def backward(self, dy, *_):
        h = self.cache; B, L, _, C = h.shape
        d = self.fc.backward(dy); dm, dx = d[:, :C], d[:, C:]
        dh = np.broadcast_to(dm[:, None, None, :] / L, h.shape).copy()
        am = h.reshape(B, L, C).argmax(1)
        np.add.at(dh.reshape(B, L, C), (np.arange(B)[:, None], am, np.arange(C)[None, :]), dx)
        return self.features.backward(dh)


class SeqORFNet(Layer):
    """Sequence-ORF network: the pooled features of the sequence network (512 values) and of a perceptron on the five
    ORF features (two hidden layers of 32 units) are concatenated and passed through dropout and a linear layer;
    linear heads on each branch alone provide auxiliary losses during training."""

    def __init__(self, rng, num_classes=2):
        super().__init__()
        self.seq = SeqNet(rng, num_classes); self.seq.fc = None
        self.orf1 = Linear(5, 32, rng); self.orf2 = Linear(32, 32, rng); self.r1 = ReLU(); self.r2 = ReLU()
        self.head_seq = Linear(512, num_classes, rng); self.head_orf = Linear(32, num_classes, rng)
        self.drop = Dropout(0.2, rng); self.fuse = Linear(512 + 32, num_classes, rng)

    def forward(self, x, aux=None):
        aux = self.training if aux is None else aux
        sq, orf = x
        h = self.seq.features.forward(sq); self.seq.cache = h
        fs = np.concatenate([h.mean((1, 2)), h.max((1, 2))], 1)
        fo = self.r2.forward(self.orf2.forward(self.r1.forward(self.orf1.forward(orf))))
        fused = self.fuse.forward(self.drop.forward(np.concatenate([fs, fo], 1)))
        self.cache_aux = aux
        if aux:
            return fused, self.head_seq.forward(fs), self.head_orf.forward(fo)
        return fused

    def backward(self, dfused, dseq=None, dorf=None):
        d = self.drop.backward(self.fuse.backward(dfused)); ds, do = d[:, :512], d[:, 512:]
        if self.cache_aux:
            ds = ds + self.head_seq.backward(dseq); do = do + self.head_orf.backward(dorf)
        self.orf1.backward(self.r1.backward(self.orf2.backward(self.r2.backward(do))))
        h = self.seq.cache; B, L, _, C = h.shape
        dm, dx = ds[:, :C], ds[:, C:]
        dh = np.broadcast_to(dm[:, None, None, :] / L, h.shape).copy()
        am = h.reshape(B, L, C).argmax(1)
        np.add.at(dh.reshape(B, L, C), (np.arange(B)[:, None], am, np.arange(C)[None, :]), dx)
        return self.seq.features.backward(dh)


def build(name, rng, num_classes=2):
    """Network of an identifier: frcgr_k5, frcgr_k6, frcgr_k7, frcgr_k5_summed, frcgr_k5_partition, frcgr_k5_permuted,
    frcgr_k5_onekernel (one kernel per depthwise layer), frcgr_k5_noneighbour (no neighbourhood blocks),
    plaincnn_k5, seq_net or seq_orf_net."""
    if name == 'plaincnn_k5':
        return PlainCNN(rng, num_classes)
    if name in FRCGR_INPUTS:
        return FRCGRNet(rng, num_classes, neighbour_after_stages=() if name.endswith('noneighbour') else (3, 4),
                        plain=name.endswith('onekernel'))
    if name == 'seq_net':
        return SeqNet(rng, num_classes)
    if name == 'seq_orf_net':
        return SeqORFNet(rng, num_classes)
    raise ValueError(f'unknown network {name}')


def count_macs(model, x):
    model.eval(); model.forward(x[:1] if not isinstance(x, tuple) else tuple(v[:1] for v in x), aux=False)
    return int(sum(getattr(m, 'macs', 0) for m in model.modules()) + sum(m.params['w'].size for m in model.modules() if isinstance(m, Linear)))


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predict(model, batches):
    """Coding probability of every input of an iterable of batches."""
    model.eval(); out = []
    for x in batches:
        z = model.forward(x, aux=False); model.free()
        out.append(softmax(z)[:, 1])
    return np.concatenate(out)


def load(path):
    """Network and identifier of a checkpoint written by scripts/genodirnet/train_numpy.py."""
    ck = np.load(path, allow_pickle=True)
    name = str(ck['model'])
    model = build(name, np.random.default_rng(0)); model.load_state_dict(ck['state'].item()); model.eval()
    return model, name
