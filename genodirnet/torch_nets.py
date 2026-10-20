"""PyTorch networks of the standard schedule (Methods): the FR-CGR network, the sequence network and the
sequence-ORF network, and ResNet18 on the FR-CGR as a reference."""
import copy
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FRCGRConfig:
    """Architecture of the FR-CGR network (0.78 million parameters, 6.0 million MACs for k = 5)."""
    num_classes: int = 2
    stem_width: int = 16
    widths: Tuple[int, ...] = (24, 40, 72, 112, 160)
    depths: Tuple[int, ...] = (1, 2, 2, 2, 1)
    expansions: Tuple[float, ...] = (2.0, 3.0, 3.0, 3.0, 2.0)
    stage_kernels: Tuple[Tuple[int, ...], ...] = ((3,), (5, 3), (3, 5), (5, 3), (5,))
    neighbour_after_stages: Tuple[int, ...] = (3, 4)
    head_width: int = 96
    dropout: float = 0.2
    drop_path_rate: float = 0.08


class DropPath(nn.Module):
    """Stochastic depth: the residual branch of a block is dropped for a whole example with the given probability."""

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvBN(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel, stride=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _pad_kernel(kernel, size):
    h, w = kernel.shape[-2:]
    ph, pw = size - h, size - w
    return F.pad(kernel, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


class DirectionalMixer(nn.Module):
    """Depthwise convolution whose kernel is the sum of five parallel paths (k x k, 1 x k, k x 1, 1 x 1 and the
    identity) during training; after training the paths are merged into one k x k kernel (switch_to_deploy)."""

    def __init__(self, channels, kernel_size, deploy=False):
        super().__init__()
        self.channels, self.kernel_size, self.deploy = channels, kernel_size, deploy
        self.post_bn = nn.BatchNorm2d(channels)
        k = kernel_size
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
        else:
            self.square = nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
            self.horizontal = nn.Conv2d(channels, channels, (1, k), padding=(0, k // 2), groups=channels, bias=False)
            self.vertical = nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0), groups=channels, bias=False)
            self.point = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

    def forward(self, x):
        if self.deploy:
            return self.post_bn(self.reparam(x))
        return self.post_bn(self.square(x) + self.horizontal(x) + self.vertical(x) + self.point(x) + x)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        k = self.kernel_size
        kernel = self.square.weight + _pad_kernel(self.horizontal.weight, k) + _pad_kernel(self.vertical.weight, k) + _pad_kernel(self.point.weight, k)
        kernel[:, 0, k // 2, k // 2] += 1.0
        self.reparam = nn.Conv2d(self.channels, self.channels, k, padding=k // 2, groups=self.channels, bias=False).to(kernel.device)
        self.reparam.weight.copy_(kernel)
        for name in ('square', 'horizontal', 'vertical', 'point'):
            delattr(self, name)
        self.deploy = True


class DirectionalBlock(nn.Module):
    """Inverted residual block: 1 x 1 expansion, directional depthwise mixer, 1 x 1 projection, residual connection."""

    def __init__(self, channels, expansion, kernel_size, drop_path):
        super().__init__()
        hidden = int(math.ceil(channels * expansion / 8.0) * 8)
        self.expand = ConvBNReLU(channels, hidden, 1)
        self.mixer = DirectionalMixer(hidden, kernel_size)
        self.mixer_activation = nn.ReLU(inplace=True)
        self.project = ConvBN(hidden, channels, 1)
        self.drop_path = DropPath(drop_path)
        self.output_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.project(self.mixer_activation(self.mixer(self.expand(x))))
        return self.output_activation(x + self.drop_path(y))


class NeighbourhoodBlock(nn.Module):
    """Adds trained projections of the averages over the axial and the diagonal neighbours of every cell, which
    follow the suffix structure of the chaos game grid."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.axial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.diagonal = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.axial.weight.requires_grad_(False); self.diagonal.weight.requires_grad_(False)
        self.axial_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.diagonal_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.post_bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.reset_fixed_kernels()

    @torch.no_grad()
    def reset_fixed_kernels(self):
        ax = torch.tensor([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]])
        dg = torch.tensor([[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]])
        self.axial.weight.copy_(ax.reshape(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))
        self.diagonal.weight.copy_(dg.reshape(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))

    def forward(self, x):
        message = self.axial_projection(self.axial(x)) + self.diagonal_projection(self.diagonal(x))
        return self.activation(x + self.post_bn(message))


class PyramidHead(nn.Module):
    """Projections of the outputs of stages 3 to 5, global average pooling, dropout and a linear output layer."""

    def __init__(self, channels, head_width, num_classes, dropout):
        super().__init__()
        self.projections = nn.ModuleList([ConvBNReLU(c, head_width, 1) for c in channels])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_width * len(channels), num_classes)

    def forward(self, features):
        v = [self.pool(p(f)).flatten(1) for p, f in zip(self.projections, features)]
        return self.classifier(self.dropout(torch.cat(v, dim=1)))


class AuxiliaryHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(x).flatten(1))


class FRCGRNet(nn.Module):
    """FR-CGR network: stem, five stages of directional blocks, neighbourhood blocks after stages 4 and 5, a pyramid
    head on stages 3 to 5 and auxiliary heads on stages 3 and 4 (training only)."""

    def __init__(self, config=FRCGRConfig()):
        super().__init__()
        c = self.config = config
        self.stem = ConvBNReLU(3, c.stem_width, 3, stride=2)
        rates = torch.linspace(0.0, c.drop_path_rate, sum(c.depths)).tolist(); i = 0
        stages, cin = [], c.stem_width
        for s, (w, d, e, ks) in enumerate(zip(c.widths, c.depths, c.expansions, c.stage_kernels)):
            stride = 1 if s == 0 else 2
            mods = [ConvBNReLU(cin, w, 1 if stride == 1 else 3, stride=stride)]
            for b in range(d):
                mods.append(DirectionalBlock(w, e, ks[b], rates[i])); i += 1
            if s in c.neighbour_after_stages:
                mods.append(NeighbourhoodBlock(w))
            stages.append(nn.Sequential(*mods)); cin = w
        self.stages = nn.ModuleList(stages)
        self.head = PyramidHead(c.widths[-3:], c.head_width, c.num_classes, c.dropout)
        self.aux_stage3 = AuxiliaryHead(c.widths[-3], c.num_classes)
        self.aux_stage4 = AuxiliaryHead(c.widths[-2], c.num_classes)
        self.apply(self._init)
        for m in self.modules():
            if isinstance(m, DirectionalBlock):
                nn.init.zeros_(m.project.bn.weight); nn.init.zeros_(m.project.bn.bias)
            elif isinstance(m, NeighbourhoodBlock):
                m.reset_fixed_kernels(); nn.init.zeros_(m.post_bn.weight); nn.init.zeros_(m.post_bn.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_aux=None):
        x = self.stem(x); feats = []
        for stage in self.stages:
            x = stage(x); feats.append(x)
        f3, f4, f5 = feats[-3:]
        logits = self.head((f3, f4, f5))
        if (self.training if return_aux is None else return_aux):
            return logits, self.aux_stage3(f3), self.aux_stage4(f4)
        return logits

    @torch.no_grad()
    def deployment_copy(self):
        """Copy with the five paths of every depthwise kernel merged into one kernel (the same outputs)."""
        m = copy.deepcopy(self).eval()
        for mod in m.modules():
            if isinstance(mod, DirectionalMixer):
                mod.switch_to_deploy()
        return m


def frcgr_net(name):
    """FR-CGR network of an identifier (frcgr_k5, frcgr_k6, frcgr_k7)."""
    return FRCGRNet()


class SeqNet(nn.Module):
    """Sequence network: six one-dimensional convolutions (kernel 9, stride 2) on the one-hot sequence of 8,192
    positions, mean and maximum pooling, linear classifier."""

    def __init__(self, num_classes=2, widths=(32, 64, 96, 128, 192, 256), kernel=9):
        super().__init__()
        layers, c = [], 4
        for w in widths:
            layers += [nn.Conv1d(c, w, kernel, stride=2, padding=kernel // 2, bias=False), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]
            c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(2 * c, num_classes)

    def forward(self, x):
        h = self.features(x)
        return self.classifier(torch.cat([h.mean(-1), h.amax(-1)], 1))


class SeqORFNet(nn.Module):
    """Sequence-ORF network: pooled features of the sequence network (512) and of a perceptron on the five ORF
    features (32) are concatenated and passed through dropout and a linear layer; linear heads on each branch give
    auxiliary losses during training."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.seq = SeqNet(num_classes); self.seq.classifier = nn.Identity()
        self.orf = nn.Sequential(nn.Linear(5, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.heads = nn.ModuleDict({'seq': nn.Linear(512, num_classes), 'orf': nn.Linear(32, num_classes)})
        self.drop = nn.Dropout(0.2)
        self.fuse = nn.Linear(512 + 32, num_classes)

    def forward(self, sq, orf):
        fs, fo = self.seq(sq), self.orf(orf)
        fused = self.fuse(self.drop(torch.cat([fs, fo], 1)))
        return fused, {'seq': self.heads['seq'](fs), 'orf': self.heads['orf'](fo)}


def build(name):
    """Network of an identifier: frcgr_k5, frcgr_k6, frcgr_k7, resnet18_k7, seq_net or seq_orf_net."""
    if name.startswith('frcgr_'):
        return frcgr_net(name)
    if name == 'resnet18_k7':
        import torchvision
        return torchvision.models.resnet18(weights=None, num_classes=2)
    if name == 'seq_net':
        return SeqNet()
    if name == 'seq_orf_net':
        return SeqORFNet()
    raise ValueError(f'unknown network {name}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, *inputs):
    """Multiply-accumulate operations of one forward pass (convolutions and linear layers) for one input."""
    macs = 0

    def hook(m, i, o):
        nonlocal macs
        out = o[0] if isinstance(o, tuple) else o
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            macs += out[0].numel() * (m.in_channels // m.groups) * int(torch.tensor(m.kernel_size).prod())
        elif isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(*[x[:1] for x in inputs])
    for h in hs:
        h.remove()
    return int(macs)
