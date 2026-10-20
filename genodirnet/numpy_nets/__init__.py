"""NumPy implementation of the networks of the standard schedule, with a few C kernels (compiled on first use).

Used to train the networks of the paired comparison with and without fragment-aware training and of the
controls of the FR-CGR on processors, and to score trained networks on any set of sequences without PyTorch."""
import ctypes
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None

def _build():
    src = os.path.join(_HERE, 'kernels.c')
    out = os.path.join(_HERE, f'kernels_{sys.platform}_{os.uname().machine}.so')
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    for cc in ('gcc', 'clang', 'cc'):
        try:
            subprocess.run([cc, '-O3', '-march=haswell', '-fopenmp', '-ffast-math', '-shared', '-fPIC', '-o', out, src, '-lm'], check=True, capture_output=True)
            return out
        except (OSError, subprocess.CalledProcessError) as e:
            last = e
    try:
        subprocess.run(['cc', '-O3', '-ffast-math', '-shared', '-fPIC', '-o', out, src, '-lm'], check=True, capture_output=True)
        return out
    except Exception:
        raise RuntimeError(f'could not compile {src}: {last}')

def lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_build())
        f32 = ctypes.POINTER(ctypes.c_float); u8 = ctypes.POINTER(ctypes.c_ubyte); i = ctypes.c_int
        for name in ('dw_fwd', 'dw_bwd_x', 'dw_bwd_w'):
            getattr(_LIB, name).argtypes = [f32, f32, f32] + [i] * 11
        for name in ('im2col', 'col2im'):
            getattr(_LIB, name).argtypes = [f32, f32] + [i] * 11
        _LIB.bn_fwd.argtypes = [f32] * 7 + [i, i, i, ctypes.c_float]
        _LIB.bn_bwd.argtypes = [f32] * 7 + [i, i, i]
        _LIB.bn_relu_fwd.argtypes = [f32, f32, f32, f32, f32, u8, f32, f32, i, i, i, ctypes.c_float]
        _LIB.relu_fwd.argtypes = [f32, f32, u8, ctypes.c_size_t]
        _LIB.relu_bwd.argtypes = [f32, u8, f32, ctypes.c_size_t]
        _LIB.maxpool2_fwd.argtypes = [f32, f32, u8, i, i, i, i]
        _LIB.maxpool2_bwd.argtypes = [f32, u8, f32, i, i, i, i]
    return _LIB

def ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def uptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

def c32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def set_threads(n):
    """Number of OpenMP threads of the C kernels and of the BLAS library."""
    try:
        ctypes.CDLL('libgomp.so.1').omp_set_num_threads(int(n))
    except OSError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass
