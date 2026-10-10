"""导向矢量（array manifold）。

契约锚点，见 docs/CONTRACT.md 第 3 节：

    a_m(f, u) = exp( -j * 2*pi*f/c * dot(mics[m], u) )            远场
    a_m(f, p) = (r_ref/||p_m-p||) * exp( -j*2*pi*f/c * (||p_m-p|| - ||p||) )  近场

C++ 导出必须逐位复现本文件的算式。
"""
from __future__ import annotations

import numpy as np

from . import acoustics as ac

C_DEFAULT = ac.C_DEFAULT


def steering_vector(mics: np.ndarray, freqs, u_or_p, c: float = C_DEFAULT,
                    near_field: bool | None = None) -> np.ndarray:
    """返回 (F, M) complex128 导向矢量。

    freqs    : 标量或 (F,) 频率（Hz）
    u_or_p   : (3,) 方向单位向量（远场）或 (3,) 坐标（近场）
    near_field: None 时按 "是否给出距离" 自动判定（向量模长≈1 视为方向）
    """
    mics = np.asarray(mics, float)
    f = np.atleast_1d(np.asarray(freqs, float))
    tgt = np.asarray(u_or_p, float).ravel()
    if tgt.size != 3:
        raise ValueError("steering target must be a 3-vector")

    if near_field is None:
        near_field = not np.isclose(np.linalg.norm(tgt), 1.0, atol=1e-6)

    if near_field:
        d = ac.distance(mics, tgt)                     # (M,)
        amp = np.linalg.norm(tgt) / np.maximum(d, 1e-12)
        tau = (d - np.linalg.norm(tgt)) / c
    else:
        u = tgt / np.linalg.norm(tgt)
        amp = np.ones(mics.shape[0])
        tau = ac.delay_from_dir(mics, u, c)

    # (F,M) = amp * exp(-j2πf tau)
    phase = -2.0 * np.pi * f[:, None] * tau[None, :]
    return np.asarray(amp[None, :] * np.exp(1j * phase), dtype=np.complex128)


def steering_grid(mics: np.ndarray, freqs, dirs: np.ndarray,
                  c: float = C_DEFAULT) -> np.ndarray:
    """扫描方向网格 -> (F, G, M)。远场平面波假设。"""
    f = np.atleast_1d(np.asarray(freqs, float))
    tau = ac.delays_grid(mics, dirs, c)                 # (G,M)
    phase = -2.0 * np.pi * f[:, None, None] * tau[None, :, :]
    return np.exp(1j * phase).astype(np.complex128)


def plane_wave_delays_seconds(mics: np.ndarray, u: np.ndarray,
                              c: float = C_DEFAULT) -> np.ndarray:
    """远场时延（秒），供时域延迟求和实现使用。"""
    return ac.delay_from_dir(mics, u, c)


def propagation_delays_fractional(mics: np.ndarray, u: np.ndarray, fs: int,
                                  c: float = C_DEFAULT) -> np.ndarray:
    """以采样点为单位的时延（可为小数），供分数延迟 FIR 设计。"""
    return plane_wave_delays_seconds(mics, u, c) * float(fs)


def diffuse_coherence(mics: np.ndarray, freqs, c: float = C_DEFAULT,
                      kind: str = "spherical") -> np.ndarray:
    """扩散场（各向同性噪声场）相干矩阵 Gamma(f) -> (F,M,M)。

    spherical : Gamma_ij = sinc(2*pi*f*d_ij/c)   （三维扩散场，经典模型）
    cylindrical: Gamma_ij = J0(2*pi*f*d_ij/c)    （二维扩散场）
    identity   : 白噪声（各通道独立）
    """
    mics = np.asarray(mics, float)
    f = np.atleast_1d(np.asarray(freqs, float))
    M = mics.shape[0]
    i, j = np.triu_indices(M, k=1)
    d = np.linalg.norm(mics[i] - mics[j], axis=1)       # (P,)
    x = 2.0 * np.pi * f[:, None] * d[None, :] / c       # (F,P)

    if kind == "spherical":
        v = np.sinc(x / np.pi)                          # np.sinc 已含 pi 归一化
    elif kind == "cylindrical":
        v = _bessel_j0(x)                                # 本地实现，免 scipy
    elif kind == "identity":
        v = np.zeros_like(x)
    else:
        raise ValueError(f"unknown coherence kind: {kind}")

    G = np.zeros((f.size, M, M), dtype=np.complex128)
    idx = np.arange(M)
    G[:, idx, idx] = 1.0
    if kind != "identity":
        G[:, i, j] = v
        G[:, j, i] = v
    return G


def _bessel_j0(x: np.ndarray) -> np.ndarray:
    """零阶贝塞尔函数（多项式近似，Abramowitz & Stegun 9.4.1/9.4.3）。

    无需 scipy；精度 ~1e-7，足够用于相干矩阵。
    """
    x = np.asarray(x, float)
    ax = np.abs(x)
    out = np.empty_like(ax)
    small = ax < 3.0
    if np.any(small):
        y = (ax[small] / 3.0) ** 2
        out[small] = 1.0 - 2.2499997 * y + 1.2656208 * y ** 2 - 0.3163866 * y ** 3 \
            + 0.0444479 * y ** 4 - 0.0039444 * y ** 5 + 0.00021 * y ** 6
    if np.any(~small):
        z = 3.0 / ax[~small]
        f0 = 0.79788456 - 0.00000077 * z - 0.00552740 * z ** 2 - 0.00009512 * z ** 3 \
            + 0.00137237 * z ** 4 - 0.00072805 * z ** 5 + 0.00014476 * z ** 6
        th0 = ax[~small] - 0.78539816 - 0.04166397 * z - 0.00003954 * z ** 2 \
            + 0.00262573 * z ** 3 - 0.00054125 * z ** 4 - 0.00029333 * z ** 5 \
            + 0.00013558 * z ** 6
        out[~small] = f0 * np.cos(th0) / np.sqrt(ax[~small])
    return out


__all__ = [
    "steering_vector", "steering_grid", "plane_wave_delays_seconds",
    "propagation_delays_fractional", "diffuse_coherence",
]
