"""窗函数、STFT/ISTFT、以及一个独立 DFT（用于校验 numpy FFT）。

契约见 docs/CONTRACT.md 第 5 节：单边谱、不做幅度补偿、窗表可导出给 C++。
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 窗
# --------------------------------------------------------------------------
WINDOW_NAMES = ("rect", "hann", "hamming", "blackman", "blackmanharris", "flattop")


def window(name: str, n: int, periodic: bool = True) -> np.ndarray:
    """窗函数，与 numpy.hanning / scipy.signal.get_window 的系数一致。

    periodic=True 时用 N 点周期窗（STFT 标准做法），
    公式里以 N 为分母；periodic=False 时以 N-1 为分母（对称窗）。
    """
    n = int(n)
    if n <= 0:
        raise ValueError("window length must be positive")
    denom = n if periodic else max(n - 1, 1)
    k = np.arange(n, dtype=np.float64)

    if name == "rect":
        w = np.ones(n)
    elif name == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * k / denom)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * k / denom)
    elif name == "blackman":
        w = 0.42 - 0.5 * np.cos(2 * np.pi * k / denom) \
            + 0.08 * np.cos(4 * np.pi * k / denom)
    elif name == "blackmanharris":
        # 4 项最小旁瓣窗（harris 1978），旁瓣 ~ -92 dB
        w = 0.35875 - 0.48829 * np.cos(2 * np.pi * k / denom) \
            + 0.14128 * np.cos(4 * np.pi * k / denom) \
            - 0.01168 * np.cos(6 * np.pi * k / denom)
    elif name == "flattop":
        w = 0.21557895 - 0.41663158 * np.cos(2 * np.pi * k / denom) \
            + 0.277263158 * np.cos(4 * np.pi * k / denom) \
            - 0.083578947 * np.cos(6 * np.pi * k / denom) \
            + 0.006947368 * np.cos(8 * np.pi * k / denom)
    else:
        raise ValueError(f"unknown window: {name}")
    return np.asarray(w, dtype=np.float64)


def window_table(name: str = "blackmanharris", n_fft: int = 512,
                 sig: int = 9) -> list[float]:
    """导出给 C++ 的窗表：四舍五入到 sig 位有效数字。

    Python 侧在使用导出表时也必须用同一张表，否则一致性无从谈起。
    """
    w = window(name, n_fft, periodic=True)
    return [float(f"%.{sig}g" % v) for v in w]


# --------------------------------------------------------------------------
# STFT
# --------------------------------------------------------------------------
def frame_count(n_samples: int, n_fft: int, hop: int) -> int:
    """与 stft() 一致的帧数：末尾不足一帧则补零。"""
    if n_samples <= 0:
        return 0
    if n_samples <= n_fft:
        return 1
    return 1 + int(np.ceil((n_samples - n_fft) / hop))


def stft(x: np.ndarray, fs: int, n_fft: int = 512, hop: int | None = None,
         win: str = "blackmanharris", win_table: list[float] | None = None,
         center: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """短时傅里叶变换。

    x      : (N,) 或 (M,N) 实数信号
    center : True 时在首尾各补 n_fft//2 个零（帧数 = ceil(N/hop)+1）
    return : Xf (..., K, F) complex128, freqs (F,)
    """
    x = np.asarray(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    M, N = x.shape
    hop = hop or n_fft // 2
    w = np.asarray(win_table, dtype=np.float64) if win_table is not None \
        else window(win, n_fft)
    if w.size != n_fft:
        raise ValueError("win_table length must equal n_fft")

    if center:
        pad = n_fft // 2
        x = np.pad(x, ((0, 0), (pad, pad)))
        N = x.shape[1]

    K = frame_count(N, n_fft, hop)
    padded = np.zeros((M, (K - 1) * hop + n_fft), dtype=np.float64)
    padded[:, :N] = x
    # (M, K, n_fft) 滑窗视图
    idx = np.arange(n_fft)[None, :] + hop * np.arange(K)[:, None]      # (K,n_fft)
    frames = padded[:, idx]                                           # (M,K,n_fft)
    Xf = np.fft.rfft(frames * w[None, None, :], n=n_fft, axis=-1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return (Xf[0] if single else Xf), freqs


def istft(Xf: np.ndarray, fs: int, n_fft: int = 512, hop: int | None = None,
          win: str = "blackmanharris", win_table: list[float] | None = None,
          length: int | None = None, center: bool = False) -> np.ndarray:
    """重叠相加逆变换。

    **窗端归一化是一个真陷阱，必须理解清楚**：
    Blackman-Harris 窗的两端 `w[0]^2 ≈ 3.6e-9`。若频谱被修改过（波束权重、
    谱增益、相位掩蔽……），WOLA 恒等式不再逐点成立，此时用
    `y/wsum` 归一化会把首尾十几个样点放大到 1e4 倍量级 —— 实测随机相位
    掩蔽下 `max|y|` 从 3.9 涨到 9121（2339 倍），SI-SDR 直接崩掉。

    因此这里做两件事：
    1. `wsum` 低于 `1e-6 * max(wsum)` 的位置**直接置零**（该处有效抽头太少，
       任何归一化都是放大数值噪声）；
    2. 提供 `center=True`：首尾各补 n_fft//2 个零再变换，两端一定是完整
       重叠区，从根本上不存在窗端问题。做频域修改时**应当用 center=True**。

    注意 stft/istft 的 center 必须一致，且 center=True 时输出比输入长
    n_fft 个样点，需自行裁掉首尾各 n_fft//2。
    """
    Xf = np.asarray(Xf)
    single = Xf.ndim == 2
    if single:
        Xf = Xf[None, ...]
    M, K, F = Xf.shape
    hop = hop or n_fft // 2
    w = np.asarray(win_table, dtype=np.float64) if win_table is not None \
        else window(win, n_fft)

    frames = np.fft.irfft(Xf, n=n_fft, axis=-1) * w[None, None, :]    # (M,K,n_fft)
    out_len = (K - 1) * hop + n_fft
    y = np.zeros((M, out_len))
    wsum = np.zeros(out_len)
    for k in range(K):
        y[:, k * hop:k * hop + n_fft] += frames[:, k, :]
        wsum[k * hop:k * hop + n_fft] += w ** 2
    # 归一化只在"有效重叠足够"的位置做。
    # 阈值取 0.05*max 而不是一个极小值：BH 窗端点的 wsum 比稳态小 3e7 倍，
    # 用极小阈值（如 1e-6）等于允许放大 1e5 倍，实测随机相位掩蔽下 max|y|
    # 仍被放大 287 倍。阈值以下直接置零 —— 那里本来就没有足够多的重叠抽头，
    # 任何"保幅度"的折中都只是把数值噪声放大成信号。
    # 需要完整覆盖两端时请用 center=True（首尾各有 n_fft//2 的补零重叠区）。
    if wsum.size == 0:
        return y[0] if single else y
    thr = 0.05 * float(wsum.max())
    ok = wsum > thr
    y[:, ok] /= wsum[ok]
    y[:, ~ok] = 0.0
    if center:
        pad = n_fft // 2
        y = y[:, pad:y.shape[1] - pad] if y.shape[1] > 2 * pad else y[:, :0]
    if length is not None:
        y = y[:, :length]
    return y[0] if single else y


# --------------------------------------------------------------------------
# 独立 DFT：用于校验 numpy.fft 与 C++ 实现
# --------------------------------------------------------------------------
def dft_matrix(n_fft: int) -> np.ndarray:
    """(F,N) 单边 DFT 矩阵，W[f,n] = exp(-j2πfn/N)。"""
    n = np.arange(n_fft)
    f = np.arange(n_fft // 2 + 1)
    return np.exp(-2j * np.pi * np.outer(f, n) / n_fft)


def dft_single_sided(x: np.ndarray, n_fft: int) -> np.ndarray:
    """朴素 DFT（O(N^2)），只用于小尺寸校验。

    x 最后一维为时间 -> 输出 (..., F)。与 np.fft.rfft(x, n=n_fft) 数值等价
    （x 长度不足 n_fft 时需自行补零）。
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.shape[-1] < n_fft:
        pad = [(0, 0)] * (x.ndim - 1) + [(0, n_fft - x.shape[-1])]
        x = np.pad(x, pad)
    # W (F,N)：out[...,f] = sum_n x[...,n] * W[f,n]
    return np.einsum("...n,fn->...f", x, dft_matrix(n_fft))


# --------------------------------------------------------------------------
# 窗与增益指标
# --------------------------------------------------------------------------
def window_metrics(name: str, n_fft: int = 512) -> dict:
    w = window(name, n_fft)
    cg = float(w.mean())                       # 相干增益
    return {
        "name": name,
        "coherent_gain": round(cg, 9),
        "coherent_gain_db": round(20 * np.log10(max(cg, 1e-12)), 4),
        "enbw_bins": round(float(n_fft * (w ** 2).sum() / (w.sum() ** 2)), 4),
        "amplitude_scale": round(2.0 / w.sum(), 9),
    }


__all__ = [
    "WINDOW_NAMES", "window", "window_table", "stft", "istft", "frame_count",
    "dft_matrix", "dft_single_sided", "window_metrics",
]
