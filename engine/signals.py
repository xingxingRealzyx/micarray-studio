"""信号源生成与噪声/失配模型。

信号约定见 docs/CONTRACT.md 第 4 节。
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 信号源
# --------------------------------------------------------------------------
def tone(fs: int, n: int, freq: float, phase: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * freq * np.arange(n) / fs + phase)


def multitone(fs: int, n: int, freqs, amps=None, seed: int = 0) -> np.ndarray:
    freqs = np.atleast_1d(np.asarray(freqs, float))
    amps = np.ones(freqs.size) if amps is None else np.asarray(amps, float)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, freqs.size)
    return np.asarray(sum(a * np.sin(2 * np.pi * f * t + p)
                          for f, a, p in zip(freqs, amps, ph)))


def white_noise(fs: int, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def pink_noise(fs: int, n: int, seed: int = 0) -> np.ndarray:
    """1/f 噪声，用频域整形实现（无 scipy）。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.ones_like(f)
    nz = f > 0
    shape[nz] = 1.0 / np.sqrt(f[nz])
    shape[0] = 0.0
    y = np.fft.irfft(W * shape, n=n)
    return y / (np.std(y) + 1e-12)


def chirp(fs: int, n: int, f0: float = 200.0, f1: float = 4000.0) -> np.ndarray:
    t = np.arange(n) / fs
    T = max(n / fs, 1e-9)
    k = (f1 - f0) / T
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))


def speech_like(fs: int, n: int, f0: float = 120.0, seed: int = 0,
                formants=(700.0, 1220.0, 2600.0)) -> np.ndarray:
    """类语音信号：谐波激励 + 三个共振峰，带音节包络。

    不是真实语音，但保留了谐波结构、谱包络与 4 Hz 音节调制 ——
    足以让增强算法（后置滤波、谱增益）表现出真实行为。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    # 基频缓慢起伏
    f0_t = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28)))
    ph = 2 * np.pi * np.cumsum(f0_t) / fs
    exc = np.zeros(n)
    for h in range(1, 41):
        if h * f0 > fs / 2 * 0.9:
            break
        exc += (1.0 / h) * np.sin(h * ph)
    # 共振峰谐振器（简单二阶 IIR）
    y = exc.copy()
    for fc in formants:
        r = 0.94
        w0 = 2 * np.pi * fc / fs
        a1, a2 = -2 * r * np.cos(w0), r * r
        out = np.zeros(n)
        for i in range(2, n):
            out[i] = y[i] - a1 * out[i - 1] - a2 * out[i - 2]
        y = out
    # 音节包络：约 4 Hz，带随机停顿
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    env *= (rng.random(max(1, n // (fs // 4))) > 0.25).repeat(
        int(np.ceil(n / max(1, n // (fs // 4)))))[:n]
    y = y * env
    y -= y.mean()
    return y / (np.std(y) + 1e-12)


GENERATORS = {
    "tone": lambda fs, n, p: tone(fs, n, p.get("freq", 1000.0),
                                  p.get("phase", 0.0)),
    "multitone": lambda fs, n, p: multitone(fs, n, p.get("freqs", [500, 1000, 2000]),
                                            p.get("amps"), int(p.get("seed", 0))),
    "white": lambda fs, n, p: white_noise(fs, n, int(p.get("seed", 0))),
    "pink": lambda fs, n, p: pink_noise(fs, n, int(p.get("seed", 0))),
    "chirp": lambda fs, n, p: chirp(fs, n, p.get("f0", 200.0), p.get("f1", 4000.0)),
    "speech": lambda fs, n, p: speech_like(fs, n, p.get("f0", 120.0),
                                           int(p.get("seed", 0)),
                                           tuple(p.get("formants", (700, 1220, 2600)))),
}


def generate(kind: str, fs: int, n: int, params: dict | None = None) -> np.ndarray:
    if kind not in GENERATORS:
        raise ValueError(f"unknown signal kind: {kind}")
    return np.asarray(GENERATORS[kind](fs, n, params or {}), dtype=np.float64)


# --------------------------------------------------------------------------
# 噪声与失配
# --------------------------------------------------------------------------
def add_noise(X: np.ndarray, snr_db: float, kind: str = "white",
              seed: int = 0, coherent: bool = False) -> tuple[np.ndarray, float]:
    """按目标 SNR 加噪。SNR 定义为所有通道平均的信号功率 / 噪声功率。

    coherent=False 时各通道噪声独立（麦克风自噪）；
    True 时各通道共用同一噪声（相干干扰，如远处空调噪声）。
    """
    X = np.asarray(X, float)
    M, N = X.shape
    sig_p = float(np.mean(X ** 2))
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    n = white_noise(0, N, seed) if kind == "white" else pink_noise(0, N, seed)
    n = n / (np.std(n) + 1e-12)
    if coherent:
        noise = np.repeat(n[None, :], M, axis=0)
    else:
        rng = np.random.default_rng(seed + 991)
        noise = rng.standard_normal((M, N))
        if kind != "white":
            base = pink_noise(0, N, seed)
            noise = np.stack([np.convolve(base, rng.standard_normal(64), mode="same")
                              for _ in range(M)])
        noise = (noise - noise.mean(axis=1, keepdims=True)) / \
            (noise.std(axis=1, keepdims=True) + 1e-12)
    noise *= np.sqrt(noise_p)
    out = X + noise
    snr_actual = 10 * np.log10(np.mean(X ** 2) / max(np.mean(noise ** 2), 1e-30))
    return out, float(snr_actual)


def apply_mismatch(X: np.ndarray, gain_db_std: float = 0.0,
                   phase_deg_std: float = 0.0, pos_std_m: float = 0.0,
                   seed: int = 0, fs: int = 16000, c: float = 343.0,
                   direction: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """通道幅相/位置失配。这是真实阵列旁瓣抬头的首要原因。

    - 增益误差：直接乘在时域
    - 相位误差：频域乘 exp(jφ)，φ 为**频率无关**的固定相偏（最坏情况，
      比线性相位更难校准，能真实压低旁瓣抑制）
    - 位置误差：若给 direction，则折算为等效时延误差 exp(-j2πf·Δτ)，
      Δτ = -dot(dpos, u)/c；否则只记录不施加
    三者相互独立。
    """
    X = np.asarray(X, float)
    M, N = X.shape
    rng = np.random.default_rng(seed)
    g = 10 ** (rng.normal(0, gain_db_std, M) / 20.0) if gain_db_std > 0 else np.ones(M)
    p = rng.normal(0, phase_deg_std, M) if phase_deg_std > 0 else np.zeros(M)
    dpos = rng.normal(0, pos_std_m, (M, 3)) if pos_std_m > 0 else np.zeros((M, 3))

    Y = X * g[:, None]
    if np.any(p != 0) or (np.any(dpos != 0) and direction is not None):
        f = np.fft.rfftfreq(N, 1.0 / fs)
        phi = np.deg2rad(p)                                  # (M,)
        if direction is not None and np.any(dpos != 0):
            u = np.asarray(direction, float)
            u = u / max(np.linalg.norm(u), 1e-12)
            dtau = -(dpos @ u) / c                           # (M,) 等效时延误差
        else:
            dtau = np.zeros(M)
        ph = phi[None, :] - 2.0 * np.pi * f[:, None] * dtau[None, :]   # (F,M)
        Yf = np.fft.rfft(Y, axis=1) * np.exp(1j * ph).T
        Y = np.fft.irfft(Yf, n=N, axis=1)
    return Y, {"gain_db": [round(float(v), 4) for v in 20 * np.log10(g)],
               "phase_deg": [round(float(v), 4) for v in p],
               "pos_err_m": [[round(float(v), 6) for v in r] for r in dpos],
               "pos_err_applied": bool(direction is not None and np.any(dpos != 0))}


def snr_db(X_sig: np.ndarray, X_noise: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(X_sig ** 2) / max(np.mean(X_noise ** 2), 1e-30)))


def segmental_snr_db(X_ref: np.ndarray, X_test: np.ndarray, fs: int,
                     frame_ms: float = 32.0) -> float:
    """分段 SNR（语音增强的标准评价），逐帧算 SNR 再对语音帧取平均。"""
    X_ref = np.asarray(X_ref, float).ravel()
    X_test = np.asarray(X_test, float).ravel()
    L = max(int(fs * frame_ms / 1000.0), 16)
    K = max(X_ref.size // L, 1)
    vals = []
    for k in range(K):
        r = X_ref[k * L:(k + 1) * L]
        t = X_test[k * L:(k + 1) * L]
        if r.size < L:
            break
        pr = float(np.mean(r ** 2))
        if pr < 1e-10 * max(float(np.mean(X_ref ** 2)), 1e-20):
            continue                      # 静音帧不计入
        pn = float(np.mean((t - r) ** 2))
        vals.append(10 * np.log10(pr / max(pn, 1e-30)))
    return float(np.mean(vals)) if vals else float("nan")


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """尺度不变信噪比（scale-invariant SDR, dB）。语音增强常用客观指标。"""
    ref = np.asarray(ref, float).ravel()
    est = np.asarray(est, float).ravel()
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(est, ref) / max(np.dot(ref, ref), 1e-30))
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10(np.sum(target ** 2) / max(np.sum(noise ** 2), 1e-30)))


__all__ = [
    "tone", "multitone", "white_noise", "pink_noise", "chirp", "speech_like",
    "generate", "GENERATORS", "add_noise", "apply_mismatch", "snr_db",
    "segmental_snr_db", "si_sdr",
]
