"""时延估计：GCC-PHAT（含抛物插值细化）与多源时延聚类。

GCC-PHAT 是声源定位的地基。本实现为确定性实现（无随机性），
便于与 C++ 导出逐位比对。
"""
from __future__ import annotations

import numpy as np


def gcc_phat(x: np.ndarray, y: np.ndarray, fs: int, n_fft: int | None = None,
             max_delay: float | None = None, interp: int = 1,
             eps: float = 1e-12) -> tuple[float, np.ndarray, np.ndarray]:
    """广义互相关（相位变换加权）。

    x, y      : (N,) 两通道信号
    max_delay : 只在该时延范围内搜索（秒）；None 则搜索全部
    interp    : 频域补零上采样倍数（>=1），用于亚采样精度
    return    : (tau_hat 秒, lags 秒, cc 相关序列)

    约定：tau > 0 表示 y 比 x **滞后** tau（即信号先到 x）。
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = x.size

    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(2 * n, 16))))
    if interp > 1:
        n_fft *= int(interp)

    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    # 约定 tau>0 表示 y 滞后 x。y[n]=x[n-D] => Y = X*exp(-jwd)，
    # 故 Y*conj(X) = |X|^2*exp(-jwd)，其 irfft 的峰落在 n=D。
    G = Y * np.conj(X)
    G /= (np.abs(G) + eps)

    cc = np.fft.irfft(G, n=n_fft)
    cc = np.concatenate([cc[n_fft // 2:], cc[:n_fft // 2]])    # 零时延居中
    lags = (np.arange(n_fft) - n_fft // 2) / float(fs)

    if max_delay is not None:
        m = np.abs(lags) <= max_delay
        if not np.any(m):
            raise ValueError("max_delay too small for this n_fft")
    else:
        m = np.ones(n_fft, bool)

    idx_local = int(np.argmax(np.where(m, cc, -np.inf)))
    tau = float(lags[idx_local])

    # 抛物插值细化（在相关峰邻域做三点拟合）
    if 0 < idx_local < n_fft - 1:
        y0, y1, y2 = cc[idx_local - 1], cc[idx_local], cc[idx_local + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > eps:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                tau += float(delta) / float(fs)

    return tau, lags, cc


def gcc_phat_matrix(X: np.ndarray, fs: int, pairs: np.ndarray | None = None,
                    n_fft: int | None = None, max_delay: float | None = None,
                    interp: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """对多通道信号一次性算所有阵元对。

    X     : (M,N)
    pairs : (P,2) 指定阵元对；None 表示所有 i<j
    return: (taus (P,), pairs (P,2))
    """
    X = np.asarray(X, float)
    M = X.shape[0]
    if pairs is None:
        pairs = np.array(list(zip(*np.triu_indices(M, k=1))), dtype=int)
    if pairs.size == 0:
        return np.zeros(0), pairs.reshape(0, 2)

    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(2 * X.shape[1], 16))))
    if interp > 1:
        n_fft *= int(interp)

    S = np.fft.rfft(X, n=n_fft, axis=1)          # (M,F)
    S /= (np.abs(S) + 1e-12)

    taus = np.empty(pairs.shape[0])
    for p, (i, j) in enumerate(pairs):
        # 与 gcc_phat 同约定：tau>0 表示 j 滞后 i
        G = S[j] * np.conj(S[i])
        cc = np.fft.irfft(G, n=n_fft)
        cc = np.concatenate([cc[n_fft // 2:], cc[:n_fft // 2]])
        lags = (np.arange(n_fft) - n_fft // 2) / float(fs)
        mask = np.ones(n_fft, bool) if max_delay is None else np.abs(lags) <= max_delay
        k = int(np.argmax(np.where(mask, cc, -np.inf)))
        tau = float(lags[k])
        if 0 < k < n_fft - 1:
            y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
            den = y0 - 2.0 * y1 + y2
            if abs(den) > 1e-12:
                delta = 0.5 * (y0 - y2) / den
                if abs(delta) <= 1.0:
                    tau += float(delta) / float(fs)
        taus[p] = tau
    return taus, pairs


def tdoa_from_delays(taus: np.ndarray, pairs: np.ndarray,
                     mics: np.ndarray) -> np.ndarray:
    """时延 -> 差分传播距离（米）。返回 (P,)。"""
    return np.asarray(taus, float) * 343.0


def delay_residual(taus: np.ndarray, pairs: np.ndarray, mics: np.ndarray,
                   u_or_p: np.ndarray, c: float = 343.0,
                   near_field: bool = False) -> np.ndarray:
    """给定候选方向/位置，计算 (观测时延 - 模型时延) 的残差向量。

    这是闭式定位、非线性最小二乘、以及自校验测试的共用内核。
    """
    from . import acoustics as ac

    tgt = np.asarray(u_or_p, float).ravel()
    if near_field:
        tau_model = ac.delay_from_point(mics, tgt, c)
    else:
        tau_model = ac.delay_from_dir(mics, tgt / np.linalg.norm(tgt), c)
    # 观测 tau_ij = tau_j - tau_i （正表示 j 滞后）
    model_pairs = tau_model[pairs[:, 1]] - tau_model[pairs[:, 0]]
    return np.asarray(taus, float) - model_pairs


def cluster_delays(X: np.ndarray, fs: int, mics: np.ndarray, pair: tuple[int, int],
                   n_clusters: int = 2, n_fft: int = 256, hop: int | None = None,
                   max_delay: float | None = None, iters: int = 12,
                   seed: int = 0) -> dict:
    """多声源场景：分帧算 GCC-PHAT，再对时延做一维 k-means 聚类。

    这是多源 TDOA 的常用做法 —— 不同时刻主导声源不同，
    时延直方图上会出现多个峰，聚类即可分离。
    """
    from . import dsp

    X = np.asarray(X, float)
    hop = hop or n_fft // 2
    i, j = pair
    K = dsp.frame_count(X.shape[1], n_fft, hop)
    if K == 0:
        return {"centers": np.zeros(0), "labels": np.zeros(0, int), "hist": None}

    taus = np.empty(K)
    for k in range(K):
        s = k * hop
        seg_i = X[i, s:s + n_fft]
        seg_j = X[j, s:s + n_fft]
        if seg_i.size < n_fft:
            seg_i = np.pad(seg_i, (0, n_fft - seg_i.size))
            seg_j = np.pad(seg_j, (0, n_fft - seg_j.size))
        taus[k], _, _ = gcc_phat(seg_i, seg_j, fs, n_fft=n_fft,
                                 max_delay=max_delay, interp=1)

    lo, hi = float(taus.min()), float(taus.max())
    if hi - lo < 1e-9:
        centers = np.array([float(taus.mean())])
        labels = np.zeros(K, int)
    else:
        rng = np.random.default_rng(seed)
        centers = np.sort(rng.uniform(lo, hi, size=int(n_clusters)))
        labels = np.zeros(K, int)
        for _ in range(int(iters)):
            d = np.abs(taus[:, None] - centers[None, :])
            labels = np.argmin(d, axis=1)
            new = centers.copy()
            for c in range(centers.size):
                sel = labels == c
                if np.any(sel):
                    new[c] = taus[sel].mean()
            if np.allclose(new, centers, atol=1e-12):
                centers = new
                break
            centers = new
        order = np.argsort(centers)
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        centers = centers[order]
        labels = remap[labels]

    hist, edges = np.histogram(taus, bins=64)
    return {
        "centers": centers,
        "labels": labels,
        "taus": taus,
        "hist": {"counts": hist.tolist(),
                 "edges": [round(float(e), 9) for e in edges]},
    }


__all__ = [
    "gcc_phat", "gcc_phat_matrix", "tdoa_from_delays", "delay_residual",
    "cluster_delays",
]
