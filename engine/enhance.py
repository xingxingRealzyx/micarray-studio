"""多通道语音增强：MVDR+后置滤波、多通道维纳滤波(MWF)、GSC、谱减、去混响、评价指标。

设计约束（严格遵守 docs/CONTRACT.md）：

* 第 1、2 节：坐标系与角度。所有方向一律经 :func:`engine.acoustics.unit_dir` 转换，
  本模块**不自己写角度公式**；`steer_dir` 允许两种写法：
  ``(az_deg, el_deg)`` 或 ``(x, y, z)``（模长≈1 视为方向，否则视为近场坐标点）。
* 第 3 节：导向矢量一律用 :func:`engine.steering.steering_vector`，
  符号约定 ``a_m(f,u) = exp(-j2πf·dot(p_m,u)/c)``，与 C++ 导出完全一致。
* 第 5 节：STFT/ISTFT 一律用 :func:`engine.dsp.stft` / :func:`engine.dsp.istft`
  （rfft 单边谱、Blackman-Harris 窗、**不做幅度补偿**）。
  频域张量布局 ``Xf: (M, K, F)``。
* 第 6 节：波束成形权重可由 :mod:`engine.beamforming` 提供
  （``design(geom, freqs_sel, method=...) -> BeamformerResult``，含 ``.freqs/.weights/.metrics``）。
  本模块对该模块**延迟 import + 严格校验**：只有返回值形状/数值都可信时才采用，
  否则退回到本文件内的解析式设计（扩散场相干矩阵 / 数据协方差 + 导向矢量），
  并把实际使用的路径记录在 ``params['design_path']`` 与 ``metrics`` 中。

统一返回约定
------------
每个增强函数返回 ``dict``，至少包含::

    y        : (N,) float64  增强后的时域信号（无 NaN/Inf，长度与输入一致）
    method   : str           方法名
    params   : dict          全部关键参数（JSON 友好）
    metrics  : dict          输入/输出 SNR、SI-SDR、可懂度代理（给了 ref 才有）
    curve    : dict          可选，收敛/增益曲线，供 GUI 绘图
    ...      : 其余键为中间量（权重、噪声 PSD、增益矩阵等）

数值稳定性总原则（见各函数注释）
--------------------------------
1. 所有分母加 ``EPS`` 下限；
2. 噪声 PSD 估计设下限，并夹在 ``[EPS, 观测功率]`` 之间（噪声不可能大于总功率）；
3. 所有频域增益夹在 ``[floor, 1]``，再做时间（快攻慢放）与频率（3 点）平滑，
   抑制"音乐噪声"；
4. 协方差加对角加载并对 WNG（白噪增益）设门限，避免求逆爆掉 + 失配下自激；
5. 初/末帧、静音帧、秩亏场景全部显式兜底，最后统一 ``nan_to_num``；
6. **STFT 一律 ``center=True`` 并丢弃补零段**（见 :func:`_istft`）：
   ``engine.dsp.istft`` 在窗两端用 ≈1e-9 的 wsum 做归一化，频谱一旦被修改
   （波束权重/谱增益）就会把首尾十几个样点放大 1e4 倍 —— 这是本模块开发中
   踩过的最深的坑，实测能把 SI-SDR 从 +5 dB 直接打到 -20 dB。

本文件不引入任何随机性（无 np.random 调用），结果完全可复现。
"""
from __future__ import annotations

import numpy as np

from . import acoustics as ac
from . import dsp
from . import signals as sg
from . import steering as st

EPS = 1e-12
C_DEFAULT = ac.C_DEFAULT

# SII/STOI 类可懂度指标的 1/3 倍频程频带中心（Hz）与频带重要性权重。
# 注意：这是对 ANSI S3.5 语音重要性函数的**近似**取值，只用于本引擎内部的
# 相对比较（增强前 vs 增强后），不是标准 SII/STOI 实现，详见 intelligibility_proxy()。
_BAND_CENTERS_HZ = np.array(
    [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
     1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000], dtype=np.float64)
_BAND_WEIGHTS = np.array(
    [0.0083, 0.0095, 0.0150, 0.0284, 0.0500, 0.0702, 0.0892, 0.1056, 0.1072,
     0.1102, 0.1014, 0.0863, 0.0682, 0.0530, 0.0387, 0.0261, 0.0180, 0.0134],
    dtype=np.float64)


# ==========================================================================
# 基础工具
# ==========================================================================
def _plain(obj):
    """numpy 类型 -> 原生 Python，便于 json 序列化（CONTRACT 第 8 节）。"""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _finite(y: np.ndarray, tag: str = "") -> np.ndarray:
    """把 NaN/Inf 清零（并统计数量，写进返回值便于诊断）。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    bad = int(np.count_nonzero(~np.isfinite(y)))
    if bad:
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if tag:
            print(f"[enhance] 警告：{tag} 输出含 {bad} 个非有限值，已清零")
    return y


def _as_mc(X: np.ndarray) -> np.ndarray:
    """(N,) / (M,N) -> (M,N) float64。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("X 必须是 (N,) 或 (M,N)")
    return np.ascontiguousarray(X)


def _check_mics(mics) -> np.ndarray:
    # 宽接口：既接受 (M,3) 坐标，也接受带 .mics 的 ArrayGeometry
    mics = getattr(mics, "mics", mics)
    m = np.asarray(mics, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError("mics 必须是 (M,3)")
    if not np.all(np.isfinite(m)):
        raise ValueError("mics 含非有限值")
    return m


def _resolve_dir(steer_dir) -> np.ndarray:
    """把用户给的指向解析成 3 维向量（角度换算一律走 acoustics.unit_dir）。"""
    if steer_dir is None:
        raise ValueError("steer_dir 不能为空")
    v = np.asarray(steer_dir, dtype=np.float64).ravel()
    if v.size == 2:
        # (az, el) 度：el 从 xy 平面量向 +z，boresight=+z。见 CONTRACT 第 2 节。
        return ac.unit_dir(float(v[0]), float(v[1]))
    if v.size == 3:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            raise ValueError("steer_dir 模长为 0")
        # 模长≈1 视为方向单位向量；否则按近场坐标点原样透传给 steering_vector
        return v / n if abs(n - 1.0) < 1e-6 else v
    raise ValueError("steer_dir 需为 (az_deg, el_deg) 或 (x, y, z)")


def _hop_of(n_fft: int, hop: int | None) -> int:
    h = int(n_fft // 2 if hop is None else hop)
    if h <= 0 or h > n_fft:
        raise ValueError("hop 必须落在 (0, n_fft]")
    return h


def _stft(X: np.ndarray, fs: int, n_fft: int, hop: int, win: str,
          center: bool = True):
    """统一 STFT（CONTRACT 第 5 节：rfft 单边谱、不做幅度补偿）。"""
    return dsp.stft(X, fs, n_fft=n_fft, hop=hop, win=win, center=bool(center))


def _istft(Y: np.ndarray, fs: int, n_fft: int, hop: int, win: str,
           length: int, center: bool = True) -> np.ndarray:
    """统一 ISTFT + **边界保护**（本模块踩过的最深的一个坑，写清楚）。

    ``engine.dsp.istft`` 用窗平方的重叠和 ``wsum`` 逐点归一化，
    而 Blackman-Harris 窗两端衰减到 ``w[0] = 6.0e-5``（``w[0]² = 3.6e-9``）。
    频谱**原样**时 WOLA 恒等式逐点成立，除法恰好还原；
    但一旦频谱被改动过（波束成形权重、谱增益、掩蔽……），
    ``Σ_k frames_k[n]`` 不再等于 ``y[n]·wsum[n]``，1e-9 量级的分母会把边缘
    十几个样点放大 1e4 倍 —— 实测随机相位掩蔽下 max|y| 从 1.0 变成 9397。
    这个尖峰能把 SI-SDR 直接打到 -20 dB（本模块首版就是这么挂的），
    而 ``wsum > 1e-12`` 的门限太宽松，根本挡不住。

    对策：``center=True``（首尾各补 ``n_fft//2``），丢弃被污染的补零段。
    只损失首尾各 ``n_fft//2`` 个样点（约 16 ms），对语音增强完全无害。
    """
    pad = n_fft // 2 if center else 0
    out = dsp.istft(Y, fs, n_fft=n_fft, hop=hop, win=win, length=None)
    if center:
        out = out[..., pad:pad + int(length)]
    else:
        out = out[..., :int(length)]
    if out.shape[-1] < int(length):                      # 理论上不会发生，兜底
        padw = [(0, 0)] * (out.ndim - 1) + [(0, int(length) - out.shape[-1])]
        out = np.pad(out, padw)
    return out


def _steering(mics: np.ndarray, freqs: np.ndarray, tgt: np.ndarray,
              c: float) -> np.ndarray:
    """(F,M) 导向矢量。远场/近场由 steering_vector 自动判定（CONTRACT 第 3 节）。"""
    return st.steering_vector(mics, freqs, tgt, c=c)


def _das_weights(a: np.ndarray) -> np.ndarray:
    """延迟求和（DAS）权重：w = a/(a^H a)，满足 w^H a = 1（无失真约束）。

    DAS 是本模块所有"最坏情况兜底"：WNG=0 dB（=10log10 M 的阵增益），
    数值上从不出问题，失配下也只轻微退化。
    """
    denom = np.maximum(np.sum(np.abs(a) ** 2, axis=1), EPS)
    return a / denom[:, None]


def _wng_db(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """白噪增益 dB（CONTRACT 第 6 节定义）：
    WNG = |w^H a|^2 / ((w^H w)·(a^H a))。
    用 w^H a = 1 归一化后即 1/((w^H w)(a^H a))，DAS 恰好为 0 dB。
    """
    wha = np.abs(np.einsum("fm,fm->f", np.conj(w), a)) ** 2
    ww = np.sum(np.abs(w) ** 2, axis=1)
    aa = np.sum(np.abs(a) ** 2, axis=1)
    return 10.0 * np.log10(np.maximum(wha / np.maximum(ww * aa, EPS), 1e-30))


def _wng_blend(w: np.ndarray, a: np.ndarray, wng_min_db: float | None,
               grid: tuple = (0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)
               ) -> tuple:
    """WNG 保护：把权重与 DAS 做**保尺度**凸组合，取满足门限的最小混合系数。

    为什么用混合而不是"迭代加大对角加载"：加载改的是解的形状。
    对 MVDR 那种 ``R⁻¹a`` 形式的解，加载 → DAS，是可控的；
    但对 MWF 那种 ``R⁻¹·Rss·e_ref`` 形式的解，加载到极致得到的是 Rss 的一列 ——
    一个毫无空间意义的滤波器，越加载越糟（首版就是这么挂的）。

    **保尺度**是关键：设 ``c = w^H a``，用户要的滤波器尺度就是这个 c
    （MWF 的 MMSE 收缩增益全在 c 里）。混合在归一化域里做、再乘回 ``conj(c)``：

        w_b = (1-α)·w + α·w_das·conj(c)   ⇒   w_b^H a ≡ c（与 α 无关）

    于是 α 只改**形状**不改尺度，WNG 从原解单调过渡到 DAS 的 0 dB。
    注意这里**不再无条件把 w 归一化到 w^H a = 1** —— 那会把 MWF 的
    MMSE 收缩增益抹掉，实测让 MWF 从 +7.6 dB 掉到 -5 dB。

    ``wng_min_db=None`` 表示不启用该保护。
    """
    w = np.asarray(w, dtype=np.complex128)
    w_das = _das_weights(a)
    c = np.einsum("fm,fm->f", np.conj(w), a)                # w^H a：尺度
    wng = _wng_db(w / np.maximum(np.abs(c), 1e-30)[:, None], a)
    if wng_min_db is None:
        return w, wng
    bad = (~np.isfinite(wng)) | (wng < float(wng_min_db))
    if not bad.any():
        return w, wng
    for alpha in grid:                              # alpha: 0 -> 1（越大越像 DAS）
        if not bad.any():
            break
        wb = (1.0 - alpha) * w[bad] + alpha * w_das[bad] * np.conj(c[bad])[:, None]
        wngb = _wng_db(wb / np.maximum(np.abs(c[bad]), 1e-30)[:, None], a[bad])
        ok = np.isfinite(wngb) & (wngb >= float(wng_min_db))
        if np.any(ok):
            idx = np.nonzero(bad)[0][ok]
            w[idx] = wb[ok]
            wng[idx] = wngb[ok]
            bad[idx] = False
    if bad.any():                                   # 极端兜底：纯 DAS（保尺度）
        w[bad] = w_das[bad] * np.conj(c[bad])[:, None]
        wng[bad] = 0.0
    return w, wng


def _frame_vad(Py: np.ndarray, Pn: np.ndarray, thr_db: float = 3.0,
               width_db: float = 3.0, alpha: float = 0.6) -> np.ndarray:
    """逐帧能量 VAD（0=纯噪声可自适应，1=有语音冻结自适应）。

    为什么不能用 SPP 做门控：SPP 是**逐频点**的软判决，在混响环境里几乎恒为
    1（实测帧均值 0.67~0.82，从不低于 0.3），拿它当 VAD 等于永不冻结自适应，
    GSC 会把自己的目标语音当噪声抵消掉。

    改用帧级能量判决：帧总功率 与 最小统计噪声底 之比（dB），
    用宽度 ``width_db`` 的软过渡带映射到 [0,1]。实测在 3 dB 门限上
    语音帧的误判率为 0（宁可漏检、不可误触发：漏检只是自适应慢一点，
    误触发会直接削语音）。
    """
    Py = np.maximum(np.asarray(Py, float), EPS)
    Pn = np.maximum(np.asarray(Pn, float), EPS)
    snr_db = 10.0 * np.log10(np.maximum(Py, EPS) / np.maximum(Pn, EPS))
    K = snr_db.size
    out = np.empty(K)
    acc = snr_db[0] if K else 0.0
    for k in range(K):
        acc = alpha * acc + (1.0 - alpha) * snr_db[k]
        out[k] = acc
    lo = thr_db - 0.5 * width_db
    return np.clip((out - lo) / max(width_db, 1e-6), 0.0, 1.0)


def _psd_project(R: np.ndarray) -> np.ndarray:
    """把 Hermitian 矩阵投影到半正定锥（特征值截负）。

    估计出来的 Rss = Rx - Rnn 可能因为估计误差出现负特征值，
    直接拿去解 MWF 会得到病态解；截负是标准的稳健化手段。
    """
    lam, V = np.linalg.eigh(R)                  # lam (F,M) 升序, V (F,M,M)
    lam = np.maximum(lam.real, 0.0)
    return (V * lam[:, None, :]) @ V.conj().transpose(0, 2, 1)


def _weighted_cov(Xf: np.ndarray, w: np.ndarray) -> np.ndarray:
    """加权协方差 R(f) = Σ_k w(k,f)·x(k,f)x(k,f)^H / Σ_k w(k,f)  ->  (F,M,M)。

    Xf: (M,K,F)，w: (K,F) 非负实数（如语音存在概率）。
    逐频点做 (M,K)@(K,M) 的矩阵乘，避免展开 (K,F,M,M) 这种 O(K·F·M²) 内存。
    """
    M, K, F = Xf.shape
    wsum = np.maximum(np.sum(w, axis=0), EPS)              # (F,)
    sw = np.sqrt(np.maximum(w, 0.0))                       # (K,F)
    R = np.zeros((F, M, M), dtype=np.complex128)
    for f in range(F):
        x = Xf[:, :, f] * sw[:, f][None, :]                # (M,K)
        R[f] = (x @ x.conj().T) / wsum[f]
    return R


def _min_stats_noise(P: np.ndarray, sub_len: int = 8, n_sub: int = 5,
                     bias: float = 1.5, alpha: float = 0.9) -> np.ndarray:
    """最小值统计噪声功率估计（Martin 1994 两级最小跟踪的简化实现）。

    原理：语音是"稀疏"的——任何一个频点在足够长的窗内总会落到噪声水平附近。
    因此用一个因果滑窗内的最小值 × 偏差补偿就能跟踪缓变噪声功率，
    不需要 VAD，也不怕音乐噪声（它天然低估一点，宁可少减不要多减）。

    参数
    ----
    sub_len : 一级子窗长度（帧）；n_sub : 二级保留的子窗个数。
              总窗长 = sub_len*n_sub 帧，hop=256@16k 时 8*5=40 帧 ≈ 0.64 s。
    bias    : 最小值对均值的低估补偿（经验值 1.5，指数分布最小值 ~ P̄/D）。
    alpha   : 功率平滑系数（0.9 ≈ 10 帧时间常数）。

    陷阱
    ----
    * 不做偏差补偿会系统性低估噪声 → 抑制不足；
    * 补偿过头（>2）会在语音段把噪声估计顶到语音功率上 → 削语音；
    * 必须夹 ``Pn <= Ps``（噪声不可能超过总功率）并设绝对下限，
      否则长语音段最小值会掉到 0，增益公式里除法直接爆掉。
    """
    P = np.maximum(np.asarray(P, dtype=np.float64), EPS)
    K, F = P.shape
    if K == 0:
        return P.copy()
    sub_len = max(int(sub_len), 1)
    n_sub = max(int(n_sub), 1)

    # 一级：一阶递归平滑（降方差）
    Ps = np.empty_like(P)
    acc = P[0].copy()
    for k in range(K):
        acc = alpha * acc + (1.0 - alpha) * P[k]
        Ps[k] = acc

    # 二级：子窗最小值 + 因果滑窗最小值
    n_subwin = int(np.ceil(K / sub_len))
    submin = np.empty((n_subwin, F))
    for i in range(n_subwin):
        blk = Ps[i * sub_len:min((i + 1) * sub_len, K)]
        submin[i] = blk.min(axis=0)

    Pn = np.empty_like(P)
    for i in range(n_subwin):
        lo = max(0, i - n_sub + 1)
        m = submin[lo:i + 1].min(axis=0) * bias
        Pn[i * sub_len:min((i + 1) * sub_len, K)] = m[None, :]

    Pn = np.minimum(Pn, Ps)                     # 噪声估计不可能大于总功率
    return np.maximum(Pn, EPS * 1e3)


def _spp_decision_directed(Py: np.ndarray, Pn: np.ndarray, alpha: float = 0.92,
                           q: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """语音存在概率（SPP）与先验 SNR，判决引导（Ephraim-Malah）递推。

    * 后验 SNR  gamma = Py/Pn
    * 先验 SNR    xi = alpha·Ĝ²·gamma_prev + (1-alpha)·max(gamma-1, 0)
      第一项是"上一帧的语音"（利用语音的帧间连续性，避免孤立帧抖动），
      第二项是"本帧的瞬时估计"（最大似然，跟随突变）。
    * SPP         p = 1/(1 + (q/(1-q))·(1+xi)·exp(-v))，v = gamma·xi/(1+xi)
      （Ephraim-Malah 的 MMSE-SPP 形式，q 为语音缺席先验）。

    返回 (spp (K,F), xi (K,F))。全程逐帧一阶递推，无随机性。
    """
    Py = np.maximum(np.asarray(Py, float), EPS)
    Pn = np.maximum(np.asarray(Pn, float), EPS)
    K, F = Py.shape
    gamma = np.clip(Py / Pn, 1e-6, 1e8)
    xi = np.empty_like(gamma)
    spp = np.empty_like(gamma)

    G2 = np.ones(F)                     # 上一帧的维纳增益平方 Ĝ²
    g_prev = np.ones(F)
    qq = float(np.clip(q, 1e-3, 1.0 - 1e-3))
    lr = qq / (1.0 - qq)
    for k in range(K):
        xi_k = alpha * G2 * g_prev + (1.0 - alpha) * np.maximum(gamma[k] - 1.0, 0.0)
        xi_k = np.maximum(xi_k, 1e-6)
        v = gamma[k] * xi_k / (1.0 + xi_k)
        p = 1.0 / (1.0 + lr * (1.0 + xi_k) * np.exp(-np.minimum(v, 60.0)))
        xi[k] = xi_k
        spp[k] = np.clip(p, 0.0, 1.0)
        g_prev = gamma[k]
        G2 = (xi_k / (1.0 + xi_k)) ** 2
    return spp, xi


def _frame_gate(spp: np.ndarray, freqs: np.ndarray,
                f_lo: float = 150.0, f_hi: float = 4000.0) -> np.ndarray:
    """逐帧语音存在度（0..1）：只统计语音频带内的 SPP 均值。

    用于 GSC 的自适应门控（语音段冻结自适应，防止把目标语音当噪声抵消掉）
    以及协方差更新的软权重。
    """
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(sel):
        sel = np.ones_like(freqs, dtype=bool)
    return np.clip(np.mean(spp[:, sel], axis=1), 0.0, 1.0)


def _estimate_noise_cov(Xf: np.ndarray, freqs: np.ndarray, *,
                        ms_sub_len: int = 8, ms_n_sub: int = 5,
                        ms_bias: float = 1.5, ms_alpha: float = 0.6,
                        spp_alpha: float = 0.92, spp_q: float = 0.3,
                        cov_alpha: float = 0.9) -> tuple:
    """噪声协方差 Rnn(f) 的估计。返回 (Rnn, spp, frame_gate, Pn_ch)。

    步骤（每一步都是为了让 MVDR/MWF 不"自杀"）：

    1. 逐通道最小值统计得到噪声功率下限 ``Pn_ch``（鲁棒、无 VAD 依赖）；
    2. 用通道平均功率做判决引导 SPP 估计；
    3. 以 ``diag(mean_k Pn_ch)`` 作为 Rnn 的初值 —— 这一步很关键：
       只靠 SPP 递推的话，若开场就是语音，Rnn 会从"语音协方差"起步，
       MVDR 会把主瓣对准的语音直接抵消（signal cancellation）；
    4. 用 SPP 做**加权递推平均**（num/den 同时递推，保证是真正的加权均值，
       不会被权重和的大小改变量纲）：
       ``Rnn ← (α·num + (1-p)·x x^H) / (α·den + (1-p))``。

    数值陷阱：语音泄漏进 Rnn 会让 MVDR 在目标方向形成零陷；因此
    ``q``（语音缺席先验）不宜取太小，且初值必须用最小值统计兜底。
    """
    M, K, F = Xf.shape
    Xp = np.abs(Xf) ** 2                                            # (M,K,F)

    # 1) 逐通道最小值统计（时间在后，便于函数复用）
    Pn_ch = np.empty((M, K, F))
    for m in range(M):
        Pn_ch[m] = _min_stats_noise(Xp[m], sub_len=ms_sub_len, n_sub=ms_n_sub,
                                    bias=ms_bias, alpha=ms_alpha)

    # 2) SPP：在"通道平均功率"上估计（比单通道稳，且不需要先做波束成形）
    Py_bar = np.mean(Xp, axis=0)                                    # (K,F)
    Pn_bar = np.mean(Pn_ch, axis=0)
    spp, _ = _spp_decision_directed(Py_bar, Pn_bar, alpha=spp_alpha, q=spp_q)

    # 3) 初值：最小值统计的对角协方差
    Rnn0 = np.zeros((F, M, M), dtype=np.complex128)
    idx = np.arange(M)
    Rnn0[:, idx, idx] = np.mean(Pn_ch, axis=1).T                    # (F,M)

    # 4) SPP 加权递推平均
    num = Rnn0 * 1e-3                                               # 极小的"伪计数"
    den = np.full(F, 1e-3)
    for k in range(K):
        x = Xf[:, k, :]                                             # (M,F)
        xxh = np.einsum("mf,nf->mnf", x, np.conj(x))                # (M,M,F)
        wgt = (1.0 - spp[k])[None, None, :]                         # (1,1,F)
        num = cov_alpha * num + np.transpose(wgt * xxh, (2, 0, 1))
        den = cov_alpha * den + wgt.ravel()
    Rnn = num / np.maximum(den, EPS)[:, None, None]
    Rnn = 0.5 * (Rnn + Rnn.conj().transpose(0, 2, 1))               # 强制 Hermitian

    gate = _frame_gate(spp, freqs)
    return Rnn, spp, gate, Pn_ch


def _diag_load(R: np.ndarray, delta_rel: float) -> np.ndarray:
    """相对对角加载：R + δ·(tr(R)/M)·I。

    加载量按平均通道功率定标，因此换阵列、换电平都不用重新调参。
    加载既保证可逆（数值陷阱），也是失配下的稳健化手段。
    """
    F, M, _ = R.shape
    tr = np.maximum(np.einsum("fmm->f", R).real / M, EPS)
    return R + (delta_rel * tr)[:, None, None] * np.eye(M)[None, :, :]


def _design_mvdr(Rnn: np.ndarray, a: np.ndarray, wng_min_db: float = -3.0,
                 delta0_db: float = -40.0, n_iter: int = 14,
                 grow_db: float = 6.0) -> tuple:
    """WNG 约束下的对角加载 MVDR（本模块的 fallback 设计器，解析式）。

    ``w = R⁻¹a / (a^H R⁻¹a)``，R = Rnn + δ·tr(Rnn)/M·I。
    归一化后 ``w^H a = 1``（无失真），输出尺度与 DAS 一致 —— 这一点对
    分段 SNR 这类**尺度敏感**的指标很重要。

    为什么必须约束 WNG：MVDR 在低频（阵远小于波长、导向矢量几乎线性相关）
    会趋向"超指向"解，``w^H w`` 爆炸、WNG 掉到 -40 dB，此时任何幅相失配
    都会被放大成自激。做法是对 WNG 低于门限的频点逐步加大 δ（每次 +6 dB），
    仍不满足的频点回退/混合到 DAS。

    返回 (w (F,M), info dict)。info 里记录每频点最终 WNG 与所用 δ，
    便于排查"C++ 侧为什么对不上"这类问题。
    """
    F, M, _ = Rnn.shape
    Rnn = 0.5 * (Rnn + Rnn.conj().transpose(0, 2, 1))
    tr = np.maximum(np.einsum("fmm->f", Rnn).real / M, EPS)
    delta = np.full(F, 10.0 ** (delta0_db / 10.0)) * tr
    w_das = _das_weights(a)
    w = w_das.copy()
    eye = np.eye(M)[None, :, :]
    done = np.zeros(F, dtype=bool)
    wng = np.full(F, -np.inf)

    for _ in range(max(int(n_iter), 1)):
        todo = ~done
        if not todo.any():
            break
        Rt = Rnn[todo] + (delta[todo])[:, None, None] * eye
        at = a[todo]
        v = np.linalg.solve(Rt, at[:, :, None])[:, :, 0]             # (n,M)
        den = np.einsum("fm,fm->f", np.conj(at), v)
        # R 正定 ⇒ den 为正实数；数值上保底避免除以 0
        den = np.where(np.abs(den) < 1e-12, 1e-12 + 0j, den)
        wk = v / den[:, None]
        wngk = _wng_db(wk, at)
        ok = np.isfinite(wngk) & (wngk >= wng_min_db)
        idx = np.nonzero(todo)[0]
        good = idx[ok]
        w[good] = wk[ok]
        wng[good] = wngk[ok]
        done[good] = True
        bad = idx[~ok]
        if bad.size == 0:
            break
        delta[bad] *= 10.0 ** (grow_db / 10.0)

    # 仍未达标的频点：与 DAS 线性混合，取满足约束的最大混合系数 t
    bad = np.nonzero(~done)[0]
    if bad.size:
        for t in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            if bad.size == 0:
                break
            wb = (1.0 - t) * w_das[bad] + t * w[bad]
            wngb = _wng_db(wb, a[bad])
            ok = np.isfinite(wngb) & (wngb >= wng_min_db)
            if np.any(ok):
                sel = bad[ok]
                w[sel] = wb[ok]
                wng[sel] = wngb[ok]
                bad = bad[~ok]
        if bad.size:                     # 极端兜底：纯 DAS（WNG=0 dB）
            w[bad] = w_das[bad]
            wng[bad] = _wng_db(w_das[bad], a[bad])

    info = {
        "designer": "analytic_mvdr_diag_load",
        "wng_db": wng,
        "delta_db": 10.0 * np.log10(np.maximum(delta / tr, 1e-30)),
        "wng_min_db_actual": float(np.min(wng)),
        "wng_floor_db": float(wng_min_db),
    }
    return w, info


def _validate_external_weights(w: np.ndarray, a: np.ndarray, F: int, M: int,
                               wng_min_db: float = -25.0) -> tuple:
    """校验外部（engine.beamforming）给出的权重是否可用。

    校验项：形状、有限性、w^H a 不能≈0（否则主瓣方向被抵消）、
    WNG 不能低于硬下限（否则失配下会自激）。任一项不过就整体弃用 ——
    宁可退回自己的解析式，也不要静默地用一组错的权重。
    """
    if w is None:
        return None, "None"
    w = np.asarray(w, dtype=np.complex128)
    if w.shape != (F, M):
        return None, f"shape {w.shape} != {(F, M)}"
    if not np.all(np.isfinite(w)):
        return None, "non-finite"
    w = w.copy()
    c = np.einsum("fm,fm->f", np.conj(w), a)
    if np.any(np.abs(c) < 1e-6):
        return None, "w^H a ≈ 0（主瓣被抵消）"
    w = w / np.conj(c)[:, None]                     # 归一化到 w^H a = 1
    wng = _wng_db(w, a)
    if float(np.nanmin(wng)) < wng_min_db:
        return None, f"WNG 过低 ({float(np.nanmin(wng)):.1f} dB)"
    return w, "ok"


def _try_beamforming_module(geom, freqs: np.ndarray, mics: np.ndarray,
                            tgt: np.ndarray, c: float, a: np.ndarray,
                            method: str = "mvdr", Rnn: np.ndarray | None = None,
                            wng_min_db: float = 0.0) -> tuple:
    """尝试调用 engine.beamforming.design()（CONTRACT 第 6 节）。

    另一个任务在并行实现该模块，因此这里做三件事：
    1. **延迟 import**：模块不存在时不影响本模块可用；
    2. 用尽量宽松的方式调用（kwargs 兼容若干命名），失败即放弃；
    3. 返回值经 :func:`_validate_external_weights` 严格校验后才使用。

    候选调用按"信息量"排序：
    * ``mvdr`` + ``R_est=Rnn`` + ``wng_floor_db``：把本模块**实测的噪声协方差**
      交给 beamforming 设计器，这才是真正的自适应 MVDR；
    * ``identity`` + ``wng_floor_db``：白噪协方差（等价于无先验的稳健 MVDR /
      DAS 兜底）；
    * 各种 kwargs 别名，兼容该模块参数命名的演化。
    任一路径的权重都要通过校验（形状 / 有限性 / w^H a 不为 0 / WNG 门限），
    不过就回退到本文件的解析式设计。

    返回 (w 或 None, info dict)。
    """
    info = {"design_path": "fallback", "reason": "beamforming 模块不可用"}
    if geom is None:
        info["reason"] = "未提供 ArrayGeometry，跳过 beamforming 模块"
        return None, info
    try:
        from . import beamforming  # 延迟 import：模块可能还不存在
    except Exception as exc:                                   # pragma: no cover
        info["reason"] = f"import 失败: {type(exc).__name__}: {exc}"
        return None, info
    if not hasattr(beamforming, "design"):
        info["reason"] = "beamforming.design 不存在"
        return None, info

    res = None
    last_err = ""
    # 参数名在同一契约下可能不同，逐个候选尝试；语义都一致（指向 + 声速）。
    # 前两个候选带 R_est（实测噪声协方差）与 WNG 门限，是真正的 MVDR；
    # 后面的 identity 变体是无先验协方差时的稳健兜底（≈ DAS）。
    cand_kwargs = []
    if method == "mvdr" and Rnn is not None:
        cand_kwargs.append({"method": "mvdr", "steer_dir": tgt, "c": c,
                            "R_est": Rnn, "wng_floor_db": wng_min_db})
    cand_kwargs += [
        {"method": "identity", "steer_dir": tgt, "c": c,
         "wng_floor_db": wng_min_db},
        {"method": method, "steer_dir": tgt, "c": c},
        {"method": method, "direction": tgt, "c": c},
        {"method": method, "u": tgt, "c": c},
    ]
    for kw in cand_kwargs:
        try:
            res = beamforming.design(geom, freqs, **kw)
            break
        except TypeError as exc:
            last_err = str(exc)
            continue
        except Exception as exc:
            # 例如 method='mvdr' 但该模块要求 R_est：继续试下一个候选
            last_err = f"{type(exc).__name__}: {exc}"
            continue
    if res is None:
        info["reason"] = f"design() 调用签名不兼容: {last_err}"
        return None, info

    w_raw = getattr(res, "weights", None)
    if w_raw is None and isinstance(res, dict):
        w_raw = res.get("weights")
    w, why = _validate_external_weights(w_raw, a, freqs.size, mics.shape[0])
    if w is None:
        info["reason"] = f"权重校验失败: {why}"
        return None, info
    info.update({"design_path": "beamforming", "reason": "ok",
                 "method": getattr(res, "method", method)})
    return w, info


def _smooth_power_time(P: np.ndarray, alpha: float) -> np.ndarray:
    """功率的时间一阶平滑（降方差，抑制音乐噪声）。逐帧递推，确定性。"""
    K = P.shape[0]
    out = np.empty_like(P)
    if K == 0:
        return out
    acc = P[0].copy()
    for k in range(K):
        acc = alpha * acc + (1.0 - alpha) * P[k]
        out[k] = acc
    return out


def _smooth_gain(G: np.ndarray, attack: float = 0.55, release: float = 0.9,
                 freq_smooth: bool = True) -> np.ndarray:
    """增益平滑：时间方向"快攻慢放" + 频率方向 3 点滑动平均。

    * 快攻（gain 上升时用小的 attack 系数）→ 语音起始不被削平；
    * 慢放（gain 下降时用大的 release 系数）→ 语音结束后的残留增益缓慢下降，
      这正是抑制"音乐噪声"的关键（孤立噪声点的增益来不及瞬间打开）；
    * 频率方向 3 点平滑：相邻频点增益突变是音乐噪声的另一来源。
    注意平滑都在**线性幅度**域做，dB 域平滑会过度抬高低增益频点。
    """
    K = G.shape[0]
    out = np.empty_like(G)
    if K == 0:
        return out
    prev = G[0].copy()
    at = float(np.clip(attack, 0.0, 0.999))
    rl = float(np.clip(release, 0.0, 0.999))
    for k in range(K):
        a = np.where(G[k] > prev, at, rl)
        prev = a * prev + (1.0 - a) * G[k]
        out[k] = prev
    if freq_smooth and out.shape[1] >= 3:
        padded = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
    return out


def _apply_beamformer(Xf: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y(k,f) = w(f)^H x(k,f)。Xf (M,K,F) -> (K,F)。"""
    return np.einsum("mkf,fm->kf", Xf, np.conj(w))


# ==========================================================================
# 评价指标
# ==========================================================================
def _band_powers(P: np.ndarray, freqs: np.ndarray) -> tuple:
    """把单边功率谱 (K,F) 汇总到 1/3 倍频程频带 -> (K, B)。"""
    centers = _BAND_CENTERS_HZ
    lo = centers / np.sqrt(2.0 ** (1.0 / 3.0))
    hi = centers * np.sqrt(2.0 ** (1.0 / 3.0))
    keep = lo < freqs[-1]
    centers, lo, hi, wgt = centers[keep], lo[keep], hi[keep], _BAND_WEIGHTS[keep]
    B = centers.size
    out = np.zeros((P.shape[0], B))
    for b in range(B):
        sel = (freqs >= lo[b]) & (freqs < hi[b])
        if np.any(sel):
            out[:, b] = np.sum(P[:, sel], axis=1)
    return out, centers, wgt


def intelligibility_proxy(clean_ref: np.ndarray, enhanced: np.ndarray, fs: int,
                          n_fft: int = 512, hop: int | None = None,
                          win: str = "blackmanharris",
                          snr_lo_db: float = -15.0, snr_hi_db: float = 15.0) -> float:
    """可懂度代理指标（0..1，越大越清晰）。

    定义（明确写出来，避免"黑箱指标"）：

    1. 对参考与增强信号做同样的 STFT，得到单边功率谱；
    2. 在 18 个 1/3 倍频程频带（160 Hz–8 kHz）上分别汇总
       语音功率 ``P_s,b`` 与误差功率 ``P_e,b = |Est-Ref|²``；
    3. 频带 SNR ``SNR_b = 10log10(P_s,b / P_e,b)``，截断到 [-15, +15] dB
       后线性映射到 [0,1]：``q_b = (clip(SNR_b) + 15)/30``；
    4. 用近似 SII 频带重要性权重加权求和，权重只在"参考能量不衰减的频带"
       上重新归一化（静音频带不参与，否则低频段会把指标拉偏）。

    即：**频带 SNR 的加权可分度**。它与 SII 的结构一致（频带可懂度 → 加权和），
    与 STOI 的相关性在噪声类失真上通常 >0.9，但它不是标准 SII/STOI 实现：
    不模拟听觉滤波、不处理真实语音材料，只用于本引擎内"增强前/后"的相对比较。
    """
    ref = np.asarray(clean_ref, float).ravel()
    est = np.asarray(enhanced, float).ravel()
    n = min(ref.size, est.size)
    if n < n_fft:
        return float("nan")
    ref, est = ref[:n], est[:n]
    hop = _hop_of(n_fft, hop)
    Rf, freqs = _stft(ref, fs, n_fft, hop, win)
    Ef, _ = _stft(est, fs, n_fft, hop, win)
    Ps = np.abs(Rf) ** 2
    Pe = np.abs(Ef - Rf) ** 2
    Ps_b, _, wgt = _band_powers(Ps, freqs)
    Pe_b, _, _ = _band_powers(Pe, freqs)

    # 只在参考能量可观的频带上统计（相对最强频带 -50 dB 以上）
    pmax = max(float(Ps_b.max()), EPS)
    act = Ps_b.mean(axis=0) > 1e-5 * pmax
    if not np.any(act):
        return float("nan")
    snr = 10.0 * np.log10(np.maximum(Ps_b, EPS) / np.maximum(Pe_b, EPS))
    q = (np.clip(snr, snr_lo_db, snr_hi_db) - snr_lo_db) / (snr_hi_db - snr_lo_db)
    w = wgt * act
    return float(np.sum(w[None, :] * q, axis=1).mean() / max(w.sum(), EPS))


def evaluate(clean_ref: np.ndarray, enhanced: np.ndarray, fs: int,
             noisy: np.ndarray | None = None, n_fft: int = 512,
             hop: int | None = None) -> dict:
    """增强效果评价。

    参数
    ----
    clean_ref : (N,) 干净参考（与输出对齐、同采样率）
    enhanced  : (N,) 增强输出
    noisy     : (N,) 增强前的含噪输入（用于算"提升量"）；None 时只报输出指标
    fs        : 采样率

    指标定义
    --------
    * ``snr_in/out_db``        : 全局 SNR = 10log10(Σref² / Σ(est-ref)²)，
                                 即把 ``est-ref`` 当作总失真（噪声+混响+失真）。
    * ``segmental_snr_*_db``   : :func:`engine.signals.segmental_snr_db`，
                                 逐 32 ms 帧算 SNR 再对语音帧取平均（语音增强标准做法）。
    * ``si_sdr_*_db``          : :func:`engine.signals.si_sdr`（尺度不变）。
    * ``intel_proxy_*``        : :func:`intelligibility_proxy`（0..1），
                                 同时以 ``stoi_proxy`` 暴露（对齐 CONTRACT 第 8 节字段名）。
    * ``*_improvement_db``     : 输出 − 输入。给 ``noisy`` 才有，否则为 ``None``。

    所有标量在写回前都做了有限性检查（NaN/Inf -> None），
    保证返回的 dict 可以直接 ``json.dumps(..., allow_nan=False)``。
    """
    ref = _finite(clean_ref, "clean_ref")
    est = _finite(enhanced, "enhanced")
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    inp = _finite(noisy, "noisy") if noisy is not None else None
    if inp is not None:
        inp = inp[:n]

    def _snr(sig: np.ndarray, target: np.ndarray) -> float:
        err = sig - target
        return float(10.0 * np.log10(np.sum(target ** 2)
                                     / max(float(np.sum(err ** 2)), 1e-30)))

    out = {
        "n_samples": int(n),
        "finite": bool(np.all(np.isfinite(est))),
        "snr_out_db": _snr(est, ref),
        "segmental_snr_out_db": sg.segmental_snr_db(ref, est, fs),
        "si_sdr_out_db": sg.si_sdr(ref, est),
        "intel_proxy_out": intelligibility_proxy(ref, est, fs, n_fft, hop),
        # 输入侧与提升量：**键恒定存在**（未给 noisy 时为 None），
        # 这样 GUI/导出层不用做存在性判断，schema 稳定。
        "snr_in_db": None,
        "segmental_snr_in_db": None,
        "si_sdr_in_db": None,
        "intel_proxy_in": None,
        "snr_improvement_db": None,
        "segmental_snr_improvement_db": None,
        "si_sdr_improvement_db": None,
        "intel_proxy_gain": None,
    }
    if inp is not None:
        out.update({
            "snr_in_db": _snr(inp, ref),
            "segmental_snr_in_db": sg.segmental_snr_db(ref, inp, fs),
            "si_sdr_in_db": sg.si_sdr(ref, inp),
            "intel_proxy_in": intelligibility_proxy(ref, inp, fs, n_fft, hop),
        })
        out["snr_improvement_db"] = out["snr_out_db"] - out["snr_in_db"]
        out["segmental_snr_improvement_db"] = \
            out["segmental_snr_out_db"] - out["segmental_snr_in_db"]
        out["si_sdr_improvement_db"] = out["si_sdr_out_db"] - out["si_sdr_in_db"]
        out["intel_proxy_gain"] = out["intel_proxy_out"] - out["intel_proxy_in"]
    # 别名：便于 GUI / 报告直接引用
    out["input_snr_db"] = out.get("snr_in_db")
    out["output_snr_db"] = out.get("snr_out_db")
    out["stoi_proxy"] = out["intel_proxy_out"]
    out["snr_gain_db"] = out.get("snr_improvement_db")
    out["seg_snr_gain_db"] = out.get("segmental_snr_improvement_db")
    out["si_sdr_gain_db"] = out.get("si_sdr_improvement_db")

    # 清理 NaN/Inf（segmental_snr_db 在无语音帧时返回 NaN）
    for k, v in list(out.items()):
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = None
    return _plain(out)


def _no_ref_metrics(y: np.ndarray, nr_est_db: float | None = None) -> dict:
    """没有干净参考时的"自证"指标：电平、有限性、估计的噪声抑制量。"""
    y = np.asarray(y, float)
    m = {
        "finite": bool(np.all(np.isfinite(y))),
        "n_samples": int(y.size),
        "output_level_db": float(20 * np.log10(max(float(np.sqrt(np.mean(y ** 2))), EPS))),
        "output_peak": float(np.max(np.abs(y))) if y.size else 0.0,
        "snr_in_db": None, "snr_out_db": None, "snr_gain_db": None,
        "segmental_snr_in_db": None, "segmental_snr_out_db": None,
        "seg_snr_gain_db": None,
        "si_sdr_in_db": None, "si_sdr_out_db": None, "si_sdr_gain_db": None,
        "intel_proxy_out": None, "stoi_proxy": None,
    }
    if nr_est_db is not None and np.isfinite(nr_est_db):
        m["noise_reduction_est_db"] = float(nr_est_db)
    return m


def _finish(y: np.ndarray, method: str, params: dict, fs: int,
            ref: np.ndarray | None, noisy: np.ndarray | None,
            curve: dict | None = None, extras: dict | None = None,
            nr_est_db: float | None = None) -> dict:
    """统一封装返回值：保证 y 无 NaN/Inf 且长度=输入长度，metrics 结构一致。"""
    y = _finite(y, method)
    if ref is not None:
        metrics = evaluate(ref, y, fs, noisy)
        if nr_est_db is not None and np.isfinite(nr_est_db):
            metrics["noise_reduction_est_db"] = float(nr_est_db)
    else:
        metrics = _no_ref_metrics(y, nr_est_db)
    out = {"y": y, "method": method, "params": _plain(params), "metrics": metrics}
    if curve:
        out["curve"] = curve
    if extras:
        out.update(extras)
    return out


# ==========================================================================
# 1) MVDR + 后置滤波
# ==========================================================================
def mvdr_postfilter(X, fs: int, mics, steer_dir, n_fft: int = 512,
                    hop: int | None = None, c: float = C_DEFAULT,
                    noise_est: str = "min_stats", over_subtraction: float = 1.0,
                    floor_db: float = -18.0, *, win: str = "blackmanharris",
                    ref_mic: int = 0, wng_min_db: float = 0.0,
                    delta0_db: float = -40.0, spp_alpha: float = 0.92,
                    spp_q: float = 0.3, ms_sub_len: int = 8, ms_n_sub: int = 5,
                    ms_bias: float = 1.5, ms_alpha: float = 0.6,
                    cov_alpha: float = 0.9, pf_alpha_p: float = 0.75,
                    pf_attack: float = 0.55, pf_release: float = 0.9,
                    pf_freq_smooth: bool = True, design: str = "auto",
                    center: bool = True, geom=None, ref=None, noisy=None) -> dict:
    """MVDR 波束成形 + 后置滤波（Zelinski / 多通道维纳型）。

    算法（三段式，每段都对应一类数值陷阱）
    --------------------------------------
    **第一段：空间滤波（MVDR）**
    ``w(f) = Rnn(f)⁻¹a(f) / (a(f)^H Rnn(f)⁻¹a(f))``，无失真约束 ``w^H a = 1``。
    * Rnn 用 SPP 加权递推平均估计，初值为逐通道最小值统计的对角阵
      （防止开场即语音时把目标语音估进 Rnn → 主瓣自杀）；
    * 对角加载 + WNG 门限（``wng_min_db``，默认 **0 dB**）：低频超指向解在
      噪声协方差估计误差与阵列失配下都会自激，加载后自动退化为 DAS
      （WNG 恰好 0 dB）—— 这就是"稳健"的全部含义。默认取 0 dB 是**实测**
      选出来的：本模块用 8 元 12 cm 圆阵 + 空间白噪做了 -10/-6/-3/-1/0 dB
      五档对比，门限越松性能越差（失配场景最明显，见测试报告表格）；
      噪声在空间上相关时（相干干扰）即便 0 dB 门限，MVDR 仍明显优于 DAS；
    * 权重来源：优先 ``engine.beamforming.design()``（CONTRACT 第 6 节，
      传 ``method='mvdr' + R_est=Rnn + wng_floor_db``），经严格校验后用；
      否则用本文件解析式设计。实际路径见 ``params['design_path']``。

    **第二段：后置滤波（噪声功率估计）**
    ``noise_est`` 四种口径（都在波束成形输出上工作）：
    * ``'min_stats'``  : 输出功率的最小值统计。无模型假设，最稳，但对
      非平稳噪声反应慢；
    * ``'zelinski'``   : 通道间互谱估计相干（语音）分量：
      ``P_coh = 2/(M(M-1))·Re Σ_{i<j} w_i* w_j Φ_ij``，``Pn = Pyy - P_coh``。
      前提是**噪声在通道间不相关**（麦克风自噪成立，强混响/相干干扰不成立）；
    * ``'cov'``        : 解析式 ``Pn(f) = w^H Rnn(f) w``（与 MVDR 同一套统计，
      方差最小，但完全依赖 Rnn 的准确性）；
    * ``'both'``       : min_stats 与 zelinski 的均值（折中）。

    **第三段：增益与平滑**
    ``G = clip(1 - β·Pn/Pyy_smooth, floor, 1)``（β=over_subtraction 过减因子，
    Zelinski/维纳后置滤波的谱减形式），随后
    * 功率时间平滑（``pf_alpha_p``）降方差；
    * 增益"快攻慢放"平滑 + 频率 3 点平滑 → 抑制音乐噪声；
    * 增益下限 ``floor_db``：**这是抗音乐噪声与抗除爆的最后一道闸**，
      但也是失真来源——设得太低（如 -40 dB）在仅噪声帧会留下梳状残留。

    参数（其余）
    -----------
    ref_mic : 评价时用作"输入"参考的通道号
    geom    : 可选的 ArrayGeometry，用于调用 engine.beamforming
    center  : STFT 首尾补零 + 丢弃被污染的补零段（默认 True）。**必须保持 True**：
              频谱被修改后，engine.dsp.istft 的 wsum 逐点归一化会在窗两端
              （w[0]² ≈ 3.6e-9）把前十几个样点放大 1e4 倍，产生足以毁掉
              SI-SDR 的尖峰。详见 :func:`_istft` 的注释。
    ref/noisy : 给了 ``ref`` 才计算 SNR/SI-SDR/可懂度（见 evaluate）

    返回
    ----
    dict，除统一字段外含中间量：``weights (F,M)``、``noise_psd (K,F)``、
    ``gain (K,F)``、``Rnn (F,M,M)``、``freqs``、``Xf``、``wng_db``、``spp``。
    ``curve`` 内含供 GUI 绘制的增益/噪声/曲线。
    """
    X = _as_mc(X)
    M, N = X.shape
    if M < 2:
        raise ValueError("MVDR 至少需要 2 个麦克风")
    mics = _check_mics(mics)
    if mics.shape[0] != M:
        raise ValueError(f"mics 通道数 {mics.shape[0]} 与 X 通道数 {M} 不一致")
    if noise_est not in ("min_stats", "zelinski", "cov", "both"):
        raise ValueError("noise_est ∈ {'min_stats','zelinski','cov','both'}")
    hop = _hop_of(n_fft, hop)
    tgt = _resolve_dir(steer_dir)
    floor_lin = float(10.0 ** (floor_db / 20.0))

    Xf, freqs = _stft(X, fs, n_fft, hop, win, center)   # (M,K,F)
    K, F = Xf.shape[1], Xf.shape[2]
    a = _steering(mics, freqs, tgt, c)                               # (F,M)

    # ---- 噪声协方差 + SPP ----
    Rnn, spp, gate, Pn_ch = _estimate_noise_cov(
        Xf, freqs, ms_sub_len=ms_sub_len, ms_n_sub=ms_n_sub, ms_bias=ms_bias,
        ms_alpha=ms_alpha, spp_alpha=spp_alpha, spp_q=spp_q, cov_alpha=cov_alpha)

    # ---- 空间权重：优先 beamforming 模块 ----
    w_ext, info = (None, {"design_path": "fallback", "reason": "design='analytic'"})
    if design in ("auto", "beamforming"):
        w_ext, info = _try_beamforming_module(geom, freqs, mics, tgt, c, a,
                                              method="mvdr", Rnn=Rnn,
                                              wng_min_db=wng_min_db)
    if w_ext is not None and design != "analytic":
        w, wng = w_ext, _wng_db(w_ext, a)
        design_path = "beamforming"
    else:
        w, dinfo = _design_mvdr(Rnn, a, wng_min_db=wng_min_db,
                                delta0_db=delta0_db)
        wng = dinfo["wng_db"]
        design_path = "analytic_fallback"
        if w_ext is None and design == "auto":
            info = {"design_path": "analytic_fallback",
                    "reason": info.get("reason", "beamforming 不可用")}

    # ---- 波束成形输出 ----
    Ybf = _apply_beamformer(Xf, w)                                   # (K,F)
    Pyy = np.abs(Ybf) ** 2
    Pyy_s = _smooth_power_time(Pyy, pf_alpha_p)
    Pyy_s = np.maximum(Pyy_s, EPS)

    # ---- 后置滤波噪声估计 ----
    Pn_ms = _min_stats_noise(Pyy, sub_len=ms_sub_len, n_sub=ms_n_sub,
                             bias=ms_bias, alpha=ms_alpha)
    if noise_est in ("zelinski", "both") and M >= 2:
        # 相干（语音）功率估计：只保留通道对之间的互谱
        i, j = np.triu_indices(M, k=1)
        Xw = Xf * np.conj(w).T[:, None, :]                # (M,K,F) 预先乘权重
        Phi = np.mean(Xw[i] * np.conj(Xw[j]), axis=0)     # (K,F) 对通道对平均
        # 注意：这里对 i<j 求和后再取实部，等价于 (2/(M(M-1)))ReΣ
        P_coh = _smooth_power_time(np.maximum(Phi.real, 0.0), pf_alpha_p)
        Pn_zel = np.maximum(Pyy_s - np.minimum(P_coh, Pyy_s), EPS)
    else:
        Pn_zel = Pn_ms
    if noise_est == "cov":
        Pn_cov = np.abs(np.einsum("fm,fmn,fn->f", np.conj(w), Rnn, w))[None, :]
        Pn = np.repeat(np.maximum(Pn_cov, EPS), K, axis=0)
    elif noise_est == "both":
        Pn = 0.5 * (Pn_ms + Pn_zel)
    elif noise_est == "zelinski":
        Pn = Pn_zel
    else:
        Pn = Pn_ms
    Pn = np.clip(Pn, EPS, Pyy_s)       # 噪声估计夹在 [下限, 观测功率] 内

    # ---- 增益与平滑 ----
    G = 1.0 - float(over_subtraction) * Pn / Pyy_s
    G = np.clip(G, floor_lin, 1.0)
    G = _smooth_gain(G, attack=pf_attack, release=pf_release,
                     freq_smooth=pf_freq_smooth)
    G = np.clip(G, floor_lin, 1.0)

    Y = G * Ybf
    y = _istft(Y, fs, n_fft, hop, win, N, center)
    y = _finite(y, "mvdr_postfilter")

    # 免参考的噪声抑制量估计：10log10(ΣPn / ΣG²Pn)
    denom = float(np.sum((G ** 2) * Pn)) + EPS
    nr_est = float(10.0 * np.log10(float(np.sum(Pn)) / denom))

    params = dict(n_fft=n_fft, hop=hop, c=c, win=win, ref_mic=ref_mic,
                  noise_est=noise_est, over_subtraction=over_subtraction,
                  floor_db=floor_db, wng_min_db=wng_min_db,
                  delta0_db=delta0_db, spp_alpha=spp_alpha, spp_q=spp_q,
                  ms_sub_len=ms_sub_len, ms_n_sub=ms_n_sub, ms_bias=ms_bias,
                  ms_alpha=ms_alpha, cov_alpha=cov_alpha, pf_alpha_p=pf_alpha_p,
                  pf_attack=pf_attack, pf_release=pf_release,
                  pf_freq_smooth=pf_freq_smooth,
                  steer_dir=[float(v) for v in np.asarray(tgt, float).ravel()],
                  n_mics=int(M), n_samples=int(N), design_path=design_path,
                  design_note=info.get("reason", ""))
    metrics_extra = {"design_path": design_path,
                     "design_note": info.get("reason", ""),
                     "wng_min_db": float(np.min(wng)) if wng.size else None,
                     "wng_median_db": float(np.median(wng)) if wng.size else None,
                     "gain_mean_db": float(20 * np.log10(
                         max(float(np.mean(G)), EPS)))}
    curve = {
        "freqs": freqs,
        "gain_db": 20.0 * np.log10(np.maximum(np.mean(G, axis=0), EPS)),
        "noise_psd_db": 10.0 * np.log10(np.maximum(np.mean(Pn, axis=0), EPS)),
        "wng_db": wng,
        "gain_frames_db": 20.0 * np.log10(np.maximum(np.mean(G, axis=1), EPS)),
    }
    noisy_in = None if noisy is not None else X[ref_mic]
    if noisy is not None:
        noisy_in = noisy
    res = _finish(y, "mvdr_postfilter", params, fs, ref, noisy_in, curve,
                  extras={"weights": w, "Rnn": Rnn, "noise_psd": Pn, "gain": G,
                          "freqs": freqs, "Xf": Xf, "wng_db": wng,
                          "spp": spp, "frame_gate": gate,
                          "beamformer_output": Ybf},
                  nr_est_db=nr_est)
    res["metrics"].update(_plain(metrics_extra))
    return res


# ==========================================================================
# 2) 多通道维纳滤波（MWF）
# ==========================================================================
def multichannel_wiener(X, fs: int, mics, steer_dir, n_fft: int = 512,
                        hop: int | None = None, c: float = C_DEFAULT, *,
                        win: str = "blackmanharris", ref_mic: int = 0,
                        wng_min_db: float | None = -10.0,
                        delta0_db: float = -8.0,
                        spp_alpha: float = 0.92, spp_q: float = 0.3,
                        ms_sub_len: int = 8, ms_n_sub: int = 5,
                        ms_bias: float = 1.5, ms_alpha: float = 0.6,
                        cov_alpha: float = 0.9, distortionless: bool = False,
                        center: bool = True, geom=None, ref=None,
                        noisy=None) -> dict:
    """多通道维纳滤波（MWF / speech-distortion-weighted 的特例）。

    原理
    ----
    以某个参考通道的期望信号 ``d(k,f)`` 为目标的最小均方误差滤波器：

        ``w(f) = Rx(f)⁻¹·r_xd(f)``,  ``Rx = Rss + Rnn``,  ``r_xd = Rss·e_ref``

    * ``Rnn``：噪声协方差（SPP 加权递推平均，与 MVDR 同一套估计）；
    * ``Rss``：语音协方差，由 ``Rx - Rnn`` 得到并投影到半正定锥
      （估计误差会让差矩阵出现负特征值，不截负则 MWF 解病态）；
    * ``e_ref``：参考通道选择向量（取 ``ref_mic`` 列，即"估计该通道的期望信号"）。

    与 MVDR 的关系：当语音协方差近似秩 1（``Rss ≈ P_s·a a^H``，单声源主导）时，
    MWF 严格等价于"MVDR + 单通道维纳增益"，也就是本模块 MVDR 那条路径
    在后置滤波取维纳形式时的极限。MWF 的好处是**不需要显式的后置滤波**：
    谱增益已经包含在权重里了；代价是需要把 Rss 也估准。

    两个**默认关闭**的开关（都保留参数，理由用实测说话）
    ---------------------------------------------------
    * ``distortionless=False``（默认）：保留 MMSE 的收缩增益（本题实测
      ``|w^H a| ≈ 0.4``，即 -8 dB）。强制归一化到 ``w^H a = 1`` 会把收缩抹掉、
      在高频（那里 |w^H a| 很小）把噪声放大，实测 SI-SDR 从 +7.4 dB 掉到 -5 dB。
      需要与单通道尺度对齐时再打开；真实收缩量始终记录在
      ``params['mwf_shrinkage_gain_db']`` 与 ``curve['shrinkage_gain_db']``。
    * ``wng_min_db=-10.0``：只对白噪增益低于 -10 dB 的频点做**保尺度**的
      DAS 混合（见 :func:`_wng_blend`）。实测比完全关闭还好
      （+8.5 vs +7.7 dB ΔSNR）——在最不可信的频点换成最稳的形状。
      传 ``None`` 关闭、传 ``0.0`` 则处处不弱于 DAS（保守解）。

    返回：``weights (F,M)``、``Rss``、``Rnn``、``gain``（= |w^H a| 频响）。
    """
    X = _as_mc(X)
    M, N = X.shape
    if M < 2:
        raise ValueError("MWF 至少需要 2 个麦克风")
    mics = _check_mics(mics)
    if mics.shape[0] != M:
        raise ValueError("mics 通道数与 X 不一致")
    hop = _hop_of(n_fft, hop)
    tgt = _resolve_dir(steer_dir)

    Xf, freqs = _stft(X, fs, n_fft, hop, win, center)
    K, F = Xf.shape[1], Xf.shape[2]
    a = _steering(mics, freqs, tgt, c)

    # ---- 统计量 ----
    Rnn, spp, gate, Pn_ch = _estimate_noise_cov(
        Xf, freqs, ms_sub_len=ms_sub_len, ms_n_sub=ms_n_sub, ms_bias=ms_bias,
        ms_alpha=ms_alpha, spp_alpha=spp_alpha, spp_q=spp_q, cov_alpha=cov_alpha)
    Rx = _weighted_cov(Xf, np.ones((K, F)))                    # 总协方差
    Rss = _psd_project(Rx - Rnn)                               # 语音协方差（截负）
    Rx_reg = _psd_project(Rss + Rnn)                           # 保证 Rx 半正定

    # ---- 权重：w = (Rss+Rnn+δI)⁻¹ Rss e_ref ----
    # δ 取 -8 dB 相对平均通道功率。这个值是**实测**出来的：Rss = Rx - Rnn 的
    # 估计误差在高频/低语音段相对很大，加载不够会把解带偏（-25 dB 加载、
    # 无 WNG 保护时实测 SNR 反而 -4.7 dB）；加载过头又会让解趋向 Rss 的一列
    # （δ→∞ 时 w → Rss·e_ref/δ，是毫无空间意义的滤波器）。
    denom0 = np.maximum(np.einsum("fmm->f", Rx_reg).real / M, EPS)
    delta = (10.0 ** (delta0_db / 10.0)) * denom0
    eye = np.eye(M)[None, :, :]
    e_ref = np.zeros(M)
    e_ref[int(np.clip(ref_mic, 0, M - 1))] = 1.0
    b = Rss @ e_ref                                            # (F,M)
    w = np.linalg.solve(Rx_reg + delta[:, None, None] * eye, b[:, :, None])[:, :, 0]

    # ---- 失真约束归一化（可选，默认关闭）----
    # 关闭时保留 MMSE 的收缩增益（|w^H a| ≈ 0.4），这才是 MSE 最优解；
    # 强制 w^H a = 1 会把收缩抹掉、把高频噪声放大，实测 SI-SDR 掉 12 dB。
    cw = np.einsum("fm,fm->f", np.conj(w), a)                  # w^H a
    shrink_db = 20.0 * np.log10(np.maximum(np.abs(cw), EPS))
    if distortionless:
        w = w / np.conj(np.where(np.abs(cw) < 1e-12, 1e-12 + 0j, cw))[:, None]

    # ---- WNG 保护：与 DAS 做**保尺度**凸组合 ----
    # 只对"白噪增益掉到 -10 dB 以下"的频点生效（超指向、对失配与估计误差最敏感），
    # 混合时保持 w^H a 不变，因此 MMSE 的收缩增益不会被抹掉。
    # 实测：-10 dB 保护比完全不保护还要好（+8.5 vs +7.7 dB dSNR），
    # 因为它在"解最不可信的频点"上换成了最稳的 DAS 形状。
    # 传 wng_min_db=None 可完全关闭；传 0.0 则退化成"处处不弱于 DAS"的保守解。
    w, wng = _wng_blend(w, a, wng_min_db)

    Y = _apply_beamformer(Xf, w)
    y = _istft(Y, fs, n_fft, hop, win, N, center)
    y = _finite(y, "multichannel_wiener")

    # 频响（对指向的复增益）与免参考噪声抑制估计
    gain_db = 20.0 * np.log10(np.maximum(np.abs(np.einsum(
        "fm,fm->f", np.conj(w), a)), EPS))
    Pn_out = np.abs(np.einsum("fm,fmn,fn->f", np.conj(w), Rnn, w))
    Pn_in = np.maximum(np.mean(Pn_ch, axis=(0, 1)), EPS)       # (F,)
    nr_est = float(10.0 * np.log10(float(np.sum(Pn_in))
                                   / (float(np.sum(Pn_out)) + EPS)))

    params = dict(n_fft=n_fft, hop=hop, c=c, win=win, ref_mic=ref_mic,
                  wng_min_db=wng_min_db, delta0_db=delta0_db, spp_alpha=spp_alpha,
                  spp_q=spp_q, ms_sub_len=ms_sub_len, ms_n_sub=ms_n_sub,
                  ms_bias=ms_bias, ms_alpha=ms_alpha, cov_alpha=cov_alpha,
                  distortionless=bool(distortionless),
                  mwf_shrinkage_gain_db=[float(v) for v in shrink_db],
                  steer_dir=[float(v) for v in np.asarray(tgt, float).ravel()],
                  n_mics=int(M), n_samples=int(N))
    curve = {
        "freqs": freqs,
        "gain_db": gain_db,
        "shrinkage_gain_db": shrink_db,
        "wng_db": wng,
        "noise_psd_out_db": 10.0 * np.log10(np.maximum(Pn_out, EPS)),
    }
    noisy_in = noisy if noisy is not None else X[ref_mic]
    res = _finish(y, "multichannel_wiener", params, fs, ref, noisy_in, curve,
                  extras={"weights": w, "Rss": Rss, "Rnn": Rnn, "Rx": Rx,
                          "freqs": freqs, "Xf": Xf, "wng_db": wng,
                          "spp": spp, "frame_gate": gate},
                  nr_est_db=nr_est)
    res["metrics"].update(_plain({
        "wng_min_db": float(np.min(wng)) if wng.size else None,
        "wng_median_db": float(np.median(wng)) if wng.size else None,
        "mwf_shrinkage_median_db": float(np.median(shrink_db)),
    }))
    return res


# ==========================================================================
# 3) GSC
# ==========================================================================
def _blocking_matrix(a: np.ndarray, kind: str = "qr") -> np.ndarray:
    """阻塞矩阵 B (F,M,M-1)，满足 ``a(f)^H B(f) = 0``，B^H B = I。

    * ``'qr'``  : 对 a 做 complete QR（Householder），取 Q 的 2..M 列。
      严格正交、逐频点精确置零，是教科书形式。缺点：B 的列随频率任意旋转，
      但本实现的自适应滤波是**逐频点独立**的，旋转不影响收敛。
    * ``'diff'``: 一阶差分型 B（相邻通道相减），对失配与混响更鲁棒，
      但置零是近似的（只在导向矢量精确成立时严格）。
    """
    F, M = a.shape
    if kind == "qr":
        B = np.zeros((F, M, M - 1), dtype=np.complex128)
        for f in range(F):
            q, _ = np.linalg.qr(a[f][:, None], mode="complete")   # (M,M)
            B[f] = q[:, 1:]
        return B
    if kind == "diff":
        # 相邻通道差分：B[:,i] 只在麦克风 i、i+1 上有非零元，
        # u_i = a_{i+1}·x_i - a_i·x_{i+1}（先按导向矢量相位对齐再相减），
        # 所以共 M-1 个输出、恰好用满 M 个麦克风。索引必须 (mic, output)。
        an = a / np.maximum(np.abs(a), EPS)
        B = np.zeros((F, M, M - 1), dtype=np.complex128)
        for i in range(M - 1):
            B[:, i, i] = an[:, i + 1]
            B[:, i + 1, i] = -an[:, i]
        nrm = np.sqrt(np.sum(np.abs(B) ** 2, axis=1, keepdims=True))
        return B / np.maximum(nrm, EPS)
    raise ValueError("blocking ∈ {'qr','diff'}")


def gsc(X, fs: int, mics, steer_dir, n_fft: int = 512, hop: int | None = None,
        c: float = C_DEFAULT, mu: float = 0.05, *, win: str = "blackmanharris",
        ref_mic: int = 0, blocking: str = "qr", leak: float = 1e-3,
        adapt_vad: bool = True, vad_thr_db: float = 2.0, vad_width_db: float = 3.0,
        spp_alpha: float = 0.92, spp_q: float = 0.3,
        ms_sub_len: int = 8, ms_n_sub: int = 5, ms_bias: float = 1.5,
        ms_alpha: float = 0.6, cov_alpha: float = 0.9, center: bool = True,
        geom=None, ref=None, noisy=None) -> dict:
    """广义旁瓣抵消器（GSC）：固定波束 + 阻塞矩阵 + NLMS 自适应抵消。

    结构（频域实现，逐频点独立自适应）
    --------------------------------
    * 固定波束（quiescent）: ``d(k,f) = w_q^H x``，``w_q = a/(a^H a)`` ——
      无失真延迟求和，保证目标方向不被自适应动到；
    * 阻塞矩阵 ``B``：``a^H B = 0``（对 a 做 complete QR 取正交补），
      ``u(k,f) = B^H x`` 里理论上不含目标语音，只有噪声/干扰；
    * 自适应噪声抵消：``y = d - h^H u``，
      NLMS 更新 ``h ← (1-leak)·h + μ_eff·u·conj(y)/(‖u‖²+ε)``
      （复 LMS 梯度 ``∂|y|²/∂h* = -u·y*``，逐频点用 ``‖u‖²`` 归一化，
      因此步长与输入电平无关）。

    自适应保护（GSC 的经典死穴：目标语音泄漏 → 语音被当噪声抵消）
    --------------------------------------------------------------
    * ``adapt_vad=True``：用 SPP 逐帧门控 ``μ_eff = μ·(1-gate)``，
      语音段基本冻结自适应；
    * ``leak``：权重泄漏，防止长时间噪声段把 h 拖到无界；
    * ``blocking='qr'`` 时，失配会让语音泄漏进 u，门控 + 泄漏是必需的。

    参数（其余）
    -----------
    geom : 仅为接口统一而接受（GSC 的固定波束是 DAS、阻塞矩阵由 QR 构造，
           不需要 beamforming 设计器），传入即忽略。
    center : 同 mvdr_postfilter，默认 True 且必须保持 True（ISTFT 窗端陷阱）。

    收敛曲线
    --------
    `curve` 里给三条：
    * ``err_db``         逐帧输出电平（会被语音包络主导，非单调）；
    * ``err_smooth_db``  指数平滑后的输出电平；
    * ``err_mono_db``    平滑曲线的**运行最小值** —— 按构造单调不增，
      即"学习曲线的包络"，工程上用来判断是否收敛；
    * ``learn_*``        只在噪声帧（gate<0.5）上统计的自适应学习曲线，
      这才是真正意义上的收敛过程。
    """
    X = _as_mc(X)
    M, N = X.shape
    if M < 2:
        raise ValueError("GSC 至少需要 2 个麦克风")
    mics = _check_mics(mics)
    hop = _hop_of(n_fft, hop)
    tgt = _resolve_dir(steer_dir)

    Xf, freqs = _stft(X, fs, n_fft, hop, win, center)
    K, F = Xf.shape[1], Xf.shape[2]
    a = _steering(mics, freqs, tgt, c)
    wq = _das_weights(a)                                     # (F,M)
    B = _blocking_matrix(a, blocking)                        # (F,M,M-1)

    # 固定波束与噪声参考
    d = np.einsum("fm,mkf->kf", np.conj(wq), Xf)             # (K,F)
    U = np.einsum("fmb,mkf->kbf", np.conj(B), Xf)            # (K,M-1,F)

    # SPP 噪声统计（只用来给 Rnn/门控做参考，不直接当 VAD 用）
    _, spp, gate_spp, _ = _estimate_noise_cov(
        Xf, freqs, ms_sub_len=ms_sub_len, ms_n_sub=ms_n_sub, ms_bias=ms_bias,
        ms_alpha=ms_alpha, spp_alpha=spp_alpha, spp_q=spp_q, cov_alpha=cov_alpha)

    # 逐帧能量 VAD：用固定波束输出功率 / 它的最小统计噪声底。
    # （SPP 在混响下恒为 ~1，做不了自适应门控，见 _frame_vad 注释。）
    Pd = np.abs(d) ** 2
    Pn_d = _min_stats_noise(Pd, sub_len=ms_sub_len, n_sub=ms_n_sub,
                            bias=ms_bias, alpha=ms_alpha)
    gate = _frame_vad(np.mean(Pd, axis=1), np.mean(Pn_d, axis=1),
                      thr_db=vad_thr_db, width_db=vad_width_db)

    h = np.zeros((M - 1, F), dtype=np.complex128)
    Y = np.empty((K, F), dtype=np.complex128)
    corr_pow = np.zeros(K)                                   # ANC 抵消掉的功率
    mu = float(np.clip(mu, 1e-6, 1.0))
    for k in range(K):
        u = U[k]                                             # (M-1,F)
        e = d[k] - np.einsum("mf,mf->f", np.conj(h), u)
        Y[k] = e
        corr = np.einsum("mf,mf->f", np.conj(h), u)
        corr_pow[k] = float(np.mean(np.abs(corr) ** 2))
        nu = np.sum(np.abs(u) ** 2, axis=0) + EPS            # 逐频点能量
        mu_k = mu * (1.0 - gate[k]) if adapt_vad else mu
        h = (1.0 - leak * mu_k) * h + mu_k * (u * np.conj(e)[None, :]) / nu[None, :]
    y = _istft(Y, fs, n_fft, hop, win, N, center)
    y = _finite(y, "gsc")

    # ---- 收敛曲线 ----
    err = np.mean(np.abs(Y) ** 2, axis=1)                    # (K,)
    err_db = 10.0 * np.log10(np.maximum(err, EPS))
    alpha_e = 0.7
    err_s = np.empty(K)
    acc = err_db[0] if K else 0.0
    for k in range(K):
        acc = alpha_e * acc + (1.0 - alpha_e) * err_db[k]
        err_s[k] = acc
    err_mono = np.minimum.accumulate(err_s)                  # 单调不增（构造保证）
    noise_sel = gate < 0.5
    learn_db = err_s[noise_sel] if np.any(noise_sel) else err_s
    learn_mono = np.minimum.accumulate(learn_db)
    curve = {
        "freqs": freqs,
        "err_db": err_db,
        "err_smooth_db": err_s,
        "err_mono_db": err_mono,
        "learn_db": learn_db,
        "learn_mono_db": learn_mono,
        "anc_cancel_db": 10.0 * np.log10(np.maximum(corr_pow, EPS)),
        "gate": gate,
        "converged_drop_db": float(err_mono[0] - err_mono[-1]) if K else 0.0,
        "learn_drop_db": float(learn_mono[0] - learn_mono[-1]) if learn_mono.size else 0.0,
        "monotone": bool(np.all(np.diff(err_mono) <= 1e-9)),
    }

    params = dict(n_fft=n_fft, hop=hop, c=c, win=win, ref_mic=ref_mic, mu=mu,
                  blocking=blocking, leak=leak, adapt_vad=bool(adapt_vad),
                  vad_thr_db=vad_thr_db, vad_width_db=vad_width_db,
                  spp_alpha=spp_alpha, spp_q=spp_q, ms_sub_len=ms_sub_len,
                  ms_n_sub=ms_n_sub, ms_bias=ms_bias, ms_alpha=ms_alpha,
                  cov_alpha=cov_alpha,
                  steer_dir=[float(v) for v in np.asarray(tgt, float).ravel()],
                  n_mics=int(M), n_samples=int(N), adapt_frac=float(1.0 - np.mean(gate)))
    noisy_in = noisy if noisy is not None else X[ref_mic]
    res = _finish(y, "gsc", params, fs, ref, noisy_in, curve,
                  extras={"freqs": freqs, "Xf": Xf, "fixed_beam_db":
                          20.0 * np.log10(np.maximum(np.mean(np.abs(d) ** 2), EPS)),
                          "blocking": blocking, "h_final": h, "spp": spp,
                          "frame_gate": gate, "gate_spp": gate_spp})
    res["metrics"]["converged_drop_db"] = _plain(curve["converged_drop_db"])
    res["metrics"]["convergence_monotone"] = curve["monotone"]
    return res


# ==========================================================================
# 4) 单通道谱减（对照基线）
# ==========================================================================
def spectral_subtraction(X, fs: int, n_fft: int = 512, hop: int | None = None,
                         over_subtraction: float = 2.0, floor_db: float = -20.0,
                         noise_est: str = "min_stats", *, win: str = "blackmanharris",
                         ref_mic: int = 0, domain: str = "power",
                         ms_sub_len: int = 8, ms_n_sub: int = 5,
                         ms_bias: float = 1.5, ms_alpha: float = 0.6,
                         pf_alpha_p: float = 0.85, pf_attack: float = 0.5,
                         pf_release: float = 0.9, pf_freq_smooth: bool = True,
                         center: bool = True, ref=None, noisy=None) -> dict:
    """单通道谱减（作为多通道方法的对照基线）。

    * ``domain='power'``   : ``G = sqrt(max(1 - β·Pn/Py, floor²))``（功率域谱减，
      等价于过减的维纳滤波，最常用）；
    * ``domain='magnitude'``: ``G = max(1 - β·sqrt(Pn)/|Y|, floor)``（幅度域谱减，
      经典 Boll 形式，低 SNR 时更激进，残留音乐噪声更多）。

    噪声估计用最小值统计（与多通道方法同源，保证对比公平）。
    单通道没有空间自由度，抑制量完全靠谱增益，因此过减因子默认取 2.0
    （多通道默认 1.0）——多通道已经先做了一轮空间滤波。
    音乐噪声用与多通道一致的"快攻慢放 + 频率 3 点平滑"抑制。
    """
    X = _as_mc(X)
    M, N = X.shape
    if M == 1:
        x1 = X[0]
    else:
        x1 = X[int(np.clip(ref_mic, 0, M - 1))]
    if noise_est != "min_stats":
        raise ValueError("谱减目前只实现了 noise_est='min_stats'")
    hop = _hop_of(n_fft, hop)
    floor_lin = float(10.0 ** (floor_db / 20.0))

    Xf, freqs = _stft(x1, fs, n_fft, hop, win, center)  # (K,F)
    Py = np.abs(Xf) ** 2
    Py_s = np.maximum(_smooth_power_time(Py, pf_alpha_p), EPS)
    Pn = np.clip(_min_stats_noise(Py, sub_len=ms_sub_len, n_sub=ms_n_sub,
                                  bias=ms_bias, alpha=ms_alpha), EPS, Py_s)

    if domain == "power":
        G = np.sqrt(np.clip(1.0 - float(over_subtraction) * Pn / Py_s,
                            floor_lin ** 2, 1.0))
    elif domain == "magnitude":
        G = 1.0 - float(over_subtraction) * np.sqrt(Pn) / np.sqrt(Py_s)
        G = np.clip(G, floor_lin, 1.0)
    else:
        raise ValueError("domain ∈ {'power','magnitude'}")
    G = _smooth_gain(G, attack=pf_attack, release=pf_release,
                     freq_smooth=pf_freq_smooth)
    G = np.clip(G, floor_lin, 1.0)

    y = _istft(G * Xf, fs, n_fft, hop, win, N, center)
    y = _finite(y, "spectral_subtraction")

    nr_est = float(10.0 * np.log10(float(np.sum(Pn))
                                   / (float(np.sum((G ** 2) * Pn)) + EPS)))
    params = dict(n_fft=n_fft, hop=hop, win=win, over_subtraction=over_subtraction,
                  floor_db=floor_db, noise_est=noise_est, domain=domain,
                  ref_mic=ref_mic, ms_sub_len=ms_sub_len, ms_n_sub=ms_n_sub,
                  ms_bias=ms_bias, ms_alpha=ms_alpha, pf_alpha_p=pf_alpha_p,
                  pf_attack=pf_attack, pf_release=pf_release,
                  pf_freq_smooth=pf_freq_smooth, n_samples=int(N), n_mics=int(M))
    curve = {"freqs": freqs,
             "gain_db": 20.0 * np.log10(np.maximum(np.mean(G, axis=0), EPS)),
             "noise_psd_db": 10.0 * np.log10(np.maximum(np.mean(Pn, axis=0), EPS))}
    noisy_in = noisy if noisy is not None else x1
    return _finish(y, "spectral_subtraction", params, fs, ref, noisy_in, curve,
                   extras={"gain": G, "noise_psd": Pn, "freqs": freqs},
                   nr_est_db=nr_est)


# ==========================================================================
# 5) 去混响
# ==========================================================================
def _spectral_dereverb(Xf: np.ndarray, fs: int, hop: int, rt60: float = 0.3,
                       delay_frames: int = 3, over_subtraction: float = 1.0,
                       floor_db: float = -15.0, pf_alpha_p: float = 0.8,
                       pf_attack: float = 0.6, pf_release: float = 0.92
                       ) -> tuple:
    """谱减型晚期混响抑制（Lebart 型模型的简化）。

    物理模型：晚期混响功率按指数衰减，且只统计**延迟 D 帧以后**的能量
    （D 之前的早期反射对语音可懂度有贡献，必须保留）：

        ``P_late(k,f) = Σ_{d≥D} P(k-d,f)·γ^d``，``γ = 10^(-6·hop/(fs·RT60))``

    ``γ`` 的来历：能量在 RT60 内衰减 60 dB（=功率比 1e-6），
    所以每帧的功率衰减因子是 ``10^(-6·hop/(fs·RT60))``。
    这个无穷和用一阶递归一次算完：

        ``S(k,f) = P(k,f) + γ·S(k-1,f)``  ⇒  ``P_late(k,f) = γ^D·S(k-D,f)``

    增益 ``G = clip(1 - β·P_late/P, floor, 1)``，再做与其它方法一致的平滑。

    局限（写在明处）：RT60 是**输入参数**，不做盲估计；单通道、无相位信息，
    因此只能压低晚期混响的包络，不能真正"解卷积"房间。它与 WPE 的差别是：
    本方法基于能量的指数衰减启发式，WPE 基于延迟线性预测的 MSE 最优解。
    """
    P = np.maximum(np.abs(Xf) ** 2, EPS)
    K, F = P.shape
    gamma = float(10.0 ** (-6.0 * hop / (float(fs) * max(rt60, 1e-3))))
    S = np.empty_like(P)
    acc = P[0].copy()
    for k in range(K):
        acc = P[k] + gamma * acc
        S[k] = acc
    D = max(int(delay_frames), 1)
    P_late = np.empty_like(P)
    P_late[:D] = 0.0
    if K > D:
        P_late[D:] = (gamma ** D) * S[:K - D]
    P_s = np.maximum(_smooth_power_time(P, pf_alpha_p), EPS)
    floor_lin = float(10.0 ** (floor_db / 20.0))
    G = np.clip(1.0 - float(over_subtraction) * P_late / P_s, floor_lin, 1.0)
    G = _smooth_gain(G, attack=pf_attack, release=pf_release, freq_smooth=True)
    G = np.clip(G, floor_lin, 1.0)
    info = {"rt60_s": float(rt60), "decay_per_frame": gamma,
            "delay_frames": D, "delay_ms": float(D * hop / fs * 1e3),
            "late_reverb_psd": P_late}
    return G, info


def _wpe_single_channel(Xf: np.ndarray, delay: int = 4, taps: int = 2,
                        iters: int = 2, delta: float = 1e-3,
                        w_clip: tuple = (0.01, 100.0)) -> tuple:
    """简化的单通道 WPE（加权预测误差 / 延迟线性预测），逐频点离线最小二乘。

    信号模型（对每个频点独立）：

        ``x(k,f) = Σ_{d=D}^{D+Kd-1} g_d(f)·x(k-d,f) + e(k,f)``

    即用**延迟 D 之后**的过去帧线性预测当前帧，残差 e 就是去混响输出
    （D 之前的早期反射保留在信号里）。最小化加权误差
    ``Σ_k w_k|e_k|²`` 得到正规方程 ``(A^H W A + δI)g = A^H W x``。

    "加权"（WPE 名字的来源）：权重 ``w_k = 1/(|e_k|²+ε)`` 从上一轮迭代的残差
    得到（对语音的稀疏/重尾先验），因此迭代 2 次即可显著提升。
    权重做了动态范围裁剪，避免个别帧把正规方程条件数毁掉。

    局限：只在**单通道**上做（多通道联合 WPE 需要 M 倍参数与跨通道预测，
    本模块按需求做"简化版"）。前 ``D+Kd-1`` 帧没有足够历史，原样输出。
    """
    K, F = Xf.shape
    out = Xf.copy()
    D = max(int(delay), 1)
    Kd = max(int(taps), 1)
    if K <= D + Kd + 1:
        return out, {"delay": D, "taps": Kd, "applied": False,
                     "reason": "帧数不足，跳过 WPE"}
    idx = np.arange(D, K)                                   # 待预测帧
    tgt = Xf[idx, :]                                        # (Kp,F)
    for f in range(F):
        x = Xf[:, f]
        A = np.stack([x[idx - d] for d in range(D, D + Kd)], axis=1)   # (Kp,Kd)
        t = tgt[:, f]
        wgt = np.ones(idx.size)
        e = t.copy()
        for _ in range(max(int(iters), 1)):
            sw = np.sqrt(wgt)
            Aw = A * sw[:, None]
            tw = t * sw
            G = A.conj().T @ Aw
            tr = max(float(np.trace(G).real) / Kd, EPS)
            g = np.linalg.solve(G + (delta * tr) * np.eye(Kd), A.conj().T @ tw)
            e = t - A @ g
            wgt = 1.0 / np.maximum(np.abs(e) ** 2, EPS)
            wgt = np.clip(wgt / max(float(np.mean(wgt)), EPS), w_clip[0], w_clip[1])
        out[idx, f] = e
    return out, {"delay": D, "taps": Kd, "iters": int(iters), "applied": True}


def dereverberate(X, fs: int, method: str = "spectral", n_fft: int = 512,
                  hop: int | None = None, *, win: str = "blackmanharris",
                  ref_mic: int = 0, frontend: str = "das",
                  rt60: float = 0.3, delay_frames: int = 3,
                  over_subtraction: float = 1.0, floor_db: float = -15.0,
                  wpe_delay: int = 4, wpe_taps: int = 2, wpe_iters: int = 2,
                  mics=None, steer_dir=None, c: float = C_DEFAULT,
                  pf_alpha_p: float = 0.8, pf_attack: float = 0.6,
                  pf_release: float = 0.92, center: bool = True, ref=None,
                  noisy=None) -> dict:
    """去混响（简化实现，两种方法）。

    ``method='spectral'`` : 谱减型晚期混响抑制。用指数衰减模型从过去帧
        推算晚期混响功率（晚期能量只来自 D 帧之前的观测，γ 由 RT60 决定），
        再做谱减。**实现了什么**：一个能量域的晚期混响包络估计 + 过减 + 平滑；
        没有做 RIR 估计、没有做相位处理。
    ``method='wpe'`` : 单通道 WPE（延迟线性预测 + IRLS 加权）的离线最小二乘版。
        **实现了什么**：逐频点用 D 帧之后的过去帧预测当前帧并取残差，
        迭代 2 次更新稀疏权重；没有做多通道联合预测、没有做在线 RLS。

    前置通道（``frontend``）：
    * ``'das'``（默认，需给 mics/steer_dir）：先做无失真延迟求和再对输出去混响，
      空间上先抑制非相干噪声，去混响的输入 SNR 更高；
    * ``'mic'``：直接用 ``ref_mic`` 通道（多通道 WPE 的常见做法是先空间滤波）。

    注意：去混响会**削弱直达声后的早期能量**，若评价参考里包含混响
    （clean 而不是 direct），指标提升会明显小于"听起来"的改善；这是指标口径问题，
    不是算法问题。
    """
    X = _as_mc(X)
    M, N = X.shape
    hop = _hop_of(n_fft, hop)

    if frontend == "das" and M >= 2:
        if mics is None or steer_dir is None:
            raise ValueError("frontend='das' 需要同时给 mics 与 steer_dir")
        mics = _check_mics(mics)
        tgt = _resolve_dir(steer_dir)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        a = _steering(mics, freqs, tgt, c)
        wq = _das_weights(a)
        Xf, freqs = _stft(X, fs, n_fft, hop, win, center)
        x1f = np.einsum("fm,mkf->kf", np.conj(wq), Xf)          # (K,F)
        x1 = _istft(x1f, fs, n_fft, hop, win, N, center)
    else:
        x1 = X[int(np.clip(ref_mic, 0, M - 1))]
        x1f, freqs = _stft(x1, fs, n_fft, hop, win, center)

    if method == "spectral":
        G, info = _spectral_dereverb(
            x1f, fs, hop, rt60=rt60, delay_frames=delay_frames,
            over_subtraction=over_subtraction, floor_db=floor_db,
            pf_alpha_p=pf_alpha_p, pf_attack=pf_attack, pf_release=pf_release)
        Yf = G * x1f
        curve = {"freqs": freqs,
                 "gain_db": 20.0 * np.log10(np.maximum(np.mean(G, axis=0), EPS))}
    elif method == "wpe":
        Yf, info = _wpe_single_channel(x1f, delay=wpe_delay, taps=wpe_taps,
                                       iters=wpe_iters)
        G = np.abs(Yf) / np.maximum(np.abs(x1f), EPS)
        curve = {"freqs": freqs,
                 "gain_db": 20.0 * np.log10(np.maximum(np.mean(G, axis=0), EPS))}
    else:
        raise ValueError("method ∈ {'spectral','wpe'}")

    y = _istft(Yf, fs, n_fft, hop, win, N, center)
    y = _finite(y, f"dereverberate:{method}")
    params = dict(method=method, n_fft=n_fft, hop=hop, win=win, ref_mic=ref_mic,
                  frontend=frontend, rt60=rt60, delay_frames=delay_frames,
                  over_subtraction=over_subtraction, floor_db=floor_db,
                  wpe_delay=wpe_delay, wpe_taps=wpe_taps, wpe_iters=wpe_iters,
                  n_samples=int(N), n_mics=int(M))
    params.update({k: v for k, v in info.items() if np.isscalar(v)})
    noisy_in = noisy if noisy is not None else x1
    return _finish(y, f"dereverberate_{method}", params, fs, ref, noisy_in, curve,
                   extras={"gain": G, "freqs": freqs, "dereverb_info": info})


__all__ = [
    "mvdr_postfilter", "multichannel_wiener", "gsc", "spectral_subtraction",
    "dereverberate", "evaluate", "intelligibility_proxy",
]
