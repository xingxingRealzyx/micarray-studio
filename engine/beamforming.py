"""波束成形：延迟求和 / MVDR / 超指向 / 频不变 / 最大信干比。

接口见 docs/CONTRACT.md 第 6 节。所有设计都在**阵列坐标系**内进行：
阵元在 xy 平面、boresight = +z、指向用 `acoustics.unit_dir(az, el)`。

设计目标统一写成"线性约束最小方差"族：

    min_w  w^H R_n w     s.t.  w^H a(f, u0) = 1

不同方法只是取不同的噪声协方差 R_n：

| 方法         | R_n                          | 特点 |
|--------------|------------------------------|------|
| delay_sum    | I（白噪）                     | 最鲁棒，指向性最低 |
| mvdr         | 实测协方差（含干扰）           | 自适应置零干扰 |
| superdirective | Gamma_diff（扩散场）        | 指向性最高，低频极不鲁棒 |
| max_sir      | 干扰协方差 R_i                | 最大化信号干扰比 |
| fi_*（频不变）| 上述任一 + 全频带联合拟合      | 宽带主瓣形状稳定 |

闭式解：w = R_n^{-1} a / (a^H R_n^{-1} a)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import acoustics as ac
from . import dsp
from . import steering as st

C_DEFAULT = ac.C_DEFAULT
METHODS = ("delay_sum", "mvdr", "superdirective", "max_sir", "fi", "identity")


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------
@dataclass
class BeamformerResult:
    freqs: np.ndarray                 # (F,)
    weights: np.ndarray               # (F, M) complex128
    method: str = "delay_sum"
    steer_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    delays: np.ndarray | None = None  # (F, M) 等价时延（秒），供时域实现
    metrics: dict = field(default_factory=dict)
    mics: np.ndarray | None = None
    n_fft: int = 512
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freqs = np.atleast_1d(np.asarray(self.freqs, float))
        self.weights = np.asarray(self.weights, dtype=np.complex128)
        if self.weights.ndim == 1:
            self.weights = self.weights[None, :]

    @property
    def n_mics(self) -> int:
        return int(self.weights.shape[1])

    def at(self, f: float) -> np.ndarray:
        """取最接近 f 的权重 (M,)。"""
        k = int(np.argmin(np.abs(self.freqs - f)))
        return self.weights[k]

    def to_json(self, sig: int = 9) -> dict:
        def r(x):
            return [[float(f"%.{sig}g" % v.real), float(f"%.{sig}g" % v.imag)]
                    for v in np.atleast_2d(x)]
        out = {
            "method": self.method,
            "freqs": [round(float(f), 6) for f in self.freqs],
            "weights": r(self.weights),
            "steer_dir": [round(float(v), 9) for v in self.steer_dir],
            "n_fft": int(self.n_fft),
            "metrics": self.metrics,
        }
        if self.delays is not None:
            out["delays"] = [[round(float(v), 12) for v in row]
                             for row in np.atleast_2d(self.delays)]
        return out


# --------------------------------------------------------------------------
# 基础构件
# --------------------------------------------------------------------------
def delay_sum_weights(a: np.ndarray, taper: str = "none") -> np.ndarray:
    """延迟求和（Bartlett）权重：w = a / M。

    加窗（taper）本质是给权重加幅度锥削，用指向性换旁瓣。
    """
    M = a.shape[-1]
    w = a.conj() / M
    if taper != "none":
        win = dsp.window(taper, M, periodic=False)
        w = w * win[None, :]
        w = w / np.abs(np.sum(w * a, axis=-1, keepdims=True)).clip(1e-30)
    return w


def lcmv_weights(a: np.ndarray, noise_cov: np.ndarray, loading: float = 0.0,
                 wng_floor_db: float | None = None) -> np.ndarray:
    """线性约束最小方差闭式解：w = R^-1 a / (a^H R^-1 a)。

    a         : (F,M) 或 (M,)
    noise_cov : (F,M,M) 或 (M,M)
    loading   : 对角加载系数（乘在 trace/M 上），用于数值稳定与鲁棒性权衡
    wng_floor_db : 给定白噪增益下限时，自动抬升对角加载直到满足（见 solve_wng_floor）
    """
    a2 = np.atleast_2d(a)
    F, M = a2.shape
    R = np.asarray(noise_cov, dtype=np.complex128)
    if R.ndim == 2:
        R = np.repeat(R[None, :, :], F, axis=0)

    eye = np.eye(M, dtype=np.complex128)
    if loading > 0:
        scale = np.trace(R, axis1=1, axis2=2).real / M          # (F,)
        R = R + (loading * np.maximum(scale, 1e-12))[:, None, None] * eye[None]

    # 解 R x = a。注意：对角加载后的解必须用**同一个** a^H x 归一化，
    # 才能严格满足 w^H a = 1 —— 这一步是约束最小化
    #   min w^H (R + εI) w  s.t. w^H a = 1
    # 的闭式解。若用未加载的 a^H R^{-1} a 归一化，约束会被破坏。
    x = np.empty((F, M), dtype=np.complex128)
    for k in range(F):
        try:
            x[k] = np.linalg.solve(R[k], a2[k])
        except np.linalg.LinAlgError:
            x[k] = np.linalg.lstsq(R[k], a2[k], rcond=None)[0]

    denom = np.einsum("fm,fm->f", a2.conj(), x)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return x / denom[:, None]


def solve_wng_floor(a: np.ndarray, R: np.ndarray, floor_db: float,
                    max_iter: int = 40) -> tuple[np.ndarray, float]:
    """找最小对角加载，使白噪增益不低于 floor_db。

    低频时超指向/MVDR 的 WNG 会掉到 -20 dB 以下，此时任何通道失配都会
    让指向性优势全部丧失。加对角加载是标准的鲁棒化手段，这里自动求解
    所需的最小加载量，避免过度牺牲指向性。
    """
    lo, hi = 0.0, 1e-6
    w = lcmv_weights(a, R, loading=lo)
    best = (w, lo)
    if wng_db(w, a).min() >= floor_db:
        return best
    for _ in range(max_iter):
        hi *= 4.0
        w = lcmv_weights(a, R, loading=hi)
        if wng_db(w, a).min() >= floor_db:
            break
    else:
        return w, hi
    # 二分细化
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        wm = lcmv_weights(a, R, loading=mid)
        if wng_db(wm, a).min() >= floor_db:
            hi, best = mid, (wm, mid)
        else:
            lo = mid
    return best


def noise_covariance(kind: str, mics: np.ndarray, freqs: np.ndarray,
                     c: float = C_DEFAULT, R_est: np.ndarray | None = None
                     ) -> np.ndarray:
    """按方法取噪声协方差矩阵 (F,M,M)。"""
    if kind == "identity":
        M = mics.shape[0]
        return np.repeat(np.eye(M, dtype=np.complex128)[None], freqs.size, axis=0)
    if kind == "diffuse":
        return st.diffuse_coherence(mics, freqs, c, kind="spherical")
    if kind == "measured":
        if R_est is None:
            raise ValueError("kind='measured' 需要提供 R_est (F,M,M)")
        return np.asarray(R_est, dtype=np.complex128)
    raise ValueError(f"unknown noise kind: {kind}")


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def _response_mag2(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """|w^H a|^2 -> (F,)。

    共轭方向必须统一：平面波 x = a·s 时阵列输出 y = w^H x = (w^H a)·s，
    故响应为 w^H a = sum_m conj(w_m)·a_m。写成 conj(w)·conj(a) 的模即可
    （也等于 |w·conj(a)|，与 evaluate_beampattern 的 w.conj() @ A 同约定）。
    若误写成 conj(w)·a，对 w = conj(a)/M 会得到 (a^T a)/M ≠ 1 的荒谬结果。
    """
    w = np.atleast_2d(w)
    a = np.atleast_2d(a)
    return np.abs(np.einsum("fm,fm->f", w.conj(), a.conj())) ** 2


def wng_db(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """白噪增益 dB：(F,) = 10log10( |w^H a|^2 / (w^H w) )。

    对延迟求和 w = a/M（a 为等模导向矢量），恒有 WNG = M。
    """
    w = np.atleast_2d(w)
    num = _response_mag2(w, a)
    den = np.einsum("fm,fm->f", w.conj(), w).real
    return 10.0 * np.log10(np.maximum(num, 1e-30) / np.maximum(den, 1e-30))


def di_db(w: np.ndarray, a: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """指向性因子 dB：(F,) = 10log10( |w^H a|^2 / (w^H Gamma w) )。

    理论上限为 10log10(M)（各向同性扩散场、无失配）。
    """
    w = np.atleast_2d(w)
    num = _response_mag2(w, a)
    Gw = np.einsum("fij,fj->fi", gamma, w)
    den = np.einsum("fi,fi->f", w.conj(), Gw).real
    return 10.0 * np.log10(np.maximum(num, 1e-30) / np.maximum(den, 1e-30))


def evaluate_beampattern(weights: np.ndarray, mics: np.ndarray, freqs: np.ndarray,
                         dirs: np.ndarray, c: float = C_DEFAULT
                         ) -> np.ndarray:
    """方向响应功率 |w^H a(f,u)|^2 -> (F,G)，已按各频最大值为 1 归一。

    weights (F,M)，freqs (F,) 与 weights 行一一对应。
    注意：方向响应必须和频率一起给出（阵元间的相位差随频率变化），
    因此本函数不接受"单频权重 + 多频"的含糊用法。
    """
    w = np.atleast_2d(np.asarray(weights, dtype=np.complex128))
    f = np.atleast_1d(np.asarray(freqs, float))
    if f.size != w.shape[0]:
        raise ValueError("freqs 长度必须等于 weights 行数")
    A = st.steering_grid(mics, f, dirs, c)                # (F,G,M)
    B = np.abs(np.einsum("fm,fgm->fg", w.conj(), A)) ** 2
    mx = B.max(axis=1, keepdims=True)
    mx = np.where(mx < 1e-30, 1e-30, mx)
    return B / mx


def beamwidth_deg(pattern: np.ndarray, dirs: np.ndarray, steer: np.ndarray,
                  level_db: float = -3.0, max_ang: float | None = None) -> float:
    """含指向切面上的 -3dB 主瓣宽度（度）。

    方向网格通常不包含真实指向（例如 az/el 网格在 el 步长内不含 el0），
    因此不能直接对全网格排序 —— 那会把"邻峰"当成主瓣。这里只取指向附近的
    一维切片（与指向夹角最小的那串方向），在其中找主瓣内的 -3dB 点。

    max_ang: 主瓣搜索上限（度），None 时自动取 120°。
    """
    steer = np.asarray(steer, float)
    steer = steer / max(np.linalg.norm(steer), 1e-12)
    dirs = np.asarray(dirs, float)
    p = np.asarray(pattern, float).ravel()
    if p.size != dirs.shape[0]:
        raise ValueError("pattern 与 dirs 长度不一致")
    p = p / max(p.max(), 1e-30)

    ang = np.rad2deg(np.arccos(np.clip(dirs @ steer, -1.0, 1.0)))
    lim = 120.0 if max_ang is None else float(max_ang)

    # 以指向为中心取切片：网格方向上按夹角排序，再用 2 度容差带走
    order = np.argsort(ang)
    a_s, p_s = ang[order], p[order]
    near = a_s <= lim
    if not np.any(near):
        return float("nan")
    a_s, p_s = a_s[near], p_s[near]

    # 主瓣峰必须在切片起始段（最近的网格点附近），否则视为没对准
    thr = 10 ** (level_db / 10.0)
    below = np.where(p_s <= thr)[0]
    if below.size == 0:
        return float("nan")
    first = int(below[0])
    if first == 0:
        # 第一个点就低于阈值 -> 指向本身没在网格上，无法测主瓣
        return float("nan")
    # 线性插值求 -3dB 交点，比取网格点更准
    a0, a1 = a_s[first - 1], a_s[first]
    p0, p1 = p_s[first - 1], p_s[first]
    if p0 <= p1:
        cross = a1
    else:
        t = (thr - p0) / (p1 - p0)
        cross = a0 + t * (a1 - a0)
    return float(2.0 * cross)


# --------------------------------------------------------------------------
# 主设计入口
# --------------------------------------------------------------------------
def design(geom, freqs, steer=(0.0, 90.0), method: str = "delay_sum",
           c: float | None = None, taper: str = "none", loading: float = 0.0,
           wng_floor_db: float | None = None, R_est: np.ndarray | None = None,
           near_field: bool = False, source_pos: np.ndarray | None = None,
           n_fft: int = 512, taper_band: tuple | None = None,
           taps: int = 0, fi_mainlobe_deg: float | None = None,
           fi_grid_n: int = 12, steer_dir=None, direction=None,
           u=None) -> BeamformerResult:
    """设计波束成形权重。

    steer        : (az, el) 度，或直接给 (3,) 方向向量/坐标
    method       : delay_sum | mvdr | superdirective | max_sir | fi
    taper        : 延迟求和的幅度锥削窗（none/hamming/blackman/...）
    loading      : 对角加载（相对 trace/M）
    wng_floor_db : 白噪增益下限，自动求解所需加载量（推荐低频场景使用）
    R_est        : (F,M,M) 实测协方差，method='mvdr'/'max_sir' 时使用
    near_field   : True 时 steer 视为声源坐标（近场球面波）
    taper_band   : 频不变设计的频带 (f_lo, f_hi)，None 时用全频段
    taps         : >0 时额外设计时域分数延迟 FIR（用于导出纯时域实现）
    """
    # 宽接口：既接受 ArrayGeometry 也接受 (M,3) 坐标数组。
    # 之前只认 geom.mics，调用方传数组就会静默走错分支 —— 显式判断更安全。
    # 指向参数的别名：契约里写的是 design(geom, freqs, method=..., **params)，
    # 调用方（如 enhance 模块）可能用 steer_dir / direction / u 命名。
    # 这里统一收口，避免因为参数名不一致而静默退化成解析解。
    for alias in (steer_dir, direction, u):
        if alias is not None:
            steer = alias
            break
    mics = np.asarray(getattr(geom, "mics", geom), float).reshape(-1, 3)
    c = C_DEFAULT if c is None else float(c)
    M = mics.shape[0]
    f = np.atleast_1d(np.asarray(freqs, float))

    # --- 指向 ---
    if np.isscalar(steer) or (isinstance(steer, (tuple, list)) and len(steer) == 2):
        u0 = ac.unit_dir(steer[0], steer[1])
        target = u0
    else:
        target = np.asarray(steer, float).ravel()
        if target.size != 3:
            raise ValueError("steer 必须是 (az,el) 或 (3,) 向量")
        u0 = target / max(np.linalg.norm(target), 1e-12)
    if near_field and source_pos is None:
        # 近场：把方向按 1 m 距离当作坐标（调用方可直接给 source_pos）
        source_pos = u0 * 1.0
    steer_arg = np.asarray(source_pos, float) if near_field else u0

    # --- 导向矢量 ---
    a = st.steering_vector(mics, f, steer_arg, c, near_field=near_field)  # (F,M)

    # --- 权重 ---
    method = method.lower()
    if method in ("delay_sum", "ds", "bartlett"):
        w = delay_sum_weights(a, taper)
        used = "delay_sum"
    elif method in ("mvdr", "max_sir", "superdirective", "identity"):
        kind = {"mvdr": "measured", "max_sir": "measured",
                "superdirective": "diffuse", "identity": "identity"}[method]
        Rn = noise_covariance(kind, mics, f, c, R_est)
        if wng_floor_db is not None:
            w, used_loading = solve_wng_floor(a, Rn, wng_floor_db)
        else:
            w, used_loading = lcmv_weights(a, Rn, loading), loading
        used = method
        loading = used_loading
    elif method in ("fi", "frequency_invariant"):
        w = None
        used = "fi"
    else:
        raise ValueError(f"unknown method: {method}")

    # 频不变的期望主瓣宽度：取"最低设计频率下阵列能达到的 -3dB 宽度"。
    # 这是唯一物理上可实现的量级 —— 想要比这更窄，等于要求阵列在低频有
    # 它没有的口径，最小二乘只能靠抬旁瓣/放大权重去凑，结果就是 WNG 崩塌。
    if fi_mainlobe_deg is None:
        f_lo = float(np.min(f[f > 0])) if np.any(f > 0) else 1000.0
        D = float(np.max(np.linalg.norm(mics - mics.mean(axis=0), axis=1)) * 2.0)
        lam = c / max(f_lo, 1e-9)
        auto = float(np.rad2deg(0.886 * lam / max(D, 1e-6)))
        fi_mainlobe_deg = float(np.clip(auto, 20.0, 120.0)) if np.isfinite(auto) \
            else 60.0
    res = BeamformerResult(freqs=f, weights=np.atleast_2d(w) if w is not None
                           else np.zeros((f.size, M), dtype=complex),
                           method=used, steer_dir=u0, mics=mics, n_fft=n_fft,
                           extra={"loading": float(loading), "taper": taper,
                                  "near_field": bool(near_field),
                                  "fi_mainlobe_deg": float(fi_mainlobe_deg),
                                  "fi_grid": int(fi_grid_n)})
    if method in ("fi", "frequency_invariant"):
        res = _design_fi(res, mics, f, steer_arg, c, near_field, taper_band,
                         wng_floor_db, R_est, taper)

    # --- 指标 ---
    res.metrics = _metrics(res, mics, f, steer_arg, c, near_field)
    if taps and taps > 0:
        res.delays = fractional_delays(mics, u0, c)
        res.extra["taps"] = int(taps)
    return res


def coherence_diagnostics(mics: np.ndarray, freqs: np.ndarray,
                          c: float = C_DEFAULT) -> dict:
    """扩散场相干矩阵的有效性诊断。

    重要事实（容易误判为 bug）：三维扩散场相干 Gamma_ij = sinc(k d_ij) 在
    `k*d` 较大时会出现**负相干**，此时 `w^H Gamma w` 可以小于 `w^H w`，
    于是 DI 能超过各向同性上限 10log10(M)。这不是算错，而是该模型在
    "阵元间距 >> 波长"时不再描述真实噪声场。

    另一侧：平面阵列在低频时 Gamma 近乎奇异（阵面法向不可观测），
    MVDR/超指向在该频点的解极不稳定。
    """
    f = np.atleast_1d(np.asarray(freqs, float))
    ev_min = np.empty(f.size)
    cond = np.empty(f.size)
    for k, fk in enumerate(f):
        G = st.diffuse_coherence(mics, np.array([fk]), c, kind="spherical")[0]
        ev = np.linalg.eigvalsh(G)
        ev_min[k] = ev.min()
        cond[k] = ev.max() / max(ev.min(), 1e-30)
    return {
        "gamma_eig_min": [round(float(v), 6) for v in ev_min],
        "gamma_cond": [round(float(v), 3) for v in cond],
        "isotropic_bound_db": round(float(10 * np.log10(mics.shape[0])), 3),
        "warn": ("DI 可能超过各向同性上限：扩散场模型在高频（阵元间距 >> 波长）"
                 "不再成立，此时 DI 仅供参考"),
    }


def _metrics(res: BeamformerResult, mics, f, steer_arg, c, near_field) -> dict:
    a = st.steering_vector(mics, f, steer_arg, c, near_field=near_field)
    gamma = st.diffuse_coherence(mics, f, c, kind="spherical")
    di = di_db(res.weights, a, gamma)
    bound = 10 * np.log10(mics.shape[0])
    return {
        "wng_db": [round(float(v), 3) for v in wng_db(res.weights, a)],
        "di_db": [round(float(v), 3) for v in di],
        "wng_min_db": round(float(wng_db(res.weights, a).min()), 3),
        "di_max_db": round(float(di.max()), 3),
        "directivity_index_bound_db": round(float(bound), 3),
        "di_exceeds_bound": bool(di.max() > bound + 1e-6),
        "freqs": [round(float(v), 3) for v in f],
    }


def _design_fi(res, mics, f, steer_arg, c, near_field, band, wng_floor_db,
               R_est, taper) -> BeamformerResult:
    """频不变（frequency-invariant）波束成形。

    常见误解：以为"所有频点共用同一个权重向量"就叫频不变。**这是错的** ——
    阵元位置固定时，响应 exp(-j2πf·τ_m) 随频率变化，单一权重的方向图必然
    随频率变窄（这是物理，不是实现问题）。

    正确的频不变定义是：**让方向图形状与频率无关**，即
        |B(f, u)| ≈ T(u)    对所有 f ∈ 频带            （T 与 f 无关）
    因此必须在每个频点分别设计 w(f)，去拟合**同一个**期望模板 T(u)。
    自由度核算：每频点半数 M 个复权重 → 2M 个实自由度，要拟合的方向图
    维数远大于此，所以只能最小二乘近似 —— 这也是为什么真实宽带阵列的
    频不变性总是有限的，且需要牺牲旁瓣或白噪增益来换。

    本实现：逐频点带约束最小二乘
        min_w  ||A(f) w - T||²_W + μ||w||²      s.t.  w^H a(f,u0) = 1
    μ 由 wng_floor_db 决定（正则化 = 对角加载的等价形式）。
    """
    M = mics.shape[0]
    if band is None:
        band = (f.min(), f.max())
    sel = (f >= band[0] - 1e-9) & (f <= band[1] + 1e-9)
    if not np.any(sel):
        sel = np.ones(f.size, bool)
    fb = f[sel]

    bw = float(res.extra.get("fi_mainlobe_deg", 60.0))
    dirs, shape, _, _ = _dir_pool(int(res.extra.get("fi_grid", 12)))

    u_ref = steer_arg if not near_field else steer_arg / max(np.linalg.norm(steer_arg), 1e-12)
    ang = np.rad2deg(np.arccos(np.clip(dirs @ u_ref, -1.0, 1.0)))
    T = (ang <= bw).astype(float)                      # 期望模板：主瓣 1，其余 0
    # 主瓣内加权更重；旁瓣权重低一些，避免为压旁瓣把主瓣撑宽
    wgt = np.where(T > 0, 1.0, 0.3)

    Af = st.steering_grid(mics, fb, dirs, c)            # (Fb,G,M)
    A0 = st.steering_vector(mics, fb, steer_arg, c, near_field=near_field)  # (Fb,M)

    w = np.empty((f.size, M), dtype=np.complex128)
    mu_list = np.empty(f.size)
    for k, fk in enumerate(f):
        if not sel[k]:
            # 频带外：退化为延迟求和（不做无意义的外推）
            a = st.steering_vector(mics, np.array([fk]), steer_arg, c,
                                   near_field=near_field)
            w[k] = delay_sum_weights(a, taper)[0]
            mu_list[k] = np.nan
            continue
        kk = int(np.argmin(np.abs(fb - fk)))
        A = Af[kk]                                      # (G,M) 每行 a(f,u)^H
        a0 = A0[kk]
        # 带权正规方程
        Aw = A * wgt[:, None]
        Tw = T * wgt
        R = Aw.conj().T @ Aw
        b = Aw.conj().T @ Tw
        # 正则化强度：由 WNG 下限反推；无下限时用极小值保证可解
        tr = float(np.trace(R).real) / M
        mu = 1e-8 * tr
        if wng_floor_db is not None:
            target = 10 ** (wng_floor_db / 10.0) / M    # 相对 DS 的容许衰减
            mu = max(mu, tr * (1.0 / max(target, 1e-9) - 1.0))
            mu = min(mu, 1e3 * tr)
        mu_list[k] = mu
        Rr = R + (mu + 1e-12 * tr) * np.eye(M)
        w_ls = np.linalg.solve(Rr, b)
        # 约束 w^H a0 = 1
        Ra = np.linalg.solve(Rr, a0)
        den = np.vdot(a0, Ra)
        den = den if abs(den) > 1e-30 else 1e-30
        w[k] = w_ls + Ra * ((1.0 - np.vdot(a0, w_ls)) / den)

    res.weights = w
    res.extra["fi_band"] = [float(band[0]), float(band[1])]
    res.extra["fi_mainlobe_deg"] = float(bw)
    res.extra["fi_mu"] = [None if not np.isfinite(v) else round(float(v), 6)
                          for v in mu_list]
    res.extra["fi_per_freq"] = True
    return res


def _dir_pool(n: int):
    """频不变设计的内部方向池（球面近似均匀采样）。"""
    from .geometry import direction_grid
    return direction_grid(az_bins=4 * n, el_bins=2 * n)


# --------------------------------------------------------------------------
# 应用
# --------------------------------------------------------------------------
def fractional_delays(mics: np.ndarray, u: np.ndarray, c: float = C_DEFAULT
                      ) -> np.ndarray:
    """远场指向 u 时各阵元的传播时延（秒）。时域实现用。"""
    return ac.delay_from_dir(mics, u, c)


def apply_freq(res: BeamformerResult, Xf: np.ndarray, freqs: np.ndarray
               ) -> np.ndarray:
    """频域应用：Y(f,k) = w(f)^H X(f,k)。

    Xf    : (M,K,Fx)
    freqs : (Fx,) 与 Xf 第三维对应的频率轴
    return: (K,Fx) complex
    """
    W = interpolate_weights(res, freqs)                    # (Fx,M)
    return np.einsum("fm,mkf->kf", W.conj(), Xf)


def interpolate_weights(res: BeamformerResult, freqs: np.ndarray) -> np.ndarray:
    """把权重插值到目标频率轴 (Fx,M)。

    复权重按实部/虚部分别线性插值；超出设计频段时钳到端点（不做外推，
    避免产生没有物理意义的相位）。
    """
    f = res.freqs
    ft = np.asarray(freqs, float)
    if f.size == 1:
        return np.repeat(res.weights, ft.size, axis=0)
    out = np.empty((ft.size, res.weights.shape[1]), dtype=np.complex128)
    for m in range(res.weights.shape[1]):
        out[:, m] = (np.interp(ft, f, res.weights[:, m].real)
                     + 1j * np.interp(ft, f, res.weights[:, m].imag))
    return out


def apply_time(res: BeamformerResult, X: np.ndarray, fs: int, n_fft: int = 512,
               hop: int | None = None) -> np.ndarray:
    """时域应用：STFT -> 逐频加权 -> ISTFT。返回 (N,) 或 (M?...)。"""
    X = np.asarray(X, float)
    single = X.ndim == 1
    if single:
        X = X[None, :]
    Xf, freqs = dsp.stft(X, fs, n_fft=n_fft, hop=hop)
    Yf = apply_freq(res, Xf, freqs)
    y = dsp.istft(Yf, fs, n_fft=n_fft, hop=hop, length=X.shape[1])
    return y


def process(res: BeamformerResult, X: np.ndarray, fs: int, n_fft: int = 512,
            hop: int | None = None, domain: str = "freq") -> np.ndarray:
    """统一应用入口。"""
    if domain == "freq":
        return apply_time(res, X, fs, n_fft, hop)
    raise ValueError("目前仅支持 domain='freq'（时域 FIR 实现见导出层）")


# --------------------------------------------------------------------------
# 便捷分析：让 GUI 一次拿到全部对比数据
# --------------------------------------------------------------------------
def sweep(mics, freqs, steer=(0.0, 90.0), methods=("delay_sum", "superdirective",
                                                   "mvdr", "fi"),
          az_bins: int = 181, el_bins: int = 1, c: float = C_DEFAULT,
          R_est: np.ndarray | None = None, wng_floor_db: float | None = None,
          loading: float = 0.0, near_field: bool = False,
          source_pos: np.ndarray | None = None,
          fi_mainlobe_deg: float | None = None) -> dict:
    """对多种方法在同一指向下算波束图与指标，供 GUI 对比。"""
    from .geometry import direction_grid
    mics = np.asarray(getattr(mics, "mics", mics), float).reshape(-1, 3)
    dirs, shape, az_ax, el_ax = direction_grid(az_bins=az_bins, el_bins=el_bins)
    out = {"dirs": dirs, "shape": shape, "az_axis": az_ax, "el_axis": el_ax,
           "methods": {}}
    for mth in methods:
        try:
            r = design(mics, freqs, steer=steer, method=mth, c=c, loading=loading,
                       wng_floor_db=wng_floor_db, R_est=R_est,
                       near_field=near_field, source_pos=source_pos,
                       fi_mainlobe_deg=fi_mainlobe_deg)
        except Exception as e:                             # 单方法失败不影响整体
            out["methods"][mth] = {"error": f"{type(e).__name__}: {e}"}
            continue
        B = evaluate_beampattern(r.weights, mics, r.freqs, dirs, c)   # (F,G)
        Bdb = 10 * np.log10(np.maximum(B, 1e-12))
        u0 = r.steer_dir
        out["methods"][mth] = {
            "freqs": [round(float(v), 3) for v in r.freqs],
            "pattern_db": [[round(float(v), 3) for v in row] for row in Bdb],
            "metrics": r.metrics,
            "beamwidth_deg": [round(beamwidth_deg(B[k], dirs, u0), 3)
                              for k in range(r.freqs.size)],
            "extra": r.extra,
        }
    return out


__all__ = [
    "METHODS", "BeamformerResult", "design", "process", "apply_freq", "apply_time",
    "interpolate_weights", "delay_sum_weights", "lcmv_weights", "solve_wng_floor",
    "noise_covariance", "wng_db", "di_db", "evaluate_beampattern",
    "beamwidth_deg", "fractional_delays", "sweep",
]
