"""声源定位：SRP-PHAT、MUSIC、TDOA 非线性最小二乘、近场距离估计。

约定（唯一来源 docs/CONTRACT.md，本文件不重新定义任何约定）：

* 第 1、2 节 —— 全部在**阵列坐标系**里：阵元在 xy 平面、boresight = +z、
  `el` 从 xy 平面量向 +z。角度/方向互转一律走
  `acoustics.unit_dir / dirs_from_angles / angles_from_dir`，
  本文件不出现任何手写的三角函数角度公式。
* 第 3 节 —— 导向矢量锚点。远场直接复用 `steering.steering_grid`（它实现了契约
  里的 `a_m = exp(-j2πf/c·dot(p_m,u))`）；近场必须用契约的球面波式子，
  见 `_near_field_manifold`（与 `steering.steering_vector` 同式，测试里有 1e-12 一致性校验）。
* 第 5 节 —— 频域一律走 `dsp.stft()`：rfft 单边谱、Blackman-Harris、不做幅度补偿。
* 第 8 节 —— 热力图统一 **扁平数组 + shape**，且不含 NaN/Inf。

设计上的三条主线（为什么这么写）：

1. **协方差 + 二次型**：SRP 与 MUSIC 的扫描都可以写成
   `P(g) = Σ_f a(f,g)^H · C(f) · a(f,g)` 的形式。
   SRP 取 `C = R`（阵列协方差），MUSIC 取 `C = EₙEₙᴴ`（噪声子空间投影）。
   于是两者共用同一套"分频块 + 批量矩阵乘"的向量化内核，
   既省一次 STFT，又让"峰值局部细化"能在任意方向上低成本重算代价函数
   （细化只需要 9 个方向的导向矢量，而不是整张网格）。

2. **分频块（chunk）**：G=2664 的默认网格下，一次性物化 (F,G,M) 复数导向矩阵
   需要 100+ MB。按频率分块后峰值内存降到 ~20 MB，且每次都是 BLAS 的
   批量 matmul，速度反而更好（见 `_quadratic_map` 注释）。

3. **全程确定性**：没有任何随机数；粗搜索用固定网格、细化用固定步长的
   模式搜索 / LM，初值也只由数据决定。后续与 C++ 逐位比对时不需要除随机种子。

数值稳定性：所有除法都加 `_EPS`；所有进入 map 的数组都做一次 `nan_to_num`
并 clip 到非负；特征分解前把协方差 Hermitian 化（`eigh` 只读下三角，
不化等于静默丢掉一半数据）；矩阵元素的模方一律显式写 `conj`
（`np.einsum` **不会**自动共轭，漏了会得到可正可负的"能量"）。

两个**物理限制**（不是实现缺陷，全模块用同一套约定处理，见各函数注释）：

* **平面阵前后模糊**：镜像方向 `(az, -el)` 的阵元距离逐点相等，
  观测里没有分辨它所需的信息。默认 `hemisphere='auto'`：阵元共面时只在
  `el>=0`（CONTRACT 第 2 节的 boresight 正面）取峰，并在返回值里标
  `front_back_ambiguous=True`；`map` 仍完整保留镜像瓣。
* **距离可观测性**：近场距离由波前曲率 `~a²/(2r)` 决定，声源越接近
  阵列轴线（`|el|→90°`）曲率在阵元间越接近常数，距离越不可观测；
  测试 `test_near_field_observability` 把它固化成了可回归的事实。
"""
from __future__ import annotations

import numpy as np

from . import acoustics as ac
from . import dsp
from . import geometry as geo
from . import steering as st
from . import tdoa as td

C_DEFAULT = ac.C_DEFAULT
_EPS = 1e-12

__all__ = [
    "srp_phat", "music_spectrum", "tdoa_estimate", "localize",
    "near_field_distance", "select_freq_bins",
]


# ==========================================================================
# 通用小工具
# ==========================================================================
def _as_signal(X) -> np.ndarray:
    """(M,N) float64 观测。1-D 视作单通道（定位本身要求 M>=2，由各函数自行校验）。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("X must be (M,N) or (N,)")
    return X


def _as_mics(mics) -> np.ndarray:
    m = np.asarray(mics, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError("mics must be (M,3)")
    return m


def _obs(X, mics):
    """一次拿到 (X (M,N), mics (M,3)) 并校验通道数一致。

    通道数对不上时必须报错而不是"静默算出一个结果"：早先版本里
    `tdoa_estimate(X[0], fs, mics_16)` 会拿 1 个通道去配 16 个阵元的几何，
    pairs 为空、taus 为空，最后返回一堆 NaN 而不报错。
    """
    Xs = _as_signal(X)
    ms = _as_mics(mics)
    if ms.shape[0] != Xs.shape[0]:
        raise ValueError(
            f"mics 有 {ms.shape[0]} 个阵元，但 X 有 {Xs.shape[0]} 个通道，必须一致")
    return Xs, ms


def _axis_step(ax, default: float = 5.0) -> float:
    """网格轴步长（度）。没有轴信息时给一个保守缺省值。"""
    if ax is None:
        return float(default)
    a = np.asarray(ax, dtype=np.float64).ravel()
    if a.size < 2:
        return float(default)
    d = np.abs(np.diff(a))
    d = d[d > 1e-9]
    return float(np.median(d)) if d.size else float(default)


def _parse_grid(grid_dirs, az_bins=72, el_bins=37,
                az_range=(-180.0, 180.0), el_range=(-90.0, 90.0),
                grid_shape=None, az_axis=None, el_axis=None):
    """统一解析方向网格，返回 (dirs (G,3), shape, az_axis, el_axis)。

    接受四种写法（都是为了兼容 GUI / 测试 / 手工网格）：
      * None                      -> geometry.direction_grid() 的默认网格
      * geometry.direction_grid() 返回的四元组
      * dict(dirs=..., shape=..., az_axis=..., el_axis=...)
      * (G,3) 数组（此时 shape/轴信息靠 kwargs 补，缺省退化为 (G,)）
    `dirs` 一律重新归一化 —— 调用方给非单位向量时不能让波束指向悄悄偏掉。
    """
    if grid_dirs is None:
        dirs, shape, az, el = geo.direction_grid(az_range, el_range,
                                                 az_bins, el_bins)
    elif isinstance(grid_dirs, dict):
        dirs = np.asarray(grid_dirs["dirs"], dtype=np.float64)
        shape = tuple(grid_dirs.get("shape", (dirs.shape[0],)))
        az = grid_dirs.get("az_axis", az_axis)
        el = grid_dirs.get("el_axis", el_axis)
    elif isinstance(grid_dirs, (tuple, list)) and len(grid_dirs) == 4:
        dirs, shape, az, el = grid_dirs
        dirs = np.asarray(dirs, dtype=np.float64)
        shape = tuple(np.asarray(shape, dtype=int).ravel())
        az = np.asarray(az, dtype=np.float64).ravel()
        el = np.asarray(el, dtype=np.float64).ravel()
    else:
        dirs = np.asarray(grid_dirs, dtype=np.float64)
        shape = tuple(grid_shape) if grid_shape is not None else (dirs.shape[0],)
        az, el = az_axis, el_axis

    dirs = dirs.reshape(-1, 3)
    n = np.linalg.norm(dirs, axis=1)
    n = np.where(n > 0, n, 1.0)
    dirs = dirs / n[:, None]
    if int(np.prod(shape)) != dirs.shape[0]:
        # 网格自洽性检查：shape 与点数不符时宁可退化也不要越界索引
        shape = (dirs.shape[0],)
    return dirs, shape, az, el


def select_freq_bins(fs, n_fft, mics=None, c=C_DEFAULT, freqs=None,
                     freq_band=None, f_lo=300.0, f_hi=None):
    """挑选参与定位的频点（返回 `(bins, freqs_hz)`）。

    规则（README 化，避免歧义）：
      * `freqs=None, freq_band=None` -> 自动频带：`[300 Hz, min(0.45·fs, c/(2·d_min))]`。
        上界取**无空间混叠上限** `c/(2·d_min)`：超过它的频点会产生栅瓣，
        对 SRP/MUSIC 只会添乱，物理上也拿不到更多信息。
      * `freqs` 为长度 2 的序列 -> 解释为**频带** `(lo, hi)` Hz（最常用写法）。
      * 其它长度的 `freqs` -> 解释为**离散频点** Hz，就近吸附到 rfft 频点。
      * `freq_band=(lo,hi)` 显式指定频带，优先级最高。
    永远排除 DC（PHAT 在 f=0 处会 0/0）。
    """
    fs = float(fs)
    fa = np.fft.rfftfreq(int(n_fft), d=1.0 / fs)
    F = fa.size
    df = fs / float(n_fft)

    if freq_band is None and freqs is not None:
        arr = np.atleast_1d(np.asarray(freqs, dtype=np.float64)).ravel()
        if arr.size == 2 and arr[1] > arr[0]:
            freq_band = (float(arr[0]), float(arr[1]))
        else:
            bins = np.rint(arr / df).astype(int)
            bins = np.unique(np.clip(bins, 1, F - 1))
            if bins.size == 0:
                raise ValueError("no usable frequency bin selected")
            return bins, fa[bins]

    if freq_band is not None:
        lo, hi = float(freq_band[0]), float(freq_band[1])
    else:
        lo = max(float(f_lo), 3.0 * df)
        hi = 0.45 * fs if f_hi is None else float(f_hi)
        if mics is not None:
            m = np.asarray(mics, dtype=np.float64)
            if m.shape[0] > 1:
                dmin = geo.min_spacing(m)
                if dmin > 0:
                    # 留 5% 余量：栅瓣刚出现时旁瓣已经抬头了
                    hi = min(hi, 0.95 * ac.alias_limit(dmin, c))
        hi = min(hi, 0.45 * fs)
        if hi <= lo:
            # 极小阵列 / 极低采样率的退路：保证至少有可用带宽
            hi = min(0.45 * fs, max(lo * 2.0, 2000.0))

    bins = np.nonzero((fa >= lo - 1e-9) & (fa <= hi + 1e-9))[0]
    bins = bins[bins >= 1]
    if bins.size < 3:
        raise ValueError(
            f"frequency band [{lo}, {hi}] Hz selects {bins.size} bins; "
            "请放宽 freq_band / 提高 n_fft")
    return bins, fa[bins]


# --------------------------------------------------------------------------
# 导向矢量（分块）与二次型扫描内核
# --------------------------------------------------------------------------
def _near_field_manifold(mics, freqs, pts, c=C_DEFAULT):
    """近场球面波导向矢量 (F,G,M)，**逐字复刻 CONTRACT 第 3 节**：

        a_m(f,p) = (‖p‖ / ‖p_m - p‖) · exp(-j·2πf/c·(‖p_m - p‖ - ‖p‖))

    这里只是把它向量化到 (G,) 个候选点上（steering.steering_vector 一次只算一个点，
    G=2664 时逐个调用太慢）。测试里用 `steering.steering_vector` 做了 1e-12 一致性校验。
    """
    mics = np.asarray(mics, dtype=np.float64)
    f = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)                                   # (G,)
    d = np.linalg.norm(mics[None, :, :] - pts[:, None, :], axis=2)    # (G,M)
    amp = r[:, None] / np.maximum(d, _EPS)
    tau = (d - r[:, None]) / c
    phase = -2.0 * np.pi * f[:, None, None] * tau[None, :, :]
    return amp[None, :, :] * np.exp(1j * phase)


def _manifold_block(mics, f_block, dirs, c, src_dist=None):
    """(f,G,M) 导向矢量块。远场走 steering.steering_grid（契约锚点）。"""
    if src_dist is None:
        return st.steering_grid(mics, f_block, dirs, c)
    return _near_field_manifold(mics, f_block, dirs * float(src_dist), c)


def _quadratic_map(C, mics, fsel, dirs, c, chunk=8, src_dist=None):
    """P(g) = Σ_f a(f,g)ᴴ C(f) a(f,g)，按频率分块做批量 matmul。

    为什么按频率分块：一次算完全部频率需要 (F,G,M) 复数矩阵，
    G=2664/F=119/M=32 时是 162 MB；分块后每块 (8,2664,32)=10 MB，
    且 `np.matmul` 是批量 BLAS 调用，循环开销可忽略。
    """
    mics = np.asarray(mics, dtype=np.float64)
    P = np.zeros(dirs.shape[0], dtype=np.float64)
    F = fsel.size
    for s in range(0, F, int(chunk)):
        e = min(s + int(chunk), F)
        A = _manifold_block(mics, fsel[s:e], dirs, c, src_dist)   # (f,G,M)
        # a^H C a：先 A^H·C 得 (f,G,M)，再与 A 逐元素共轭相乘求和
        T = np.matmul(np.conj(A), C[s:e])                         # (f,G,M)
        P += np.einsum("fgm,fgm->g", T, A).real
    return P


# --------------------------------------------------------------------------
# 峰值提取与局部细化
# --------------------------------------------------------------------------
def _refine_direction(cost_fn, az0, el0, step_az, step_el, iters=5, shrink=0.5,
                      el_bounds=(-90.0, 90.0), exclude=(), sep_deg=0.0):
    """在网格峰附近用**收缩模式搜索**细化到亚网格精度。

    为什么必须做：5°×5° 网格的量化误差本身就有 ~3.5°，
    直接把网格索引当答案是不可能满足 "<3°" 这类指标的。
    代价函数本身是光滑的，9 点模式搜索 + 步长折半在 5 次迭代后
    步长降到 0.16°，足以把量化误差压到可忽略。

    `el_bounds` 用于"只在前半球找峰"时把细化限制在同一半球内：
    否则模式搜索可能顺着完全对称的镜像瓣翻到背面去。

    `exclude` + `sep_deg`：多声源时把代价函数在"已接受方向"的邻域内封死。
    没有这一步，第二个初值即使离第一个 8°，也会顺着宽主瓣爬回**同一个**
    全局最大点，最后输出两个完全重合的"源"。
    """
    az, el = float(az0), float(el0)
    sa, se = float(step_az), float(step_el)
    lo_e, hi_e = float(el_bounds[0]), float(el_bounds[1])
    ex = [(ac.unit_dir(a, e), float(np.cos(np.deg2rad(sep_deg))))
          for a, e in exclude] if sep_deg > 0 else []

    def _eval(a, e):
        v = np.asarray(cost_fn(ac.dirs_from_angles(np.array([a]), np.array([e]))),
                       dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        return float(v[0]) if v.size else -np.inf

    def _cost(dd):
        vals = np.asarray(cost_fn(dd), dtype=np.float64).ravel()
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        for u_ex, c_ex in ex:
            # cos 大于阈值 <=> 夹角小于 sep，即落在已接受峰的邻域内 -> 封死
            vals = np.where(dd @ u_ex > c_ex, -np.inf, vals)
        return vals

    best = _eval(az, el)
    for _ in range(int(iters)):
        da = np.array([-1.0, 0.0, 1.0]) * sa
        de = np.array([-1.0, 0.0, 1.0]) * se
        AZ, EL = np.meshgrid(az + da, np.clip(el + de, lo_e, hi_e),
                             indexing="xy")
        cand = ac.dirs_from_angles(AZ.ravel(), EL.ravel())
        vals = _cost(cand)
        k = int(np.argmax(vals))
        if np.isfinite(vals[k]) and (not np.isfinite(best)
                                     or vals[k] > best * (1.0 + 1e-12)):
            best = float(vals[k])
            az, el = float(AZ.ravel()[k]), float(EL.ravel()[k])
        sa *= shrink
        se *= shrink
        if sa < 1e-4 and se < 1e-4:
            break
    # 用 angles_from_dir 收口：保证 az∈[-180,180]、el∈[-90,90] 且极点不出现 nan
    return ac.angles_from_dir(ac.unit_dir(az, el)) + (best,)


def _beamwidth_nms_deg(mics, fsel, c, az_step, el_step, frac=1.0,
                       max_deg=45.0):
    """缺省的角度屏蔽半径（度）= `frac × 主瓣宽度`，下限 1.5 个网格步长。

    为什么不能只用"1.5 个网格步长"：主瓣宽度由 `≈0.886·λ/D`
    （acoustics.angular_resolution）决定，实测 vogel-32 上达 ~28°，
    比 5° 网格大得多。5° 的屏蔽半径挡不住同一主瓣上的"假峰"，
    多声源时会输出一堆聚在真峰旁边的伪源。

    `frac` 按方法取不同值：
      * SRP-PHAT 是常规波束形成，实测主瓣在 el 方向比 az 方向更宽，
        取 1.5 倍 -3dB 主瓣宽度做屏蔽半径才不会把肩部当第二个源；
      * MUSIC 是超分辨，能分辨远小于主瓣宽度的两个源 -> frac≈0.3。
    """
    d = 1.5 * max(az_step, el_step, 1e-6)
    try:
        ap = geo.aperture(np.asarray(mics, float))
        f_mid = float(np.median(np.asarray(fsel, float)))
        bw = ac.angular_resolution(ap, f_mid, c)
        if np.isfinite(bw) and bw > 0:
            d = max(d, float(frac) * bw)
    except Exception:                                   # 网格/几何异常时退回保守值
        pass
    return float(min(d, max_deg))


def _resolve_hemisphere(hemisphere, mics):
    """解析"前/后半球"约定，返回 `(mode, planar, ambiguous)`。

    **为什么需要它（重要，不是偷懒）**：阵元全在 xy 平面时，对任意方向 u，
    镜像方向 `(az, -el)` 给出的阵元距离**逐点完全相等**
    （`‖p_m - p‖² = ‖p_xy - p_m‖² + p_z²`，镜像只翻转 p_z），
    于是两个假设产生**逐位相同**的观测。这是平面阵的固有前后模糊，
    数据里根本没有分辨它所需的信息。

    所以必须用约定来消歧，本模块的约定与 CONTRACT 第 2 节一致：
    `el=+90` 是 boresight（正面），`el<0` 是"阵列正后方，用于检查后向抑制"。
    对平面阵 `hemisphere='auto'` 默认取 `'front'`（el>=0），
    并在返回值里标记 `front_back_ambiguous=True`；
    想要完整球面搜索就显式传 `hemisphere='both'`。
    """
    m = np.asarray(mics, dtype=np.float64)
    planar = bool(np.all(np.abs(m[:, 2]) < 1e-9))
    h = "auto" if hemisphere is None else str(hemisphere).lower()
    if h in ("auto",):
        return ("front" if planar else "both"), planar, planar
    if h in ("both", "all", "none", "full"):
        return "both", planar, False
    if h in ("front", "+z", "up", "upper", "positive"):
        return "front", planar, planar
    if h in ("back", "-z", "down", "lower", "rear", "negative"):
        return "back", planar, planar
    raise ValueError(f"unknown hemisphere: {hemisphere!r}")


def _finalize(P, dirs, shape, az_ax, el_ax, n_sources, min_sep_deg,
              az_step, el_step, refine, cost_fn, refine_iters,
              hemisphere="auto", mics=None, sep_deg=None):
    """map 归一化、无 NaN/Inf 兜底、取峰 + 细化。返回统一的结果字段。

    注意：`hemisphere` 只影响**取峰**（哪些格子可以当答案），
    不影响 `map` —— 镜像瓣在图上照旧可见，用户能一眼看出前后模糊。
    """
    P = np.asarray(P, dtype=np.float64).ravel()
    # nan_to_num 而不是 raise：一个坏频点不该让整张图报废，但也不能把 NaN 传出去
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.maximum(P, 0.0)
    mx = float(P.max()) if P.size else 0.0
    if not np.isfinite(mx) or mx <= 0.0:
        mx = 1.0
    norm = P / mx

    if min_sep_deg is None:
        # 缺省：主瓣半宽（见 _beamwidth_nms_deg），至少 1.5 个网格步长
        min_sep_deg = (float(sep_deg) if sep_deg is not None
                       else 1.5 * max(az_step, el_step, 1e-6))

    hemi, planar, ambiguous = _resolve_hemisphere(
        hemisphere, np.zeros((1, 3)) if mics is None else mics)
    if mics is None:
        hemi, ambiguous = "both", False
    if hemi == "front":
        mask = dirs[:, 2] >= -1e-12
        el_bounds = (0.0, 90.0)
    elif hemi == "back":
        mask = dirs[:, 2] <= 1e-12
        el_bounds = (-90.0, 0.0)
    else:
        mask = np.ones(dirs.shape[0], bool)
        el_bounds = (-90.0, 90.0)
    if not np.any(mask):                      # 网格与半球不相交时不要静默出错
        mask = np.ones(dirs.shape[0], bool)
        hemi, el_bounds = "both", (-90.0, 90.0)

    # 逐个取峰 + **屏蔽（deflation）**：先取网格最大点、细化，再把"细化后方向"
    # 附近的网格全部屏蔽，然后取下一个。必须先屏蔽再细化下一个 ——
    # 否则两个相隔一个网格步长的初值会各自爬山爬到同一个全局最大点上，
    # 输出两个几乎重合的"源"（多声源时必然踩到）。
    cos_sep = float(np.cos(np.deg2rad(max(min_sep_deg, 1e-6))))
    masked = mask.copy()
    accepted = []
    for _ in range(int(n_sources)):
        cand = np.where(masked, norm, -np.inf)
        k = int(np.argmax(cand))
        if not np.isfinite(cand[k]):
            break
        az_k, el_k = ac.angles_from_dir(dirs[k])
        if refine:
            az_k, el_k, v_k = _refine_direction(
                cost_fn, az_k, el_k, az_step, el_step, iters=refine_iters,
                el_bounds=el_bounds,
                exclude=[(a, e) for a, e, _, _ in accepted],
                sep_deg=min_sep_deg)
            # 细化用的是连续代价函数，取值可能比网格最大值高一点点（<1%），
            # 这里统一除以网格尺度 mx，量纲与 map 一致
            v_k = float(v_k) / mx if mx > 0 else float(v_k)
        else:
            v_k = float(norm[k])
        accepted.append((float(az_k), float(el_k), float(v_k), k))
        masked &= (dirs @ ac.unit_dir(az_k, el_k)) < cos_sep
        masked &= (dirs @ dirs[k]) < cos_sep
        masked[k] = False

    accepted.sort(key=lambda t: -t[2])
    idx = [k for _, _, _, k in accepted]
    dirs_out = [[round(az, 6), round(el, 6)] for az, el, _, _ in accepted]
    vals = [float(v) for _, _, v, _ in accepted]

    return {
        "map": norm,
        "map_scale": float(mx),
        "shape": tuple(int(v) for v in shape),
        "az_axis": None if az_ax is None else np.asarray(az_ax, float),
        "el_axis": None if el_ax is None else np.asarray(el_ax, float),
        "est_dirs": dirs_out,
        "est_idx": [int(i) for i in idx],
        "peak_vals": [float(v) for v in vals],
        "n_sources": int(len(idx)),
        "hemisphere": hemi,
        "planar_array": bool(planar),
        "front_back_ambiguous": bool(ambiguous),
    }


# ==========================================================================
# 1. SRP-PHAT
# ==========================================================================
def srp_phat(X, fs, mics, grid_dirs=None, c=C_DEFAULT, n_fft=512, hop=None,
             freqs=None, weights="phat", *, freq_band=None, beta=1.0,
             n_sources=1, min_sep_deg=None, refine=True, refine_iters=5,
             win="blackmanharris", win_table=None, src_dist=None,
             az_bins=72, el_bins=37, az_range=(-180.0, 180.0),
             el_range=(-90.0, 90.0), grid_shape=None, az_axis=None,
             el_axis=None, chunk=8, hemisphere="auto", nms_frac=1.5):
    """导向响应功率 SRP-PHAT（steered response power with phase transform）。

    代价函数（与 C++ 导出式一致）：

        P(g) = Σ_f Σ_k | Σ_m conj(a_m(f,u_g)) · Z_m(k,f) |²  / (M²·K·F)

    其中 `Z = Xf / (|Xf|^beta + eps)` 是 PHAT 加权后的单边谱
    （beta=1 为标准 PHAT：只留相位；beta=0 退化为普通 SRP）。
    实现上用恒等式 `Σ_k |aᴴz_k|² = aᴴ R a`（`R` 为阵列协方差）走二次型内核，
    数学等价、少一半乘加，而且让"峰值细化"能对任意方向低成本重算。

    参数
    ----
    X        : (M,N) 时域观测，或 (N,) 单通道
    fs       : 采样率
    mics     : (M,3) 阵元坐标（阵列坐标系，xy 平面）
    grid_dirs: 方向网格，见 `_parse_grid`；None 用 `geometry.direction_grid()`
    freqs    : None(自动频带) | 长度 2 的 (lo,hi) 频带 | 离散频点 Hz
    weights  : 'phat' 只留相位 | 'none' 原始幅度（普通 SRP）| 可调用对象
    src_dist : 非 None 时用**近场球面波**导向（候选点 = dirs·src_dist），
               用于已知距离的近场扫描；None 为远场平面波
    n_sources: 输出的峰值个数（取峰时做角度屏蔽，见 nms_frac）
    hemisphere: 'auto'|'front'|'back'|'both'，平面阵前后模糊的消歧约定，
               见 `_resolve_hemisphere`（只影响取峰，不影响 map）
    nms_frac : 取峰屏蔽半径 = nms_frac × 主瓣宽度（`_beamwidth_nms_deg`）。
               常规波束形成的分辨极限就是主瓣，SRP 默认 1.5
    refine   : 是否把网格峰细化到亚网格精度（强烈建议 True）

    返回 dict（CONTRACT 第 8 节风格）：
        map (G,) 已归一化到 max=1 且无 NaN/Inf、shape、az_axis、el_axis、
        est_dirs [[az,el],...]、est_idx、peak_vals、map_scale，以及
        freqs / bins / n_frames / hemisphere / front_back_ambiguous 等诊断字段。
    """
    X, mics = _obs(X, mics)
    M, N = X.shape
    if M < 2:
        raise ValueError("SRP-PHAT 至少需要 2 个阵元")
    fs = float(fs)

    dirs, shape, az_ax, el_ax = _parse_grid(
        grid_dirs, az_bins, el_bins, az_range, el_range,
        grid_shape, az_axis, el_axis)

    Xf, _fa = dsp.stft(X, fs, n_fft=n_fft, hop=hop, win=win,
                       win_table=win_table)                      # (M,K,F)
    bins, fsel = select_freq_bins(fs, n_fft, mics, c, freqs, freq_band)
    K = Xf.shape[1]

    Z = np.asarray(Xf[:, :, bins], dtype=np.complex128)           # (M,K,Fsel)
    if callable(weights):
        Z = Z * np.asarray(weights(Z), dtype=np.complex128)
    elif weights == "phat":
        # 只保留相位：|Z|=1。eps 防 0/0（静音帧/空频点）
        Z = Z / (np.abs(Z) ** float(beta) + _EPS)
    elif weights in ("none", "raw", None):
        pass
    else:
        raise ValueError(f"unknown weights: {weights!r}")

    # 协方差 R(f)（契约第 5 节：对帧取平均）。Hermitian 化保证 eigh/二次型稳定。
    R = np.einsum("mkf,nkf->fmn", Z, np.conj(Z)) / float(max(K, 1))
    R = 0.5 * (R + np.conj(np.transpose(R, (0, 2, 1))))

    def cost(dd):
        # 归一化到 (M²·Fsel)：PHAT 输入下量纲上是"归一化波束输出的平均功率"
        return _quadratic_map(R, mics, fsel, dd, c, chunk=chunk,
                              src_dist=src_dist) / (M * M * fsel.size)

    P = cost(dirs)
    out = _finalize(P, dirs, shape, az_ax, el_ax, n_sources, min_sep_deg,
                    _axis_step(az_ax), _axis_step(el_ax), refine, cost,
                    refine_iters, hemisphere=hemisphere, mics=mics,
                    sep_deg=_beamwidth_nms_deg(mics, fsel, c,
                                               _axis_step(az_ax),
                                               _axis_step(el_ax),
                                               frac=nms_frac))
    out.update({
        "method": "srp_phat",
        "freqs": fsel,
        "bins": bins,
        "n_frames": int(K),
        "n_fft": int(n_fft),
        "hop": int(hop or (n_fft // 2)),
        "weights": "phat" if weights == "phat" else str(weights),
        "beta": float(beta),
        "near_field_dist": None if src_dist is None else float(src_dist),
    })
    return out


# ==========================================================================
# 2. MUSIC
# ==========================================================================
def music_spectrum(X, fs, mics, freqs_sel=None, n_sources=1, grid_dirs=None,
                   c=C_DEFAULT, n_fft=512, hop=None, loading=0.0, *,
                   freq_band=None, win="blackmanharris", win_table=None,
                   min_sep_deg=None, refine=True,
                   refine_iters=5, az_bins=72, el_bins=37,
                   az_range=(-180.0, 180.0), el_range=(-90.0, 90.0),
                   grid_shape=None, az_axis=None, el_axis=None, chunk=8,
                   src_dist=None, phat=False, hemisphere="auto",
                   nms_frac=0.3):
    """宽带 MUSIC 空间谱。

        R(f) = (1/K)·Σ_k Xf[:,k,f]·Xf[:,k,f]ᴴ
        P(g) = Σ_f 1 / ( a(f,u_g)ᴴ · Eₙ(f)Eₙ(f)ᴴ · a(f,u_g) )

    `Eₙ` = `np.linalg.eigh(R)` 最小 `M - n_sources` 个特征向量张成的噪声子空间
    （纯 numpy，无 scipy）。协方差先 Hermitian 化再分解：`eigh` 只读下三角，
    不化等于静默丢掉一半数据。`loading` 为对角加载系数
    （`R += loading·tr(R)/M·I`），是低快拍/相干源时稳定噪声子空间的常规手段。

    注意：导向矢量取单位模长（MUSIC 的经典形式）。宽带求和时各频点量级差异很大，
    某个频点分母接近 0 就会主导总和；实测中"按频点归一/二值化"反而更差
    （噪声主导的频点被抬到同等权重），所以这里就是朴素求和。

    `freqs_sel / n_fft / hop / grid_* / hemisphere / src_dist` 与 `srp_phat` 同义。
    `nms_frac=0.3`：MUSIC 是超分辨，取峰屏蔽半径远小于主瓣宽度，
    否则会把真正分开的两个源合并（实测两源夹角 33° 仍能分辨）。

    **适用性提醒**（实测结论）：MUSIC 对混响/相干干扰很敏感 ——
    RT60≈0.43 s 的中等混响房间里，本实现的角度误差可达 16°~42°，
    而同一份数据上 SRP-PHAT 只有 3°~4.4°。混响场景请优先用 `srp_phat`。
    """
    X, mics = _obs(X, mics)
    M, N = X.shape
    if M < 2:
        raise ValueError("MUSIC 至少需要 2 个阵元")
    n_sources = int(n_sources)
    if not (1 <= n_sources < M):
        raise ValueError(f"n_sources 必须满足 1 <= n_sources < M (M={M})")
    fs = float(fs)

    dirs, shape, az_ax, el_ax = _parse_grid(
        grid_dirs, az_bins, el_bins, az_range, el_range,
        grid_shape, az_axis, el_axis)

    Xf, _fa = dsp.stft(X, fs, n_fft=n_fft, hop=hop, win=win,
                       win_table=win_table)
    bins, fsel = select_freq_bins(fs, n_fft, mics, c, freqs_sel, freq_band)
    K = Xf.shape[1]

    Z = np.asarray(Xf[:, :, bins], dtype=np.complex128)
    if phat:
        Z = Z / (np.abs(Z) + _EPS)
    R = np.einsum("mkf,nkf->fmn", Z, np.conj(Z)) / float(max(K, 1))
    R = 0.5 * (R + np.conj(np.transpose(R, (0, 2, 1))))
    if loading and loading > 0:
        tr = np.trace(R, axis1=1, axis2=2).real / M                  # (F,)
        R = R + (float(loading) * tr)[:, None, None] * np.eye(M)[None, :, :]

    # eigh 返回升序特征值 -> 前 M-n_sources 个即噪声子空间
    _w, V = np.linalg.eigh(R)                                       # (F,M)
    En = V[:, :, :M - n_sources]                                    # (F,M,Mn)

    def cost(dd):
        P = np.zeros(dd.shape[0], dtype=np.float64)
        F = fsel.size
        for s in range(0, F, int(chunk)):
            e = min(s + int(chunk), F)
            A = _manifold_block(mics, fsel[s:e], dd, c, src_dist)   # (f,G,M)
            An = A / np.maximum(np.linalg.norm(A, axis=2, keepdims=True), _EPS)
            # ‖Eₙᴴa‖² = ‖conj(A)·Eₙ‖²：An 是 (f,G,M)、Eₙ 是 (f,M,Mn)
            # -> Q (f,G,Mn)。直接对投影取模方，比先物化 EₙEₙᴴ 再求二次型更省，
            # 也避免把舍入误差平方放大（MUSIC 的分母本来就可能接近 0）。
            Q = np.matmul(np.conj(An), En[s:e])
            # einsum 不做共轭，必须显式 conj —— 否则算的是 ΣQ²（可正可负），
            # 谱线会完全乱掉。这是本模块最容易踩的一个坑。
            den = np.einsum("fgm,fgm->fg", Q, np.conj(Q)).real
            P += np.sum(1.0 / np.maximum(den, _EPS), axis=0)
        return P

    P = cost(dirs)
    out = _finalize(P, dirs, shape, az_ax, el_ax, n_sources, min_sep_deg,
                    _axis_step(az_ax), _axis_step(el_ax), refine, cost,
                    refine_iters, hemisphere=hemisphere, mics=mics,
                    sep_deg=_beamwidth_nms_deg(mics, fsel, c,
                                               _axis_step(az_ax),
                                               _axis_step(el_ax),
                                               frac=nms_frac))
    out.update({
        "method": "music",
        "freqs": fsel,
        "bins": bins,
        "n_frames": int(K),
        "n_fft": int(n_fft),
        "hop": int(hop or (n_fft // 2)),
        "loading": float(loading),
        "eigvals": _w,
        "near_field_dist": None if src_dist is None else float(src_dist),
    })
    return out


# ==========================================================================
# 3. TDOA 非线性最小二乘
# ==========================================================================
def _pair_delays_dirs(mics, dirs, pairs, c):
    """远场候选方向的阵元对模型时延 (G,P)。与 acoustics.delay_from_dir 同式、向量化。"""
    tau = ac.delays_grid(mics, dirs, c)                    # (G,M) = -p·u/c
    return tau[:, pairs[:, 1]] - tau[:, pairs[:, 0]]


def _pair_delays_points(mics, pts, pairs, c):
    """近场候选点的阵元对模型时延 (G,P)。与 acoustics.delay_from_point 同式、向量化。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)
    d = np.linalg.norm(mics[None, :, :] - pts[:, None, :], axis=2)   # (G,M)
    tau = (d - r[:, None]) / c
    return tau[:, pairs[:, 1]] - tau[:, pairs[:, 0]]


def _fd_jacobian(fn, x, h):
    """中心差分雅可比（前向差分精度不够，LM 收敛会明显变慢）。"""
    x = np.asarray(x, dtype=np.float64)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), x.shape)
    r0 = np.asarray(fn(x), dtype=np.float64).ravel()
    J = np.empty((r0.size, x.size), dtype=np.float64)
    for k in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[k] += h[k]
        xm[k] -= h[k]
        J[:, k] = (np.asarray(fn(xp), float).ravel()
                   - np.asarray(fn(xm), float).ravel()) / (2.0 * h[k])
    return J, r0


def _levenberg_marquardt(fn, x0, h, iters=40, lam0=1e-3, proj=None,
                         max_trials=8):
    """自实现 Levenberg–Marquardt（无 scipy）。

    残差 `fn(x)` 用 `tdoa.delay_residual()` 求（题目指定的共用内核），
    雅可比用中心差分（避免为远场/近场各写一套解析导数，也就不会和
    acoustics 的时延公式产生第二份实现）。

    阻尼项按 `tr(JᵀJ)/n` 缩放而不是直接用单位阵：远场(度)与近场(米)
    的参数尺度差了 5 个数量级，不缩放的话 λ 在一个问题里好用、
    换个问题就废掉。
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    if proj is not None:
        x = proj(x)
    J, r = _fd_jacobian(fn, x, h)
    cost = float(r @ r)
    lam = float(lam0)
    n_it = 0
    for _ in range(int(iters)):
        n_it += 1
        g = J.T @ r
        H = J.T @ J
        n = x.size
        scale = float(np.trace(H)) / max(n, 1)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        prev = cost
        accepted = False
        dx = np.zeros(n)
        for _t in range(int(max_trials)):
            try:
                dx = np.linalg.solve(H + lam * scale * np.eye(n), -g)
            except np.linalg.LinAlgError:
                dx = -g / (np.diag(H) + lam * scale + _EPS)
            xt = x + dx
            if proj is not None:
                xt = proj(xt)
            rt = np.asarray(fn(xt), dtype=np.float64).ravel()
            ct = float(rt @ rt)
            if np.isfinite(ct) and ct < cost:
                x, r, cost = xt, rt, ct
                lam = max(lam * 0.25, 1e-12)
                accepted = True
                break
            lam *= 8.0
        if not accepted:
            break                                   # 无法再下降 -> 收敛/局部极小
        if prev - cost <= 1e-14 * max(prev, 1e-300):
            break                                   # 相对改善已到浮点噪声量级
        J, r = _fd_jacobian(fn, x, h)
    return x, r, cost, n_it


def tdoa_estimate(X, fs, mics, c=C_DEFAULT, max_delay=None, interp=1, *,
                  pairs=None, pair_mode="all", n_fft=None,
                  clamp_to_aperture=True, grid_dirs=None, az_bins=72,
                  el_bins=37, coarse_dists=(0.15, 0.25, 0.4, 0.6, 0.9, 1.3,
                                            2.0, 3.0, 5.0, 8.0, 13.0),
                  r_min=0.05, r_max=20.0, refine_iters=40,
                  refine_iters_far=25, fd_step_ang=1e-3, fd_step_pos=1e-5,
                  hemisphere="auto"):
    """GCC-PHAT 时延 + 非线性最小二乘的远场/近场联合定位。

    流程（全部确定性）：
      1. `tdoa.gcc_phat_matrix` 得到所有阵元对时延 `taus`（含抛物插值亚采样细化）。
      2. **远场**：先用 `direction_grid` 粗搜（向量化模型时延），
         再在 (az, el) 上用 LM 精化（残差函数 = `tdoa.delay_residual`）。
      3. **近场**：粗搜 = 方向网格 × 一组对数间隔的距离，
         再在 (x,y,z) 上用 LM 精化（`delay_residual(..., near_field=True)`，
         即契约的球面波时延 `(‖p_m-p‖-‖p‖)/c`）。
      4. 两个模型都返回；`near_field` 标志与 `az/el` 取残差明显更小的那个
         （近场模型多一个自由度，残差天然更小，所以判据带 1e-3 相对门限）。

    近场为什么能定出距离：波前曲率在阵元间产生的**二阶**时延项
    ≈ (a²-（p_m·u)²)/(2r)，孔径越大、距离越近该项越显著；
    最小二乘直接拟合它就得到 r，这比"三角测量"稳健得多。

    返回 dict：`az, el, xyz, residual_rms, taus, pairs, iterations`
    （外加 `residual_rms_near/xyz_far/az_near/distance/near_field/...` 诊断字段）。
    `residual_rms` 单位是秒（同时给 `residual_rms_m = c·residual_rms`）。
    """
    X, mics = _obs(X, mics)
    M = mics.shape[0]
    if M < 3:
        raise ValueError("TDOA 定位至少需要 3 个阵元（2 个不足以定方位）")
    fs = float(fs)
    c = float(c)

    # ---- 时延观测 -------------------------------------------------------
    if pairs is None and pair_mode == "star":
        pairs = np.stack([np.zeros(M - 1, int), np.arange(1, M)], axis=1)
    if max_delay is None and clamp_to_aperture:
        # 物理上 |tau_ij| <= 孔径/c。全域搜索会把远处噪声峰也纳入竞争，
        # 在这里收窄是"用先验换稳健"，不改变可观测量。
        max_delay = geo.aperture(mics) / c * 1.05 + 1e-9
    taus, pairs = td.gcc_phat_matrix(X, fs, pairs, n_fft=n_fft,
                                     max_delay=max_delay, interp=interp)
    taus = np.asarray(taus, dtype=np.float64).ravel()
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    P_n = pairs.shape[0]

    dirs, _shape, _az_ax, _el_ax = _parse_grid(grid_dirs, az_bins, el_bins)
    hemi, planar, ambiguous = _resolve_hemisphere(hemisphere, mics)
    if hemi == "front":
        dmask = dirs[:, 2] >= -1e-12
        el_lo, el_hi = 0.0, 90.0
    elif hemi == "back":
        dmask = dirs[:, 2] <= 1e-12
        el_lo, el_hi = -90.0, 0.0
    else:
        dmask = np.ones(dirs.shape[0], bool)
        el_lo, el_hi = -90.0, 90.0
    dirs_c = dirs[dmask]                     # 只在前(后)半球里粗搜，见 _resolve_hemisphere

    # ---- 远场：粗搜 + LM ------------------------------------------------
    model = _pair_delays_dirs(mics, dirs_c, pairs, c)                # (G,P)
    res = taus[None, :] - model
    rms = np.sqrt(np.mean(res ** 2, axis=1))
    k0 = int(np.argmin(rms))
    az0, el0 = ac.angles_from_dir(dirs_c[k0])

    def res_far(a):
        return td.delay_residual(taus, pairs, mics, ac.unit_dir(a[0], a[1]), c,
                                 near_field=False)

    def proj_far(a):
        a = np.asarray(a, dtype=np.float64).copy()
        a[0] = (a[0] + 180.0) % 360.0 - 180.0
        a[1] = float(np.clip(a[1], el_lo, el_hi))
        return a

    xf, rf, cost_f, it_f = _levenberg_marquardt(
        res_far, np.array([az0, el0]), [fd_step_ang, fd_step_ang],
        iters=refine_iters_far, proj=proj_far)
    xf = proj_far(xf)
    az_f, el_f = ac.angles_from_dir(ac.unit_dir(xf[0], xf[1]))
    rms_f = float(np.sqrt(np.mean(rf ** 2)))

    # ---- 近场：粗搜（方向 × 距离）+ LM ---------------------------------
    dists = np.asarray(coarse_dists, dtype=np.float64)
    best_p, best_rms = None, np.inf
    for r in dists:
        pts = dirs_c * float(r)
        m = _pair_delays_points(mics, pts, pairs, c)
        rr = taus[None, :] - m
        v = np.sqrt(np.mean(rr ** 2, axis=1))
        k = int(np.argmin(v))
        if float(v[k]) < best_rms:
            best_rms = float(v[k])
            best_p = pts[k].copy()

    def res_near(p):
        return td.delay_residual(taus, pairs, mics, p, c, near_field=True)

    def proj_near(p):
        p = np.asarray(p, dtype=np.float64).copy()
        r = float(np.linalg.norm(p))
        if not np.isfinite(r) or r < 1e-9:
            return np.array([0.0, 0.0, float(r_min)])
        if r > r_max:
            p *= r_max / r
        elif r < r_min:
            p *= r_min / r
        # 半球约束同样施加在近场解上，否则同一个镜像最小点会让结果来回跳
        if hemi == "front":
            p[2] = max(p[2], 0.0)
        elif hemi == "back":
            p[2] = min(p[2], 0.0)
        return p

    xn, rn, cost_n, it_n = _levenberg_marquardt(
        res_near, best_p, [fd_step_pos] * 3, iters=refine_iters,
        proj=proj_near)
    xn = proj_near(xn)
    rms_n = float(np.sqrt(np.mean(rn ** 2)))
    az_n, el_n = ac.angles_from_dir(xn)

    # 多自由度模型残差必然 <= 少自由度模型；要求"明显更小"才认为是真近场
    near = bool(rms_n < rms_f * (1.0 - 1e-3))
    if near:
        az, el, rms_sel, it_sel = az_n, el_n, rms_n, it_n
    else:
        az, el, rms_sel, it_sel = az_f, el_f, rms_f, it_f

    az = (az + 180.0) % 360.0 - 180.0
    return {
        "az": float(az),
        "el": float(el),
        "xyz": np.asarray(xn, dtype=np.float64),
        "residual_rms": float(rms_sel),
        "residual_rms_m": float(rms_sel * c),
        "residual_rms_far": rms_f,
        "residual_rms_near": rms_n,
        "taus": taus,
        "pairs": pairs,
        "iterations": int(it_sel),
        "iterations_far": int(it_f),
        "iterations_near": int(it_n),
        "az_far": float(az_f),
        "el_far": float(el_f),
        "az_near": float(az_n),
        "el_near": float(el_n),
        "xyz_far": ac.unit_dir(az_f, el_f),
        "distance": float(np.linalg.norm(xn)),
        "near_field": near,
        "method": "tdoa",
        "hemisphere": hemi,
        "planar_array": bool(planar),
        "front_back_ambiguous": bool(ambiguous),
        "n_pairs": int(P_n),
        "max_delay": None if max_delay is None else float(max_delay),
        "interp": int(interp),
    }


# ==========================================================================
# 4. 近场距离（固定方向扫距离）
# ==========================================================================
def near_field_distance(X, fs, mics, az, el, c=C_DEFAULT, *,
                        max_delay=None, interp=1, pairs=None, n_fft=None,
                        r_min=0.1, r_max=10.0, n_scan=96, refine=True,
                        fd_step_pos=1e-5, refine_iters=30):
    """固定 (az, el) 下用波前曲率估计声源距离。

    做法：在 log 均匀的距离栅格上取 `p(r) = unit_dir(az,el)·r`，
    用 `tdoa.delay_residual(..., near_field=True)` 的 RMS 作为代价函数，
    取最小值后再在 log r 上做一次抛物线插值细化（确定性、无迭代初值问题）。

    返回 dict：`distance`（米）、`curve`（{distance, cost} 供画曲线）、
    以及 `residual_rms / taus / pairs / az / el / distance_raw` 等。
    """
    X, mics = _obs(X, mics)
    M = mics.shape[0]
    if M < 3:
        raise ValueError("近场距离估计至少需要 3 个阵元")
    fs = float(fs)
    c = float(c)

    if max_delay is None:
        max_delay = geo.aperture(mics) / c * 1.05 + 1e-9
    taus, pairs = td.gcc_phat_matrix(X, fs, pairs, n_fft=n_fft,
                                     max_delay=max_delay, interp=interp)
    taus = np.asarray(taus, float).ravel()
    pairs = np.asarray(pairs, int).reshape(-1, 2)

    u = ac.unit_dir(az, el)
    r_scan = np.geomspace(float(r_min), float(r_max), int(n_scan))
    pts = r_scan[:, None] * u[None, :]
    model = _pair_delays_points(mics, pts, pairs, c)
    res = taus[None, :] - model
    cost = np.sqrt(np.mean(res ** 2, axis=1))

    k = int(np.argmin(cost))
    r_best = float(r_scan[k])

    if refine and 0 < k < r_scan.size - 1:
        # log r 上三点抛物线插值（代价函数在 log 域近似抛物线）
        x = np.log(r_scan[k - 1:k + 2])
        y = cost[k - 1:k + 2]
        den = (y[0] - 2.0 * y[1] + y[2])
        if abs(den) > _EPS:
            d = 0.5 * (y[0] - y[2]) / den
            if abs(d) <= 1.0:
                r_best = float(np.exp(x[1] + d * (x[2] - x[1])))

    # 可选：以插值点为初值再做一次 3 参数 LM（曲面不在 log 域严格抛物线时更准）
    if refine:
        def res_near(p):
            return td.delay_residual(taus, pairs, mics, p, c, near_field=True)

        def proj(p):
            p = np.asarray(p, float).copy()
            r = float(np.linalg.norm(p))
            if not np.isfinite(r) or r < 1e-9:
                return np.array([0.0, 0.0, float(r_min)])
            return p * float(np.clip(r, r_min, r_max)) / r

        p0 = u * r_best
        xr, _rr, _cr, _it = _levenberg_marquardt(
            res_near, p0, [fd_step_pos] * 3, iters=refine_iters, proj=proj)
        xr = proj(xr)
        # 只有当 LM 没跑出方向范围时才采纳（防止跑到别的方向去）
        az_lm, el_lm = ac.angles_from_dir(xr)
        if abs(((az_lm - float(az) + 180.0) % 360.0) - 180.0) < 5.0 and \
                abs(el_lm - float(el)) < 5.0:
            r_best = float(np.linalg.norm(xr))

    return {
        "distance": float(r_best),
        "curve": {
            "distance": [float(v) for v in r_scan],
            "cost": [float(v) for v in cost],
        },
        "az": float(az),
        "el": float(el),
        "residual_rms": float(cost[k]),
        "r_scan_best": float(r_scan[k]),
        "taus": taus,
        "pairs": pairs,
        "method": "near_field_distance",
    }


# ==========================================================================
# 5. 统一入口
# ==========================================================================
def localize(X, fs, mics, method="srp_phat", *, truth=None, n_sources=1,
             grid_dirs=None, c=C_DEFAULT, n_fft=512, hop=None, freqs=None,
             freq_band=None, map_fallback=True, **kw):
    """统一定位入口：返回统一结构的 dict。

    method:
      * 'srp_phat' —— 稳健、无需源数先验，默认首选
      * 'music'    —— 高分辨，需要正确的 `n_sources`
      * 'tdoa'     —— 只用 GCC-PHAT 时延，给远场 (az,el) 与近场 xyz
      * 'hybrid'   —— SRP-PHAT 先定"正面/背面"（它对混响稳健），
                      再把该半球约束交给 TDOA 做连续域精化（精度高），
                      同时输出 `est_xyz` 与 SRP-PHAT 空间谱图

    统一字段（CONTRACT 第 8 节 /api/locate）：`map, shape, az_axis, el_axis,
    est_dirs, est_idx, peak_vals, method, n_sources, truth`，
    以及 `est_xyz`（近场坐标，仅 tdoa/hybrid 有）与各方法的诊断字段。
    `truth` 由调用方传入（`sim.SimResult.truth` 直接给即可），原样回填。
    """
    fs = float(fs)
    X, mics = _obs(X, mics)
    method = str(method).lower()

    if method in ("srp", "srp-phat", "srpphat"):
        method = "srp_phat"
    if method in ("tdoa_near", "gcc"):
        method = "tdoa"

    if method == "srp_phat":
        out = srp_phat(X, fs, mics, grid_dirs, c=c, n_fft=n_fft, hop=hop,
                       freqs=freqs, freq_band=freq_band, n_sources=n_sources,
                       **kw)
        out["est_xyz"] = None
    elif method == "music":
        out = music_spectrum(X, fs, mics, freqs, n_sources=n_sources,
                             grid_dirs=grid_dirs, c=c, n_fft=n_fft, hop=hop,
                             freq_band=freq_band, **kw)
        out["est_xyz"] = None
    elif method in ("tdoa", "hybrid"):
        vis = None
        if method == "hybrid" or map_fallback:
            # 统一 schema：没有空间谱的方法也给一张 SRP-PHAT 图做可视化兜底
            vis = srp_phat(X, fs, mics, grid_dirs, c=c, n_fft=n_fft, hop=hop,
                           freqs=freqs, freq_band=freq_band, n_sources=1,
                           refine=False,
                           **{k: v for k, v in kw.items()
                              if k in ("az_bins", "el_bins", "az_range",
                                       "el_range", "weights", "beta",
                                       "win", "win_table", "chunk",
                                       "grid_shape", "az_axis", "el_axis",
                                       "hemisphere")})
        td_kw = dict(kw)
        if (method == "hybrid" and vis is not None and vis["est_dirs"]
                and "hemisphere" not in td_kw):
            # 混合策略的真作用：用 SRP-PHAT（对混响稳健但精度低）先定出
            # 声源在阵列正面还是背面，把这个二值信息交给 TDOA（精度高但
            # 对前后模糊同样无能为力），再由 TDOA 做连续域精化。
            el_p = float(vis["est_dirs"][0][1])
            td_kw["hemisphere"] = "front" if el_p >= 0.0 else "back"
        td_out = tdoa_estimate(X, fs, mics, c=c, **td_kw)
        out = dict(td_out)
        out["est_dirs"] = [[round(td_out["az"], 6), round(td_out["el"], 6)]]
        out["est_xyz"] = np.asarray(td_out["xyz"], float)
        out["est_idx"] = []
        out["peak_vals"] = []
        out["n_sources"] = 1
        out["freqs"] = None
        if vis is not None:
            out["map"] = vis["map"]
            out["map_scale"] = vis["map_scale"]
            out["shape"] = vis["shape"]
            out["az_axis"] = vis["az_axis"]
            out["el_axis"] = vis["el_axis"]
            out["srp_peak_dirs"] = vis["est_dirs"]
            if method == "hybrid":
                out["hybrid_srp_dirs"] = vis["est_dirs"]
        if method == "hybrid":
            out["method"] = "hybrid"
    else:
        raise ValueError(f"unknown method: {method!r}")

    out["method"] = out.get("method", method)
    out["truth"] = truth
    # 统一出口的 JSON 友好化：numpy 标量/数组 -> 原生类型
    out["est_idx"] = [int(i) for i in out.get("est_idx", [])]
    out["peak_vals"] = [float(v) for v in out.get("peak_vals", [])]
    out["est_dirs"] = [[float(a), float(e)] for a, e in out.get("est_dirs", [])]
    out.setdefault("n_sources", len(out["est_dirs"]))
    return out
