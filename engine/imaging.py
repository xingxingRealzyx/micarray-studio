"""声场成像：点扩散函数、脏图、反卷积（DAMAS / CLEAN-SC）与波束成形变体。

本模块提供声源成像（声全息）的完整链路：

    point_spread_function / psf_columns   点扩散函数 PSF（G_steer × G_src）
    dirty_map                             常规延迟求和（频域）脏图
    damas                                 DAMAS 反卷积（Gauss-Seidel）
    clean_sc                              CLEAN-SC（基于空间相干性的源分量剥离）
    orthogonal_beamforming                正交波束成形（特征分解去相干）
    functional_beamforming                函数波束成形（R^p 幂律压缩）
    compare_methods                       同一份数据跑多方法 + 对比指标

约定（严格遵守 docs/CONTRACT.md）
--------------------------------
* 坐标系：阵元在 xy 平面、boresight = +z、`el` 从 xy 平面量向 +z（第 1、2 节）。
  本文件**不重复实现**任何角度→方向、时延、导向矢量公式，全部调用
  `engine.acoustics` / `engine.steering`；方向之间的夹角只做纯几何的
  `arccos(dot(u1,u2))`（球面大圆角，不是坐标系约定）。
* 频域：统一走 `engine.dsp.stft`（rfft 单边谱、Blackman-Harris 窗、
  hop = n_fft//2、不做幅度补偿），与 C++ 导出逐位一致（第 5 节）。
* 协方差：`R(f) = E_k[ Xf[:,k,f] Xf[:,k,f]^H ]`，对帧平均（第 5 节）。
* 所有 map 输出都是「扁平数组 + shape」，且不含 NaN / Inf（第 8 节）。

数值与算法说明放在各函数内部的中文注释里，重点解释：
  - PSF 的归一化为什么让对角恒为 1，以及它与脏图的尺度自洽性；
  - DAMAS 的 Gauss-Seidel 迭代、松弛、非负约束与提前停止判据；
  - CLEAN-SC 如何用「脏图峰的波束成形输出与各网格点的相干性」定义源分量，
    以及它对相干声源为什么比"直接减 PSF"更正确。
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque

import numpy as np

from . import acoustics as ac
from . import dsp
from . import geometry as gm
from . import steering as st

C_DEFAULT = ac.C_DEFAULT
_EPS = 1e-30
# 稠密 PSF 缓存上限（字节）：超过则退回"按需算列"的慢路径（内存优先）
_MAX_CACHE_BYTES = 256 << 20


# ==========================================================================
# 0. 通用小工具
# ==========================================================================
def _sanitize(a, fill: float = 0.0) -> np.ndarray:
    """把 NaN / ±Inf 换成有限值。契约要求所有输出可 JSON 序列化。"""
    a = np.asarray(a, dtype=np.float64)
    if a.size and not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
    return a


def _finite(x, default: float = 0.0) -> float:
    """标量版清洗：非有限值 → default。"""
    x = float(x)
    return x if np.isfinite(x) else float(default)


def _normalize_map(a) -> tuple[np.ndarray, int, float]:
    """归一化到 max = 1。返回 (归一化数组, 峰值下标, 峰值原值)。

    全零 / 全负 / 非有限输入都会退化为全零 + 下标 -1，绝不产生 NaN。
    """
    m = _sanitize(a)
    if m.size == 0:
        return m, -1, 0.0
    idx = int(np.argmax(m))
    pk = float(m[idx])
    if not np.isfinite(pk) or pk <= _EPS:
        return np.zeros_like(m), idx, pk
    return m / pk, idx, pk


def _as_multichannel(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("X 必须是 (M,N) 或 (N,) 的多通道时域信号")
    return X


def _check_mics(mics) -> np.ndarray:
    mics = np.asarray(mics, dtype=np.float64)
    if mics.ndim != 2 or mics.shape[1] != 3:
        raise ValueError("mics 必须是 (M,3) 的阵元坐标")
    return mics


def _flat_to_grid(flat, shape):
    """(G,) 扁平 -> shape（1 维网格则原样返回）。"""
    f = np.asarray(flat, dtype=np.float64)
    if len(shape) == 2 and int(np.prod(shape)) == f.size:
        return f.reshape(shape)
    return f.reshape(-1)


def _axes_from_dirs(dirs: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """从 (G,3) 反推 (az_axis, el_axis)。

    只对**规则网格**成立：az 轴取第一行、el 轴取第一列。
    角度反解调用 engine.acoustics.angles_from_dir（契约第 2 节）。
    """
    n_el, n_az = int(shape[0]), int(shape[1])
    az = np.array([ac.angles_from_dir(dirs[j])[0] for j in range(n_az)])
    el = np.array([ac.angles_from_dir(dirs[i * n_az])[1] for i in range(n_el)])
    return az, el


def _parse_grid(grid_dirs):
    """统一解析方向网格，返回 (dirs (G,3), shape, az_axis, el_axis)。

    接受：
      * `geometry.direction_grid()` 的返回值 `(dirs, shape, az, el)`
      * `(dirs, shape)` 二元组
      * 裸 `(G,3)` 数组
      * 含 `dirs/shape/az_axis/el_axis` 的 dict
    """
    az = el = None
    shape = None
    if isinstance(grid_dirs, dict):
        dirs = grid_dirs.get("dirs", grid_dirs.get("directions"))
        if dirs is None:
            raise ValueError("grid_dirs dict 必须含 'dirs'")
        shape = grid_dirs.get("shape")
        az, el = grid_dirs.get("az_axis"), grid_dirs.get("el_axis")
    elif isinstance(grid_dirs, (tuple, list)):
        if len(grid_dirs) == 0:
            raise ValueError("grid_dirs 不能是空元组")
        dirs = grid_dirs[0]
        if len(grid_dirs) >= 2 and grid_dirs[1] is not None:
            shape = tuple(int(v) for v in grid_dirs[1])
        if len(grid_dirs) >= 4:
            az = None if grid_dirs[2] is None else np.asarray(grid_dirs[2], float)
            el = None if grid_dirs[3] is None else np.asarray(grid_dirs[3], float)
    else:
        dirs = grid_dirs

    dirs = np.asarray(dirs, dtype=np.float64)
    if dirs.ndim == 1:
        dirs = dirs.reshape(-1, 3)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("方向网格必须是 (G,3) 单位向量")
    n = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(n, 1e-12)

    if shape is None or int(np.prod(shape)) != dirs.shape[0]:
        shape = (dirs.shape[0],)
    if (az is None or el is None) and len(shape) == 2:
        az, el = _axes_from_dirs(dirs, shape)
    return dirs, tuple(int(v) for v in shape), az, el


def _grid_step_deg(dirs: np.ndarray) -> float:
    """网格平均间距（度）：每个点到最近邻的球面大圆角的平均值。

    只用方向向量做纯几何计算，不涉及 az/el 约定。
    """
    if dirs.shape[0] < 2:
        return 5.0
    # 抽样以避免 O(G^2)：最多取 512 个点做最近邻
    n = dirs.shape[0]
    sub = dirs if n <= 512 else dirs[np.linspace(0, n - 1, 512).round().astype(int)]
    cosang = np.clip(sub @ dirs.T, -1.0, 1.0)
    ang = np.rad2deg(np.arccos(cosang))              # (S,G)
    ang[ang < 1e-9] = np.inf                          # 排除自身
    nn = np.min(ang, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:                                  # 所有方向重合（退化输入）
        return 5.0
    return float(np.mean(nn))


def _angular_distance_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """方向向量之间的球面夹角（度）。纯几何量，与坐标系约定无关。"""
    u = np.atleast_2d(u)
    v = np.atleast_2d(v)
    cosang = np.clip(u @ v.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


# ==========================================================================
# 1. 网格几何量 / 图像指标（GUI 对比图与验收测试共用）
# ==========================================================================
def _cell_solid_angles(shape, az_axis, el_axis, dirs) -> np.ndarray:
    """每个网格点代表的立体角（球面度），用于把面积换算成角宽。"""
    if len(shape) == 2 and az_axis is not None and el_axis is not None \
            and az_axis.size == shape[1] and el_axis.size == shape[0]:
        daz = abs(float(az_axis[1] - az_axis[0])) if az_axis.size > 1 else 5.0
        dele = abs(float(el_axis[1] - el_axis[0])) if el_axis.size > 1 else 5.0
        w = np.deg2rad(daz) * np.deg2rad(dele) * np.cos(np.deg2rad(np.asarray(el_axis, float)))
        return np.repeat(np.maximum(w, 1e-12), shape[1])
    # 退化情形：用最近邻角距平方近似单元面积
    step = np.deg2rad(_grid_step_deg(dirs))
    return np.full(dirs.shape[0], max(step * step, 1e-12))


def _az_periodic(az_axis, shape) -> bool:
    if len(shape) != 2 or az_axis is None or az_axis.size < 3:
        return False
    return abs(abs(float(az_axis[-1] - az_axis[0])) - 360.0) < 1e-6


def _neighbour_offsets(periodic_az: bool):
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return offs


def _dilate(mask: np.ndarray, periodic_az: bool, steps: int = 1) -> np.ndarray:
    """布尔掩码膨胀若干格（az 方向可按周期环绕）。用于"主瓣外扩一圈"。"""
    out = np.asarray(mask, bool).copy()
    for _ in range(max(int(steps), 0)):
        acc = out.copy()
        acc |= np.roll(out, 1, axis=0)
        acc |= np.roll(out, -1, axis=0)
        acc |= np.roll(out, 1, axis=1)
        acc |= np.roll(out, -1, axis=1)
        if not periodic_az:      # 非周期网格：边缘不环绕，直接取原值
            acc[0, :] = out[0, :]
            acc[-1, :] = out[-1, :]
            acc[:, 0] = out[:, 0]
            acc[:, -1] = out[:, -1]
        out = acc
    return out


def _all_components(mask: np.ndarray, periodic_az: bool) -> np.ndarray:
    """掩码里所有连通区域的编号图（0 = 背景，1..n = 区域号）。"""
    lab = np.zeros(mask.shape, dtype=np.int32)
    cur = 0
    # 只遍历被掩码点（通常远少于全部格点），避免 O(G) 次无谓的数组拷贝
    for i, j in np.argwhere(mask):
        if lab[i, j] == 0:
            cur += 1
            comp = _connected_component(mask & (lab == 0), (int(i), int(j)), periodic_az)
            lab[comp] = cur
    return lab


def _connected_component(mask: np.ndarray, start: tuple[int, int],
                         periodic_az: bool) -> np.ndarray:
    """在布尔掩码上从 start 做 4/8 邻域洪泛（az 方向可按周期环绕）。"""
    n_el, n_az = mask.shape
    out = np.zeros_like(mask)
    if not mask[start]:
        return out
    q = deque([start])
    out[start] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if ii < 0 or ii >= n_el:
                continue
            if periodic_az:
                jj %= n_az
            elif jj < 0 or jj >= n_az:
                continue
            if mask[ii, jj] and not out[ii, jj]:
                out[ii, jj] = True
                q.append((ii, jj))
    return out


def _lobe_area(flat_map, shape, az_axis, el_axis, dirs, peak_index: int,
               level: float = 0.5) -> tuple[float, np.ndarray]:
    """主瓣面积（立体角）与主瓣掩码：map >= level*peak 且与峰值连通的区域。"""
    m = _sanitize(flat_map)
    if m.size == 0 or peak_index < 0:
        return 0.0, np.zeros(m.size, bool)
    if len(shape) != 2:
        # 一维网格：无法谈"面积"，用连通点数 * 单元面积
        mask = m >= level * m[peak_index]
        return float(np.sum(_cell_solid_angles(shape, az_axis, el_axis, dirs)[mask])), mask
    grid = _flat_to_grid(m, shape)
    peak = float(grid.ravel()[peak_index])
    mask = grid >= level * peak
    i0, j0 = np.unravel_index(peak_index, shape)
    comp = _connected_component(mask, (int(i0), int(j0)), _az_periodic(az_axis, shape))
    omega = _cell_solid_angles(shape, az_axis, el_axis, dirs).reshape(shape)
    return float(np.sum(omega[comp])), comp.ravel()


def main_lobe_width(flat_map, shape, dirs, az_axis=None, el_axis=None,
                    level: float = 0.5) -> float:
    """主瓣宽度（度）——「等效圆直径」定义。

    取 map >= level*peak（默认 -3 dB）且与峰值连通的区域，累加其立体角 A，
    再换算成等面积圆的直径 2*sqrt(A/pi)。对孤立主瓣，该值≈常规 -3dB 波束宽度；
    对两个糊在一起的主瓣，连通区面积更大 → 宽度更大，能反映"分不开"。
    """
    _, idx, _ = _normalize_map(flat_map)
    A, _ = _lobe_area(flat_map, shape, az_axis, el_axis, dirs, idx, level)
    if A <= 0:
        return 0.0
    return float(np.rad2deg(2.0 * np.sqrt(A / np.pi)))


def find_peaks(flat_map, shape=None, dirs=None, az_axis=None, el_axis=None,
               threshold: float = 0.5, max_peaks: int = 8,
               min_sep_deg: float | None = None, exclude=None,
               relative_to_max: bool = True) -> list[dict]:
    """贪心峰值搜索（确定性，无 scipy）。

    反复取当前最大值，然后在角距 `min_sep_deg` 内抑制（置 -inf），直到
    峰值低于 `threshold * 全局峰值` 或达到 max_peaks。比"局部极大值"稳健：
    平顶、噪声毛刺都不会造成重复计数。

    exclude: 需要排除的网格下标（例如 CLEAN-SC 已剥离过的点）。
    """
    m = _sanitize(flat_map)
    G = m.size
    if G == 0:
        return []
    if shape is None:
        shape = (G,)
    if dirs is None:
        return []
    dirs = np.asarray(dirs, float)
    thr = float(threshold) * (float(m.max()) if relative_to_max else 1.0)
    if min_sep_deg is None:
        min_sep_deg = 1.5 * _grid_step_deg(dirs)
    work = m.copy()
    if exclude is not None and len(exclude):
        work[np.asarray(list(exclude), int)] = -np.inf
    cos_thr = np.cos(np.deg2rad(float(min_sep_deg)))
    if not np.isfinite(cos_thr):
        cos_thr = np.cos(np.deg2rad(1.5 * 5.0))
    peaks: list[dict] = []
    for _ in range(int(max_peaks)):
        idx = int(np.argmax(work))
        val = float(work[idx])
        if not np.isfinite(val) or val < thr:
            break
        az, el = ac.angles_from_dir(dirs[idx])
        peaks.append({"index": idx, "value": val,
                      "value_rel": float(val / max(m.max(), _EPS)),
                      "az": az, "el": el,
                      "dir": [float(v) for v in dirs[idx]]})
        # 抑制角距内的所有点
        supp = (dirs @ dirs[idx]) >= cos_thr
        work[supp] = -np.inf
    return peaks


def map_metrics(flat_map, shape=None, dirs=None, az_axis=None, el_axis=None,
                sidelobe_level: float = 0.25, peak_index: int | None = None,
                n_peaks_max: int = 6) -> dict:
    """单张声图的对比指标（GUI 对比图与验收测试共用）。

    * `main_lobe_width_deg` : 等效主瓣宽度（见 `main_lobe_width`）
    * `psl_db`              : 峰值旁瓣级。主瓣定义为 -6dB 连通区（跟随主瓣真实
                              形状，低仰角时主瓣在方位向被拉长也能正确排除）
                              再外扩 1 格，取其外的最大值相对峰值取 dB。
    * `dynamic_range_db`    : 峰值 /（正值 5% 分位）dB，截断到 [0,120]，
                              避免 DAMAS 大面积精确零导致 inf。
    * `null_fraction`       : 精确为零的网格比例（DAMAS 稀疏性的体现）。
    """
    m = _sanitize(flat_map)
    out = {"main_lobe_width_deg": 0.0, "psl_db": 0.0, "dynamic_range_db": 0.0,
           "null_fraction": 1.0, "peak_index": -1, "peak_az": 0.0, "peak_el": 0.0,
           "peak_value": 0.0, "n_peaks": 0, "peaks": []}
    if m.size == 0:
        return out
    if shape is None:
        shape = (m.size,)
    if dirs is None:
        dirs = np.zeros((m.size, 3))
        dirs[:, 2] = 1.0

    norm, idx, pk = _normalize_map(m)
    if peak_index is not None and 0 <= int(peak_index) < m.size:
        idx = int(peak_index)
    out["peak_index"] = int(idx)
    out["peak_value"] = _finite(m[idx])
    out["null_fraction"] = _finite(float(np.mean(np.abs(m) <= 0.0)))
    az, el = ac.angles_from_dir(dirs[idx])
    out["peak_az"], out["peak_el"] = _finite(az), _finite(el)

    if len(shape) == 2:
        _, comp = _lobe_area(m, shape, az_axis, el_axis, dirs, idx, sidelobe_level)
        periodic = _az_periodic(az_axis, shape)
        grid_m = _flat_to_grid(m, shape)
        mask = grid_m >= sidelobe_level * float(m[idx])
        # 所有 >= sidelobe_level 的连通区域都算"主瓣"（多源图里有多个主瓣，
        # 第二个源不该被当成旁瓣），各自外扩 1 格后整体排除。
        lab = _all_components(mask, periodic)
        main_lobe = _dilate(lab > 0, periodic, 1)
        if not np.any(main_lobe):        # 退化保护：至少排除峰值连通区
            main_lobe = _dilate(np.asarray(comp, bool).reshape(shape), periodic, 1)
        outside = ~main_lobe.ravel()
        if np.any(outside):
            out["psl_db"] = _finite(10.0 * np.log10(
                max(float(np.max(m[outside])), _EPS) / max(float(m[idx]), _EPS)))
    out["main_lobe_width_deg"] = _finite(main_lobe_width(
        m, shape, dirs, az_axis, el_axis, level=0.5))

    pos = m[m > 0]
    if pos.size:
        floor = float(np.percentile(pos, 5.0))
        dr = 10.0 * np.log10(max(float(m[idx]), _EPS) / max(floor, _EPS))
        out["dynamic_range_db"] = _finite(float(np.clip(dr, 0.0, 120.0)), 0.0)

    peaks = find_peaks(m, shape, dirs, az_axis, el_axis, threshold=0.5,
                       max_peaks=n_peaks_max)
    out["n_peaks"] = len(peaks)
    out["peaks"] = [{"az": _finite(p["az"]), "el": _finite(p["el"]),
                     "value_rel": _finite(p["value_rel"]),
                     "index": int(p["index"])} for p in peaks]
    return out


def resolution_check(flat_map, dirs, shape=None, az_axis=None, el_axis=None,
                     true_dirs=None, dip_ratio: float = 0.7,
                     min_rel: float = 0.25, search_deg: float | None = None,
                     tol_deg: float = 5.0) -> dict:
    """判断声源在图上是否**真的被分开**，并给出每个源的峰值位置误差。

    分辨判据（比"数局部极大值"严格）：在两个真值方向之间的网格路径上取样，
    若路径最小值相对两侧峰值有明显凹陷（`dip < dip_ratio`）且两个真值附近
    都有足够高的响应（`>= min_rel × 全局峰值`），才算分辨开。糊成一片时
    路径是平的（dip ≈ 1），即使数值上存在两个微弱的局部极大。

    另外对每个真值方向，在 `search_deg` 邻域内找**最高点**，给出它与真值的
    夹角误差 `err_deg`（这才是"峰值位置误差"；直接把真值网格点的值当结果
    是不对的）。`within_tol` 用 `tol_deg` 判定是否达标。
    """
    res = {"resolved": False, "pairs": [], "sources": [],
           "n_resolved_sources": 0}
    if true_dirs is None:
        return res
    m = _sanitize(flat_map)
    dirs = np.asarray(dirs, float)
    if shape is None:
        shape = (m.size,)
    tds = [np.asarray(d, float) for d in np.atleast_2d(true_dirs)]
    if len(tds) == 0:
        return res
    pk = float(np.max(np.abs(m))) if m.size else 0.0
    if search_deg is None:
        search_deg = max(1.5 * _grid_step_deg(dirs), 8.0)
    ang_all = np.stack([_angular_distance_deg(dirs, d)[:, 0] for d in tds], axis=1)
    idxs = [int(np.argmin(ang_all[:, k])) for k in range(len(tds))]

    # 每个真值方向的"最近峰"
    n_ok = 0
    for k, d in enumerate(tds):
        near = ang_all[:, k] <= float(search_deg)
        if np.any(near):
            j = int(np.argmax(np.where(near, m, -np.inf)))
            v = float(m[j])
            err = float(ang_all[j, k])
        else:
            j, v, err = idxs[k], float(m[idxs[k]]), float(ang_all[idxs[k], k])
        ok = bool(v >= min_rel * pk and err <= tol_deg)
        n_ok += int(ok)
        azp, elp = ac.angles_from_dir(dirs[j])
        res["sources"].append({
            "k": k, "index": j, "az": _finite(azp), "el": _finite(elp),
            "value": _finite(v), "value_rel": _finite(v / max(pk, _EPS)),
            "err_deg": _finite(err), "within_tol": ok,
            "value_at_truth": _finite(float(m[idxs[k]])),
        })
    res["n_resolved_sources"] = int(n_ok)

    if len(tds) < 2 or len(shape) != 2:
        res["resolved"] = bool(n_ok == len(tds))
        return res
    grid = _flat_to_grid(m, shape)
    n_el, n_az = shape
    periodic = _az_periodic(az_axis, shape)
    all_ok = True
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            i1, j1 = np.unravel_index(idxs[a], shape)
            i2, j2 = np.unravel_index(idxs[b], shape)
            v1, v2 = float(grid[i1, j1]), float(grid[i2, j2])
            c2 = int(j2)
            if periodic and abs(c2 - j1) > n_az / 2:      # 走短弧
                c2 = c2 - n_az if c2 > j1 else c2 + n_az
            n = int(max(abs(i2 - i1), abs(c2 - j1)) * 3) + 3
            rs = np.linspace(i1, i2, n).round().astype(int)
            cs = np.linspace(j1, c2, n).round().astype(int)
            if periodic:
                cs = cs % n_az
            path = grid[rs, cs]
            lo = float(np.min(path))
            base = min(v1, v2)
            dip = lo / base if base > _EPS else 1.0
            both = (v1 >= min_rel * pk) and (v2 >= min_rel * pk)
            ok = bool(both and dip < dip_ratio)
            all_ok = all_ok and ok
            res["pairs"].append({
                "i": a, "j": b, "resolved": ok, "dip": _finite(dip),
                "value_i": _finite(v1), "value_j": _finite(v2),
                "err_i_deg": _finite(res["sources"][a]["err_deg"]),
                "err_j_deg": _finite(res["sources"][b]["err_deg"]),
                "peak_sep_deg": _finite(_angular_distance_deg(tds[a], tds[b])[0, 0]),
            })
    res["resolved"] = bool(all_ok)
    return res


# ==========================================================================
# 2. 频点选择 / 协方差 / 导向矩阵缓存
# ==========================================================================
def _select_freqs(freqs_axis, mics, c, freqs=None, max_freqs: int | None = 64,
                  f_min=None, f_max=None) -> np.ndarray:
    """挑选参与成像的频点下标（完全确定性）。

    * 显式给 `freqs`（Hz）时，每个频率取最近的 rfft 频点并去重。
    * 否则自动选带 [f_min, f_max]，默认
        f_min = c / D              （D = 最大阵元间距；波长大于孔径时没有分辨力）
        f_max = min(0.95*Nyquist, c/(2*d_min))   （d_min = 最近阵元间距，抑制栅瓣）
    * 频点数超过 `max_freqs` 时等间隔抽稀（首尾保留）——这是"对频率降采样"
      的性能手段：宽带成像的map主要由频带包络决定，抽稀后形状几乎不变，
      但 PSF 与脏图的代价按频点数线性下降。

    注意（实测过但**没有采纳**的改进）：曾试过按信号能量挑频点
    （只用有能量的 bin，避免拿纯噪声频点成像）。理论上更合理，但实测
    在这套测试场景下没有可复现的收益（8/16/32 点两种选法误差相同，
    64 点两种都准），因此回退为纯几何选频，保持行为简单可预测。
    若你的信号谱很窄（如纯音或窄带），请**显式传 `freqs`** 指定频点，
    或把 `max_freqs` 调小 —— 宽带信号才适合自动选带。
    """
    f = np.asarray(freqs_axis, float)
    if freqs is not None:
        want = np.atleast_1d(np.asarray(freqs, float))
        want = want[np.isfinite(want)]
        if want.size == 0:
            raise ValueError("freqs 为空")
        idx = np.unique(np.array([int(np.argmin(np.abs(f - w))) for w in want]))
        idx = idx[idx > 0]
        if idx.size == 0:
            raise ValueError("freqs 全部落在 0 Hz 附近，请给出 > 0 的频率")
        return idx

    D = gm.aperture(mics)
    dmin = gm.min_spacing(mics)
    lo = float(f_min) if f_min is not None else float(c) / max(D, 1e-9)
    hi = float(f_max) if f_max is not None else min(
        0.95 * float(f[-1]), float(c) / max(2.0 * dmin, 1e-9))
    if hi <= lo:                       # 退化保护：带太小则围绕 lo 取一段
        hi = lo * 1.5
    idx = np.flatnonzero((f >= lo) & (f <= hi) & (f > 0))
    if idx.size == 0:                  # 再退化：取非零频点的中间一段
        nz = np.flatnonzero(f > 0)
        if nz.size == 0:
            raise ValueError("频率轴没有正频点")
        a = nz[max(0, nz.size // 4)]
        b = nz[max(0, (3 * nz.size) // 4)]
        idx = np.arange(a, max(b, a) + 1)
    if max_freqs is not None and idx.size > int(max_freqs):
        sel = np.linspace(0, idx.size - 1, int(max_freqs)).round().astype(int)
        idx = idx[np.unique(sel)]
    return idx


def _cross_spectral(Xf: np.ndarray, freq_idx: np.ndarray) -> np.ndarray:
    """阵列协方差 R(f) = (1/K) Σ_k Xf[:,k,f] Xf[:,k,f]^H -> (F,M,M)。"""
    Xs = Xf[:, :, freq_idx]                       # (M,K,F)
    K = max(int(Xs.shape[1]), 1)
    R = np.einsum("mkf,nkf->fmn", Xs, Xs.conj(), optimize=True) / K
    return np.asarray(R, dtype=np.complex128)


class _SteeringBank:
    """按频点缓存导向矩阵 A_f (G,M)。

    内存够就整体缓存（避免在 PSF 分块 / 多方法对比里反复算 exp）；
    超过 `max_bytes` 就不缓存、每次现算（G*M 次 exp 很便宜，代价可接受）。
    """

    __slots__ = ("mics", "freqs", "dirs", "c", "_cache")

    def __init__(self, mics, freqs, dirs, c=C_DEFAULT, max_bytes: int = 192 << 20):
        self.mics = np.asarray(mics, float)
        self.freqs = np.atleast_1d(np.asarray(freqs, float))
        self.dirs = np.asarray(dirs, float)
        self.c = float(c)
        nbytes = self.freqs.size * self.dirs.shape[0] * self.mics.shape[0] * 16
        self._cache = {} if nbytes <= int(max_bytes) else None

    def get(self, i: int) -> np.ndarray:
        if self._cache is not None:
            a = self._cache.get(i)
            if a is None:
                a = st.steering_grid(self.mics, float(self.freqs[i]),
                                     self.dirs, self.c)[0]
                self._cache[i] = a
            return a
        return st.steering_grid(self.mics, float(self.freqs[i]), self.dirs, self.c)[0]


def _norm2(A: np.ndarray) -> np.ndarray:
    """||a||^2（远场为 M）。"""
    return np.einsum("gm,gm->g", A.conj(), A).real


def _weight_matrix(A: np.ndarray, weights: str = "ds") -> np.ndarray:
    """导向矩阵 -> 复权重 W (G,M)。

    * 'ds' / 'uniform' / 'delay-sum'：w = a / ||a||^2，约定 **w^H a = 1**
      （单位增益、单位功率源在同一点给出 1.0 的脏图响应）；
    * 'raw' / 'matched'：w = a（匹配滤波，增益 M，旁瓣与 ds 相同）。
    远场下两者只差常数 M^2，归一化脏图完全一致。
    """
    w = str(weights).lower()
    if w in ("ds", "delay-sum", "delay_and_sum", "delayandsum", "uniform", "bartlett"):
        d = np.maximum(_norm2(A), _EPS)
        return A / d[:, None]
    if w in ("raw", "matched", "sum", "none"):
        return A
    raise ValueError(f"未知的 weights: {weights}（可选 ds / uniform / raw）")


def _dirty_power(R: np.ndarray, bank: _SteeringBank, weights: str = "ds",
                 weights_cache: dict | None = None) -> np.ndarray:
    """脏图功率 P(g) = (1/F) Σ_f w_f(g)^H R_f w_f(g) -> (G,)。

    频域延迟求和：每个频点独立做共轭导向加权，再对所有频点/帧的功率求和。
    这是 Bartlett（常规）波束成形，主瓣宽度由阵列孔径与频率决定。
    """
    G = bank.dirs.shape[0]
    P = np.zeros(G, dtype=np.float64)
    for i in range(bank.freqs.size):
        A = bank.get(i)
        W = None if weights_cache is None else weights_cache.get(i)
        if W is None:
            W = _weight_matrix(A, weights)
            if weights_cache is not None:
                weights_cache[i] = W
        # w^H R w = Σ_mn conj(W[g,m]) R[m,n] W[g,n]
        P += np.einsum("gm,mg->g", W.conj(), R[i] @ W.T, optimize=True).real
    return P / max(bank.freqs.size, 1)


# ==========================================================================
# 3. 点扩散函数 PSF
# ==========================================================================
def _psf_block(mics, freqs, steer_dirs, src_dirs, c) -> np.ndarray:
    """PSF 块 (len(steer_dirs), len(src_dirs))。

    定义（归一化相干）：

        PSF[i, j] = (1/F) Σ_f  |a_i^H a_j|^2 / (||a_i||^2 ||a_j||^2)

    物理含义：单位功率点源位于 j 时，指向 i 的常规波束成形器输出。
    该定义把对角**精确**归一化为 1：|a_i^H a_i|^2/(||a_i||^2||a_i||^2) = 1，
    因此与"归一化脏图（max=1）"的尺度严格自洽——DAMAS 直接解 dirty = PSF·q。
    远场（||a||^2 = M）下它与 'ds' 权重的 PSF 完全相同，与权重选择无关。
    """
    steer_dirs = np.asarray(steer_dirs, float)
    src_dirs = np.asarray(src_dirs, float)
    freqs = np.atleast_1d(np.asarray(freqs, float))
    out = np.zeros((steer_dirs.shape[0], src_dirs.shape[0]), dtype=np.float64)
    for k in range(freqs.size):
        As = st.steering_grid(mics, float(freqs[k]), steer_dirs, c)[0]     # (Bs,M)
        Ag = st.steering_grid(mics, float(freqs[k]), src_dirs, c)[0]       # (Bg,M)
        inner = As.conj() @ Ag.T                                           # a_i^H a_j
        ns = np.maximum(_norm2(As), _EPS)[:, None]
        ng = np.maximum(_norm2(Ag), _EPS)[None, :]
        out += (np.abs(inner) ** 2) / (ns * ng)
    return out / max(freqs.size, 1)


class PointSpreadFunction:
    """PSF 矩阵 (G_steer, G_src) 的**惰性**实现。

    为什么不用一个普通 ndarray：
      * G 上千时 G^2 会到 GB 量级，GUI 只需要"某几列 / 某一行 / 峰值附近的小块"；
      * DAMAS 只需要按源下标取列（`psf_fn(j)`），配合分块计算就不必存全矩阵。

    用法：
        psf = point_spread_function(mics, freqs, grid)     # 返回本对象
        psf[i, j]           # 单个元素
        psf[:, j]           # 第 j 列（DAMAS 的 psf_fn）
        psf[i, :]           # 第 i 行
        psf.columns([3, 5]) # 多列
        psf.matrix()        # 显式稠密化（分块，峰值内存 O(chunk*G)）
        np.asarray(psf)     # 等价于 matrix()

    内部按 steer 方向分块（块大小按元素数自适应），因此建全矩阵时的
    峰值内存是 chunk × G_src，而不是 G_steer × G_src 的中间临时量。
    """

    def __init__(self, mics, freqs, src_dirs, steer_dirs=None, c=C_DEFAULT,
                 weights: str = "ds", chunk_rows: int | None = None,
                 max_bytes: int = 192 << 20, shape=None, az_axis=None,
                 el_axis=None, name: str = "psf"):
        _weight_matrix(np.zeros((1, 1)), weights)      # 提前校验 weights 字符串
        self.mics = _check_mics(mics)
        self.freqs = np.atleast_1d(np.asarray(freqs, float))
        self.src_dirs = np.asarray(src_dirs, float)
        self.steer_dirs = self.src_dirs if steer_dirs is None \
            else np.asarray(steer_dirs, float)
        self.c = float(c)
        self.weights = str(weights)
        self.shape = (self.steer_dirs.shape[0], self.src_dirs.shape[0])
        self.name = name
        self.shape_src = tuple(shape) if shape is not None else (self.src_dirs.shape[0],)
        self.az_axis = az_axis
        self.el_axis = el_axis
        if chunk_rows is None:
            # 分块大小按元素数自适应：块内临时量控制在 ~16 MB 量级
            chunk_rows = int(max(1, min(self.shape[0], np.ceil(2.0e6 / max(self.shape[1], 1)))))
        self.chunk_rows = int(max(1, chunk_rows))
        self._src_bank = _SteeringBank(self.mics, self.freqs, self.src_dirs,
                                       self.c, max_bytes=max_bytes)

    # ---------------- 基本属性 ----------------
    @property
    def T(self):
        return PointSpreadFunction(self.mics, self.freqs, self.steer_dirs,
                                   self.src_dirs, self.c, self.weights,
                                   self.chunk_rows, name=self.name + ".T")

    def __repr__(self) -> str:
        return (f"PointSpreadFunction(shape={self.shape}, F={self.freqs.size}, "
                f"M={self.mics.shape[0]}, weights='{self.weights}')")

    def __len__(self) -> int:
        return self.shape[0]

    # ---------------- 计算 ----------------
    def _block(self, steer_idx, src_idx) -> np.ndarray:
        si = np.atleast_1d(np.asarray(steer_idx, int))
        gi = np.atleast_1d(np.asarray(src_idx, int))
        if si.size == 0 or gi.size == 0:
            return np.zeros((si.size, gi.size))
        return _psf_block(self.mics, self.freqs, self.steer_dirs[si],
                          self.src_dirs[gi], self.c)

    def columns(self, idx) -> np.ndarray:
        """按源下标取列 -> (G_steer, len(idx))。DAMAS 的 `psf_fn` 就用它。"""
        gi = np.atleast_1d(np.asarray(idx, int))
        out = np.zeros((self.shape[0], gi.size), dtype=np.float64)
        for k in range(self.freqs.size):
            As = st.steering_grid(self.mics, float(self.freqs[k]),
                                  self.steer_dirs, self.c)[0]
            Ag = self._src_bank.get(k)[gi]
            inner = As.conj() @ Ag.T
            ns = np.maximum(_norm2(As), _EPS)[:, None]
            ng = np.maximum(_norm2(Ag), _EPS)[None, :]
            out += (np.abs(inner) ** 2) / (ns * ng)
        return out / max(self.freqs.size, 1)

    def rows(self, idx) -> np.ndarray:
        """按 steer 下标取行 -> (len(idx), G_src)。"""
        si = np.atleast_1d(np.asarray(idx, int))
        out = np.zeros((si.size, self.shape[1]), dtype=np.float64)
        for k in range(self.freqs.size):
            As = st.steering_grid(self.mics, float(self.freqs[k]),
                                  self.steer_dirs[si], self.c)[0]
            Ag = self._src_bank.get(k)
            inner = As.conj() @ Ag.T
            ns = np.maximum(_norm2(As), _EPS)[:, None]
            ng = np.maximum(_norm2(Ag), _EPS)[None, :]
            out += (np.abs(inner) ** 2) / (ns * ng)
        return out / max(self.freqs.size, 1)

    def matrix(self, max_elements: int | None = None,
               chunk_rows: int | None = None) -> np.ndarray:
        """分块算出完整 (G_steer, G_src) 矩阵。"""
        n = self.shape[0] * self.shape[1]
        if max_elements is not None and n > int(max_elements):
            raise ValueError(
                f"PSF 全矩阵需要 {n} 个元素（超过 max_elements={max_elements}）；"
                "请改用 psf[:, j] / psf.columns(...) 按需取列")
        chunk = int(chunk_rows or self.chunk_rows)
        out = np.empty(self.shape, dtype=np.float64)
        for i0 in range(0, self.shape[0], chunk):
            i1 = min(i0 + chunk, self.shape[0])
            si = np.arange(i0, i1)
            for k in range(self.freqs.size):
                As = st.steering_grid(self.mics, float(self.freqs[k]),
                                      self.steer_dirs[si], self.c)[0]
                Ag = self._src_bank.get(k)
                inner = As.conj() @ Ag.T
                ns = np.maximum(_norm2(As), _EPS)[:, None]
                ng = np.maximum(_norm2(Ag), _EPS)[None, :]
                blk = (np.abs(inner) ** 2) / (ns * ng)
                if k == 0:
                    out[i0:i1] = blk
                else:
                    out[i0:i1] += blk
        out /= max(self.freqs.size, 1)
        return out

    def diagonal(self) -> np.ndarray:
        """主对角——按定义恒为 1（|a^H a|^2 / (||a||^2||a||^2) = 1）。"""
        return np.ones(min(self.shape), dtype=np.float64)

    def column_fn(self):
        """返回 `j -> psf[:, j]` 的回调，可直接喂给 `damas(psf_fn=...)`。"""
        return lambda j: self.columns([int(j)])[:, 0]

    def dense(self, **kw) -> np.ndarray:
        return self.matrix(**kw)

    # ---------------- 索引 / 转换 ----------------
    def _idx(self, key, n):
        if isinstance(key, slice):
            return np.arange(*key.indices(n))
        if isinstance(key, (int, np.integer)):
            return np.array([int(key)])
        return np.atleast_1d(np.asarray(key, int)).ravel()

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        if len(key) == 1:
            key = (key[0], slice(None))
        if len(key) != 2:
            raise IndexError("PSF 只支持二维索引")
        rk, ck = key
        r_scalar = isinstance(rk, (int, np.integer))
        c_scalar = isinstance(ck, (int, np.integer))
        ri = self._idx(rk, self.shape[0])
        ci = self._idx(ck, self.shape[1])
        blk = self._block(ri, ci)
        if r_scalar and c_scalar:
            return float(blk[0, 0])
        if r_scalar:
            return blk[0]
        if c_scalar:
            return blk[:, 0]
        return blk

    def __array__(self, dtype=None, copy=None):
        a = self.matrix()
        if dtype is not None:
            a = a.astype(dtype)
        return a


def point_spread_function(mics, freqs, grid_dirs, steer_dirs=None, c: float = C_DEFAULT,
                          weights: str = "ds", chunk_rows: int | None = None,
                          max_bytes: int = 192 << 20, dense: bool = False):
    """点扩散函数 PSF。

    参数
    ----
    mics      : (M,3) 阵元坐标（阵列坐标系，xy 平面）
    freqs     : (F,) 频点（Hz）。**建议直接传 dirty_map(...)['freqs_used']**，
                保证 PSF 与脏图用的是同一批频点。
    grid_dirs : 源网格（声源可能出现的网格）——`direction_grid()` 的返回值或 (G,3)
    steer_dirs: 导向网格（波束扫描网格），None 表示与 grid_dirs 相同（方阵 PSF）
    weights   : 'ds'（默认）/ 'uniform' / 'raw'，见 `_weight_matrix`
    dense     : True 时直接返回 (G_steer, G_src) ndarray

    返回
    ----
    `PointSpreadFunction` 对象（惰性、分块，可按列/行/块取用；`np.asarray` 可稠密化）。
    内存提示：G=2000 时全矩阵约 32 MB 尚可；G=10000 时请只用 `psf[:, j]`。
    """
    src_dirs, shape, az, el = _parse_grid(grid_dirs)
    if steer_dirs is None:
        sd, sshape, saz, sel = src_dirs, shape, az, el
    else:
        sd, sshape, saz, sel = _parse_grid(steer_dirs)
    psf = PointSpreadFunction(mics, freqs, src_dirs, sd, c=c, weights=weights,
                              chunk_rows=chunk_rows, max_bytes=max_bytes,
                              shape=shape, az_axis=az, el_axis=el)
    if dense:
        return psf.matrix()
    return psf


def psf_columns(mics, freqs, grid_dirs, src_idx, steer_dirs=None,
                c: float = C_DEFAULT, weights: str = "ds") -> np.ndarray:
    """**按需**算若干 PSF 列 -> (G_steer, len(src_idx))。

    这是 O(G) 内存的入口：DAMAS 只需要源下标 j 对应的一列，
    不必把 G×G 全矩阵建出来。
    """
    psf = point_spread_function(mics, freqs, grid_dirs, steer_dirs, c, weights)
    return psf.columns(src_idx)


def psf_column(mics, freqs, grid_dirs, src_idx, steer_dirs=None,
               c: float = C_DEFAULT, weights: str = "ds") -> np.ndarray:
    """单列 PSF -> (G_steer,)。"""
    return psf_columns(mics, freqs, grid_dirs, [int(src_idx)], steer_dirs,
                       c, weights)[:, 0]


# ==========================================================================
# 4. 成像公共前置：STFT + 协方差 + 导向缓存
# ==========================================================================
def _imaging_setup(X, fs, mics, grid_dirs, freqs=None, n_fft: int = 512,
                   hop=None, c: float = C_DEFAULT, weights: str = "ds",
                   max_freqs: int | None = 64, f_min=None, f_max=None,
                   win: str = "blackmanharris", win_table=None,
                   max_steer_bytes: int = 192 << 20) -> dict:
    """把 STFT / 频点选择 / 协方差 / 导向缓存一次做好，供所有成像方法复用。

    这样 `compare_methods` 里 5 种方法共用同一份 R(f)，不会重复算 STFT。
    """
    X = _as_multichannel(X)
    mics = _check_mics(mics)
    if X.shape[0] != mics.shape[0]:
        raise ValueError(f"X 的通道数 {X.shape[0]} 与 mics 的个数 {mics.shape[0]} 不一致")
    dirs, shape, az, el = _parse_grid(grid_dirs)
    Xf, freqs_axis = dsp.stft(X, int(fs), int(n_fft), hop, win=win, win_table=win_table)
    idx = _select_freqs(freqs_axis, mics, float(c), freqs=freqs,
                        max_freqs=max_freqs, f_min=f_min, f_max=f_max)
    R = _cross_spectral(Xf, idx)
    used = np.asarray(freqs_axis[idx], float)
    bank = _SteeringBank(mics, used, dirs, float(c), max_bytes=max_steer_bytes)
    return {
        "X": X, "Xf": Xf, "fs": int(fs), "n_fft": int(n_fft),
        "hop": int(hop or (int(n_fft) // 2)), "mics": mics, "c": float(c),
        "dirs": dirs, "shape": shape, "az_axis": az, "el_axis": el,
        "freq_idx": idx, "freqs_used": used, "R": R, "bank": bank,
        "weights": str(weights), "n_frames": int(Xf.shape[1]),
        "n_channels": int(X.shape[0]),
    }


def _pack_map(flat, S, method: str, extra: dict | None = None) -> dict:
    """统一打包：扁平数组 + shape + 轴 + 峰值信息 + 指标，且全部有限。"""
    norm, idx, pk = _normalize_map(flat)
    out = {
        "method": method,
        "map": norm,
        "shape": tuple(S["shape"]),
        "az_axis": None if S["az_axis"] is None else np.asarray(S["az_axis"], float),
        "el_axis": None if S["el_axis"] is None else np.asarray(S["el_axis"], float),
        "freqs_used": np.asarray(S["freqs_used"], float),
        "dirs": S["dirs"],
        "peak_index": int(idx),
        "peak_value": _finite(pk),
        "n_freqs": int(S["freqs_used"].size),
        "n_channels": int(S["n_channels"]),
        "weights": S["weights"],
        "n_fft": int(S["n_fft"]),
        "hop": int(S["hop"]),
        "n_frames": int(S["n_frames"]),
    }
    if idx >= 0:
        az, el = ac.angles_from_dir(S["dirs"][idx])
        out["peak_az_el"] = [_finite(az), _finite(el)]
        out["peak_dir"] = [float(v) for v in S["dirs"][idx]]
        out["metrics"] = map_metrics(norm, S["shape"], S["dirs"],
                                     S["az_axis"], S["el_axis"], peak_index=idx)
    else:
        out["peak_az_el"] = [0.0, 0.0]
        out["peak_dir"] = [0.0, 0.0, 1.0]
        out["metrics"] = map_metrics(norm, S["shape"], S["dirs"],
                                     S["az_axis"], S["el_axis"])
    if extra:
        out.update(extra)
    return out


# ==========================================================================
# 5. 脏图（常规波束成形）
# ==========================================================================
def dirty_map(X, fs, mics, grid_dirs, freqs=None, n_fft: int = 512, hop=None,
              c: float = C_DEFAULT, weights: str = "ds", max_freqs: int | None = 64,
              f_min=None, f_max=None, win: str = "blackmanharris",
              win_table=None, return_cov: bool = False) -> dict:
    """常规波束成形（频域延迟求和）脏图。

    原理：对每个频点 f，把各通道谱按导向矢量对齐相加（权 w_f(g) = a/||a||²，
    即 w^H a = 1），得到该方向的输出谱；再对所有频点与所有帧取功率平均：

        P(g) = (1/F) Σ_f (1/K) Σ_k | w_f(g)^H Xf[:,k,f] |²
             = (1/F) Σ_f w_f(g)^H R(f) w_f(g)

    用 R(f) 而非逐帧求和，是把 K 次外积折叠成一次矩阵乘法（K 帧平均）。

    返回 dict
    --------
    map        : (G,) 扁平，归一化 max = 1（无 NaN/Inf）
    shape      : (n_el, n_az)（一维网格时为 (G,)）
    az_axis/el_axis : 角度轴（度，契约第 2 节）
    freqs_used : (F_sel,) 参与成像的频点（Hz）
    power      : (G,) 未归一化的原始功率（相对电平，可做绝对比较）
    peak_index / peak_az_el / peak_dir / metrics
    其余：n_fft / hop / n_frames / n_channels / weights
    return_cov=True 时额外返回 `R`（(F,M,M) 复协方差）与 `freq_idx`，
    供 DAMAS / CLEAN-SC / 正交波束成形复用（这些字段不可 JSON 序列化）。
    """
    S = _imaging_setup(X, fs, mics, grid_dirs, freqs=freqs, n_fft=n_fft, hop=hop,
                       c=c, weights=weights, max_freqs=max_freqs,
                       f_min=f_min, f_max=f_max, win=win, win_table=win_table)
    power = _dirty_power(S["R"], S["bank"], weights)
    out = _pack_map(power, S, "dirty")
    out["power"] = _sanitize(power)
    if return_cov:
        out["R"] = S["R"]
        out["freq_idx"] = S["freq_idx"]
        out["dirs"] = S["dirs"]
    return out


# ==========================================================================
# 6. DAMAS 反卷积
# ==========================================================================
def _prefer_f_order(P: np.ndarray, copy_limit_bytes: int = 64 << 20) -> np.ndarray:
    """如果代价可接受就转成 Fortran 序（列连续，DAMAS 内层循环快数倍）。

    大矩阵不转：`asfortranarray` 会复制一份，峰值内存翻倍，得不偿失。
    """
    P = np.asarray(P, dtype=np.float64)
    if P.flags.f_contiguous:
        return P
    if P.size * 8 <= int(copy_limit_bytes):
        return np.asfortranarray(P)
    return P


class _PSFAccessor:
    """把多种 PSF 输入统一成「按列访问」+「可选稠密矩阵」。
    支持：
      * (G,G) ndarray
      * `PointSpreadFunction` 对象（用 .columns / .matrix）
      * 回调 `psf_fn(j) -> (G,)`（或 `psf_fn([j1,j2]) -> (G,2)`）

    内存策略：G^2*8 <= max_cache_bytes 时把列懒加载成稠密矩阵（DAMAS 快路径）；
    否则用有界 LRU 缓存，超限就丢列重算——慢但绝不会爆内存。

    布局策略：DAMAS 的内层循环是 `r -= P[:, j] * dq`（按列），因此稠密矩阵
    统一转成 **Fortran 序**，让列连续。G=5000 时这一项就有 ~3.5 倍差距
    （C 序列步长 40 KB，每列都跨越 5000 条 cache line）。
    """

    def __init__(self, psf_fn, G: int, max_cache_bytes: int = _MAX_CACHE_BYTES):
        self.G = int(G)
        self.max_cache_bytes = int(max_cache_bytes)
        self.dense: np.ndarray | None = None
        self._fn = None
        self._vec_fn = None
        self._lru: OrderedDict = OrderedDict()
        self._diag: np.ndarray | None = None
        if psf_fn is None:
            raise ValueError("damas 需要 psf_fn（回调 / PointSpreadFunction / (G,G) 矩阵）")
        if isinstance(psf_fn, np.ndarray):
            P = np.asarray(psf_fn, float)
            if P.shape != (G, G):
                raise ValueError(f"PSF 矩阵形状 {P.shape} 与脏图长度 {G} 不匹配")
            self.dense = _prefer_f_order(P)
            return
        if hasattr(psf_fn, "matrix") and hasattr(psf_fn, "shape"):
            gs, gg = psf_fn.shape
            if (gs, gg) != (G, G):
                raise ValueError(f"PSF 形状 {(gs, gg)} 与脏图长度 {G} 不匹配")
            if G * G * 8 <= self.max_cache_bytes:
                self.dense = _prefer_f_order(np.asarray(psf_fn.matrix(), float))
                return
            self._vec_fn = lambda idx: np.asarray(psf_fn.columns(idx), float)
            return
        if callable(psf_fn):
            self._fn = psf_fn
            # 探测是否支持向量化下标（成功就成批取列，省 Python 开销）
            try:
                probe = np.asarray(psf_fn(np.array([0], dtype=int)))
                if probe.ndim == 2 and probe.shape[0] == G:
                    self._vec_fn = lambda idx: np.asarray(psf_fn(np.asarray(idx, int)), float)
            except Exception:
                self._vec_fn = None
            return
        raise TypeError("psf_fn 必须是回调、PointSpreadFunction 或 (G,G) 矩阵")

    def dense_matrix(self) -> np.ndarray:
        """尽可能给出稠密矩阵；超出内存预算返回 None。

        注意：回调路径必须**成批**取列。逐列调用 `psf_fn(j)` 在按需实现下
        每次都要重算整个导向矩阵（G*M*F 次 exp），G 上千时会慢几十倍。
        """
        if self.dense is not None:
            return self.dense
        if self.G * self.G * 8 > self.max_cache_bytes:
            return None
        P = np.empty((self.G, self.G), dtype=np.float64, order="F")
        if self._vec_fn is not None:
            step = int(max(1, min(self.G, np.ceil(2.0e6 / max(self.G, 1)))))
            for j0 in range(0, self.G, step):
                j1 = min(j0 + step, self.G)
                blk = np.asarray(self._vec_fn(np.arange(j0, j1)), float)
                if blk.shape != (self.G, j1 - j0):
                    raise ValueError(
                        f"PSF 回调返回形状 {blk.shape}，期望 {(self.G, j1 - j0)}")
                P[:, j0:j1] = blk
        else:
            for j in range(self.G):
                P[:, j] = self.col(j)
        self.dense = P
        return P

    def col(self, j: int) -> np.ndarray:
        j = int(j)
        if self.dense is not None:
            return self.dense[:, j]
        c = self._lru.get(j)
        if c is not None:
            self._lru.move_to_end(j)
            return c
        if self._vec_fn is not None:
            c = self._vec_fn(np.array([j]))[:, 0]
        else:
            c = np.asarray(self._fn(j), float).reshape(-1)
        if c.size != self.G:
            raise ValueError(f"PSF 回调返回的长度 {c.size} != 脏图长度 {self.G}")
        max_cols = max(1, self.max_cache_bytes // max(self.G * 8, 1))
        self._lru[j] = c
        while len(self._lru) > max_cols:
            self._lru.popitem(last=False)
        return c

    def diag(self) -> np.ndarray:
        if self.dense is not None:
            return np.diag(self.dense).copy()
        if self._diag is None:
            self._diag = np.array([float(self.col(j)[j]) for j in range(self.G)])
        return self._diag


def damas(dirty, psf_fn=None, n_iter: int = 200, relax: float = 1.0,
          nonneg: bool = True, tol: float = 1e-6, shape=None,
          normalize: bool = True, block_size: int = 1,
          max_cache_bytes: int = _MAX_CACHE_BYTES,
          keep_history: bool = True) -> dict:
    """DAMAS 反卷积（Deconvolution Approach for the Mapping of Acoustic Sources）。

    算法
    ----
    脏图是 PSF 与真实源分布 q 的卷积（离散线性系统）：

        dirty = PSF · q          PSF[i,j] = 单位功率源 j 在网格 i 的响应

    DAMAS 就是解这个非负线性方程组。Brooks & Humphreys 用 Gauss-Seidel 逐点
    更新，本实现沿用：

        r = dirty - PSF·q                       （残差，始终显式维护）
        q_j ← q_j + relax · r_j / PSF[j,j]      （第 j 点单独求解）
        r   ← r - PSF[:,j] · Δq_j               （秩一更新，O(G) 而不是重算矩阵乘）

    数值处理
    --------
    * **非负约束**：`nonneg=True` 时把负的 q_j 截断到 0（声源功率非负），
      这一步也是 DAMAS 能"压旁瓣"的关键——旁瓣处的负更新被钉在 0。
    * **松弛** `relax<1` 可提高鲁棒性（PSF 病态/有失配时避免振荡）。
    * **提前停止**：每轮比较 `||q_new - q_old|| / ||q_old|| < tol`（默认 1e-6），
      达到即认为收敛；同时记录相对残差 `||r||/||dirty||`。
    * **数值保护**：PSF 对角被下限截断（避免除零）；一旦出现非有限值立即
      停止并返回上一轮有限解；输出做 nan/inf 清洗，仍然可能为负（nonneg=False）。
    * `block_size>1` 时用块 Gauss-Seidel（块内 Jacobi、块间顺序更新），
      定点相同、BLAS 友好，G 很大时可显著加速。

    参数
    ----
    dirty    : (G,) 脏图，或 `dirty_map(...)` 的返回 dict（自动取 'map'/'shape'）
    psf_fn   : 回调 `j -> PSF[:,j]`，或 `PointSpreadFunction`，(G,G) 矩阵
    shape    : 输出 shape；给 dict 时自动继承
    normalize: 输出 map 归一化到 max=1（`raw`/`q` 字段保留原始解）

    返回 dict
    --------
    map / q / raw : (G,) 解（map 已归一化，raw 为原始尺度）
    shape / az_axis / el_axis / freqs_used / metrics / peak_*
    history       : {'iter', 'residual', 'delta', 'peak'} 列表，残差随迭代
    n_iter / converged / stop_reason / psf_dense / runtime_s
    residual_map  : dirty - PSF·q（(G,)）
    """
    t0 = time.perf_counter()
    az = el = None
    freqs_used = None
    if isinstance(dirty, dict):
        base = dirty
        d = np.asarray(base.get("map", base.get("power")), float).reshape(-1)
        shape = base.get("shape", shape)
        az, el = base.get("az_axis"), base.get("el_axis")
        freqs_used = base.get("freqs_used")
    else:
        d = np.asarray(dirty, float).reshape(-1)
    d = _sanitize(d)
    G = int(d.size)
    if G == 0:
        raise ValueError("dirty 为空")
    if shape is None:
        shape = (G,)

    acc = _PSFAccessor(psf_fn, G, max_cache_bytes=max_cache_bytes)
    P = acc.dense_matrix()
    psf_dense = P is not None

    diag = _sanitize(acc.diag(), 1.0)
    dref = max(float(np.max(np.abs(diag))), _EPS)
    inv_diag = 1.0 / np.maximum(np.abs(diag), 1e-6 * dref)   # 防除零
    dnorm = float(np.linalg.norm(d)) or 1.0

    q = np.zeros(G, dtype=np.float64)
    r = d.copy()
    q_prev = np.zeros(G, dtype=np.float64)
    buf = np.empty(G, dtype=np.float64)
    history = {"iter": [], "residual": [], "delta": [], "peak": []}
    converged = False
    stop_reason = "max_iter"
    q_last = q.copy()
    bs = int(max(1, block_size))

    for it in range(int(max(1, n_iter))):
        q_prev[:] = q
        if P is not None:
            if bs <= 1:
                for j in range(G):
                    dq = relax * r[j] * inv_diag[j]
                    if nonneg:
                        v = q[j] + dq
                        if v < 0.0:
                            v = 0.0
                        dq = v - q[j]
                        q[j] = v
                    else:
                        q[j] += dq
                    if dq != 0.0:
                        np.multiply(P[:, j], dq, out=buf)
                        np.subtract(r, buf, out=r)
            else:
                for j0 in range(0, G, bs):
                    j1 = min(j0 + bs, G)
                    dq = relax * r[j0:j1] * inv_diag[j0:j1]
                    if nonneg:
                        dq = np.maximum(q[j0:j1] + dq, 0.0) - q[j0:j1]
                    q[j0:j1] += dq
                    if np.any(dq != 0.0):
                        np.subtract(r, P[:, j0:j1] @ dq, out=r)
        else:
            for j in range(G):
                col = acc.col(j)
                dq = relax * r[j] * inv_diag[j]
                if nonneg:
                    v = q[j] + dq
                    if v < 0.0:
                        v = 0.0
                    dq = v - q[j]
                    q[j] = v
                else:
                    q[j] += dq
                if dq != 0.0:
                    np.multiply(col, dq, out=buf)
                    np.subtract(r, buf, out=r)

        res = float(np.linalg.norm(r)) / dnorm
        nq = float(np.linalg.norm(q))
        delta = float(np.linalg.norm(q - q_prev)) / (nq if nq > _EPS else _EPS)
        if not (np.isfinite(res) and np.isfinite(delta) and np.isfinite(nq)):
            stop_reason = "diverged"
            q = q_last.copy()
            break
        q_last = q.copy()
        if keep_history:
            history["iter"].append(it)
            history["residual"].append(res)
            history["delta"].append(delta)
            history["peak"].append(float(np.max(q)))
        if delta < tol:
            converged = True
            stop_reason = "tol"
            break

    q = _sanitize(q)
    norm_map, idx, pk = _normalize_map(q) if normalize else (q, int(np.argmax(q)), float(np.max(q)))
    # 残差图：迭代里维护的 r 就是 dirty - PSF·q（数学上恒等，秩一更新保证），
    # 直接复用，避免再花一次 O(G^2)。只有发散回退时才需要重算。
    if stop_reason == "diverged" and P is not None:
        r_final = _sanitize(d - P @ q)
    else:
        r_final = _sanitize(r)

    out = {
        "method": "damas",
        "map": _sanitize(norm_map),
        "raw": q,
        "q": q,
        "shape": tuple(int(v) for v in np.atleast_1d(shape)),
        "az_axis": None if az is None else np.asarray(az, float),
        "el_axis": None if el is None else np.asarray(el, float),
        "freqs_used": None if freqs_used is None else np.asarray(freqs_used, float),
        "history": history,
        "n_iter": int(len(history["iter"])),
        "n_iter_max": int(n_iter),
        "converged": bool(converged),
        "stop_reason": stop_reason,
        "residual": _finite(history["residual"][-1] if history["residual"] else 1.0, 1.0),
        "residual_map": r_final,
        "peak_index": int(idx),
        "peak_value": _finite(pk),
        "nonneg": bool(nonneg),
        "relax": float(relax),
        "psf_dense": bool(psf_dense),
        "runtime_s": float(time.perf_counter() - t0),
    }
    if idx >= 0 and isinstance(dirty, dict) and "dirs" in dirty:
        azp, elp = ac.angles_from_dir(dirty["dirs"][idx])
        out["peak_az_el"] = [_finite(azp), _finite(elp)]
    if isinstance(dirty, dict):
        out["metrics"] = map_metrics(out["map"], out["shape"], dirty.get("dirs"),
                                     az, el, peak_index=idx)
        for k in ("dirs", "n_fft", "hop", "n_frames", "n_channels", "weights"):
            if k in dirty:
                out.setdefault(k, dirty[k])
    return out


# ==========================================================================
# 7. CLEAN-SC
# ==========================================================================
def _clean_beam(dirs: np.ndarray, u0: np.ndarray, fwhm_deg: float) -> np.ndarray:
    """干净束：以 u0 为中心、FWHM=fwhm_deg 的高斯核（球面夹角）。

    CLEAN 类算法剥离源分量后要用一个"理想波束"把源放回去。这里用高斯核：
    宽度由调用方控制 —— 等于阵列主瓣时是经典 Högbom 干净束（只压旁瓣），
    小于主瓣时给出超分辨的视觉效果（分量位置本身是超分辨的，
    所以干净束取半宽在物理上更诚实）。
    """
    ang = _angular_distance_deg(dirs, u0)[:, 0]
    return np.exp(-4.0 * np.log(2.0) * (ang / max(float(fwhm_deg), 1e-6)) ** 2)


def _clean_sc_core(S: dict, n_sources: int = 3, loop_gain: float = 0.9,
                   n_iter: int = 20, stop_ratio: float = 0.01,
                   clean_beam_fwhm_deg=None, clean_beam: str = "gaussian",
                   exclude_picked: bool = True,
                   weights: str | None = None) -> dict:
    """CLEAN-SC 主体（基于已算好的协方差 R 与导向缓存）。"""
    t0 = time.perf_counter()
    dirs, shape = S["dirs"], S["shape"]
    G, F = dirs.shape[0], S["freqs_used"].size
    wname = weights or S["weights"]
    bank = S["bank"]
    R = np.array(S["R"], dtype=np.complex128, copy=True)     # (F,M,M) 会被逐步剥离
    W = [_weight_matrix(bank.get(i), wname) for i in range(F)]

    # 1) 初始脏图（未归一化功率，保留绝对电平用于记录分量功率）
    dirty = np.zeros(G)
    for i in range(F):
        dirty += np.einsum("gm,mg->g", W[i].conj(), R[i] @ W[i].T,
                           optimize=True).real
    dirty /= max(F, 1)
    p_max0 = float(np.max(dirty))
    if not np.isfinite(p_max0) or p_max0 <= _EPS:
        return {"map": np.zeros(G), "components": [], "dirty": _sanitize(dirty),
                "residual": _sanitize(dirty), "history": {"iter": [], "peak": [], "removed": []},
                "n_components": 0, "stop_reason": "empty"}

    # 干净束宽度：默认取脏图主瓣宽度的一半（下限 1.5 个网格步长）
    if clean_beam_fwhm_deg is None:
        lw = main_lobe_width(dirty / p_max0, shape, dirs, S["az_axis"], S["el_axis"])
        clean_beam_fwhm_deg = max(1.5 * _grid_step_deg(dirs), 0.5 * lw)

    clean_map = np.zeros(G)
    comp_map = np.zeros(G)
    components: list[dict] = []
    history = {"iter": [], "peak": [], "removed": [], "residual_peak": []}
    work = dirty.copy()
    stop_reason = "max_iter"
    # 最多剥离多少个分量：n_sources 是"期望声源数"，n_iter 是硬上限
    n_run = int(n_iter) if n_sources is None else int(min(max(int(n_iter), 1),
                                                         max(int(n_sources), 1)))
    n_run = max(n_run, 1)

    for k in range(n_run):
        pk_idx = int(np.argmax(work))
        pk_val = float(work[pk_idx])
        if not np.isfinite(pk_val) or pk_val <= stop_ratio * p_max0:
            stop_reason = "stop_ratio"
            break
        # 2) 该峰的波束成形输出（每频点一个"源分量向量"）：
        #    ĥ_f = R_f w_f(j*) / sqrt(P_f(j*))，单位功率点源时 ĥ_f = sqrt(π_f)·a_f。
        #    这与"直接减 PSF"的本质区别：ĥ 保留了 R 里的**相干结构**——
        #    多个相干源在同一 ĥ 里以复数叠加，剥离时按相干分量整体扣除，
        #    而不是按非相干功率扣一条固定的 PSF。
        w_star = [W[i][pk_idx] for i in range(F)]                       # (M,) 每频点
        p_f = np.array([max(float(np.einsum("m,mn,n->", w.conj(), R[i], w).real), _EPS)
                        for i, w in enumerate(w_star)])
        h = [R[i] @ w_star[i] / np.sqrt(p_f[i]) for i in range(F)]      # (M,)
        # 3) 源分量的"脏图"：P_h(g) = (1/F) Σ_f |w_f(g)^H ĥ_f|²，
        #    即各网格点与该源分量的**空间相干性**（不是 PSF 列）。
        comp = np.zeros(G)
        for i in range(F):
            comp += np.abs(W[i].conj() @ h[i]) ** 2
        comp /= max(F, 1)
        gain = float(np.clip(loop_gain, 0.0, 1.0))
        # 4) 从脏图与协方差里同时剥离（协方差减秩一：相干源去除）
        work -= gain * comp
        np.maximum(work, 0.0, out=work)
        for i in range(F):
            R[i] -= gain * np.outer(h[i], h[i].conj())
        # 5) 用干净束把分量放回干净图
        if str(clean_beam).lower() == "psf":
            beam = np.asarray(point_spread_function(
                S["mics"], S["freqs_used"], dirs, dirs, S["c"], wname
            ).columns([pk_idx])[:, 0], float)
        else:
            beam = _clean_beam(dirs, dirs[pk_idx], float(clean_beam_fwhm_deg))
        clean_map += gain * pk_val * beam
        comp_map[pk_idx] += gain * pk_val

        azp, elp = ac.angles_from_dir(dirs[pk_idx])
        components.append({
            "loop": int(k), "index": pk_idx, "az": _finite(azp), "el": _finite(elp),
            "dir": [float(v) for v in dirs[pk_idx]],
            "power": _finite(gain * pk_val),
            "power_rel": _finite(gain * pk_val / max(p_max0, _EPS)),
            "freq_power": [float(v) for v in p_f],
            "beam_fwhm_deg": _finite(clean_beam_fwhm_deg),
        })
        history["iter"].append(int(k))
        history["peak"].append(_finite(pk_val / max(p_max0, _EPS)))
        history["removed"].append(_finite(gain * pk_val / max(p_max0, _EPS)))
        history["residual_peak"].append(
            _finite(float(np.max(work)) / max(p_max0, _EPS)))
        if exclude_picked:
            work[pk_idx] = 0.0        # 同一格点不重复取
    if stop_reason == "max_iter":
        if len(components) >= n_run:
            stop_reason = "n_iter"
        elif len(components) == 0:
            stop_reason = "empty"
        else:
            stop_reason = "stop_ratio"

    return {
        "clean_map": _sanitize(clean_map),
        "components_map": _sanitize(comp_map),
        "dirty": _sanitize(dirty),
        "residual": _sanitize(work),
        "components": components,
        "history": history,
        "stop_reason": stop_reason,
        "clean_beam_fwhm_deg": _finite(clean_beam_fwhm_deg),
        "runtime_s": float(time.perf_counter() - t0),
        "n_components": len(components),
    }


def clean_sc(X, fs, mics, grid_dirs, n_sources: int = 3, loop_gain: float = 0.9,
             n_iter: int = 20, freqs=None, n_fft: int = 512, hop=None,
             c: float = C_DEFAULT, weights: str = "ds", max_freqs: int | None = 64,
             f_min=None, f_max=None, stop_ratio: float = 0.01,
             clean_beam: str = "gaussian", clean_beam_fwhm_deg=None,
             exclude_picked: bool = True, return_cov: bool = False) -> dict:
    """CLEAN-SC：基于空间相干性的迭代源剥离（Sijtsma 2007）。

    与经典 CLEAN 的区别（也就是本题强调的"相干性处理"）
    ---------------------------------------------------
    经典 CLEAN 找到脏图峰后，减去 `增益 × 峰值 × PSF列`——这是把源当成
    **非相干**点源，遇到相干声源（多个源在同一批频点上保持固定相位关系）时，
    脏图里的交叉项会形成虚假峰，减 PSF 无法正确扣除。

    CLEAN-SC 改为：用该峰的波束成形输出反推一个"源分量向量"

        ĥ_f = R_f w_f(j*) / sqrt( w_f(j*)^H R_f w_f(j*) )

    单源时 ĥ_f 正比于 sqrt(功率)·a_f；多源相干时 ĥ_f 是各源导向矢量的**复数
    叠加**，因此它天然携带了相干结构。随后：

      * 该分量的图 = 各网格点与它的相干性 `|w_f(g)^H ĥ_f|²`（而不是 PSF 列）；
      * 从协方差里扣掉秩一项 `ĥ ĥ^H`（而非扣功率）。

    这样每次剥离的是"一个相干成分"，相干源的交叉项被一起带走，剩下的图里
    另一个源的主瓣才会露出来。

    停止条件：残差峰 < `stop_ratio × 初始峰`、达到 `n_sources` 个分量
    或达到 `n_iter` 次循环，三者取先到者。

    返回 dict
    --------
    map           : (G,) 重建的干净图（用干净束把分量放回，归一化 max=1）
    components    : 每个分量 {loop, index, az, el, dir, power, power_rel, ...}
    components_map: 分量位置上的冲激图（便于 GUI 打点 / 数峰）
    dirty/residual: 初始脏图与残差图（均未归一化，已清洗）
    shape/az_axis/el_axis/freqs_used/metrics/peak_*/history/stop_reason/runtime_s
    """
    S = _imaging_setup(X, fs, mics, grid_dirs, freqs=freqs, n_fft=n_fft, hop=hop,
                       c=c, weights=weights, max_freqs=max_freqs,
                       f_min=f_min, f_max=f_max)
    core = _clean_sc_core(S, n_sources=n_sources, loop_gain=loop_gain,
                          n_iter=n_iter, stop_ratio=stop_ratio,
                          clean_beam=clean_beam,
                          clean_beam_fwhm_deg=clean_beam_fwhm_deg,
                          exclude_picked=exclude_picked)
    out = _pack_map(core["clean_map"], S, "clean_sc", extra={
        "components": core["components"],
        "n_components": core["n_components"],
        "dirty": core["dirty"] / max(float(np.max(core["dirty"])), _EPS),
        "residual": core["residual"] / max(float(np.max(core["dirty"])), _EPS),
        "components_map": core["components_map"],
        "history": core["history"],
        "stop_reason": core["stop_reason"],
        "clean_beam_fwhm_deg": core["clean_beam_fwhm_deg"],
        "runtime_s": core["runtime_s"],
        "loop_gain": float(loop_gain),
    })
    if return_cov:
        out["R"] = S["R"]
        out["dirs"] = S["dirs"]
    return out


# ==========================================================================
# 8. 正交波束成形 / 函数波束成形
# ==========================================================================
def _eig_decompose(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐频点特征分解 R_f = Σ λ_i v_i v_i^H，按 λ 降序。

    R 是 Hermitian 半正定，用 eigh（升序）后反转。M 通常 ≤ 64，
    全频点一起算是 O(F·M³)，可以忽略。
    """
    lam, vec = np.linalg.eigh(R)                 # (F,M), (F,M,M)
    lam = lam[:, ::-1]
    vec = vec[:, :, ::-1]
    lam = np.maximum(lam.real, 0.0)
    return lam, vec


def _ob_core(S: dict, n_sources: int | None = None, eig_rel_floor: float = 1e-3,
             weights: str | None = None, merge_deg: float | None = None,
             smooth_beam_deg: float | None = None) -> dict:
    """正交波束成形主体。

    做法（Sarradj 的"正交波束成形"）：
      1. 逐频点特征分解 R_f = Σ λ_i v_i v_i^H，特征向量两两正交 ——
         它们对应**互不相干**的声场成分；
      2. 对每个显著特征向量（λ_i > eig_rel_floor·λ_max）算它的波束图
         `|w_f(g)^H v_i|²`，取峰值位置，把功率 λ_i 记到该位置；
      3. 把所有 (频点, 特征向量) 的峰按角距合并成"源"，得到一张只含峰的图。

    与常规波束成形的区别：常规图里几个源的主瓣/旁瓣互相叠加，
    正交图里每个源只从自己的特征向量冒出来（旁瓣被压掉）。
    **限制**：完全相干的两个源在 R 里只占一个特征向量（秩一），
    此时正交波束成形也分不开——这与 CLEAN-SC 的限制同源。
    """
    t0 = time.perf_counter()
    dirs, shape = S["dirs"], S["shape"]
    G, F = dirs.shape[0], S["freqs_used"].size
    wname = weights or S["weights"]
    W = [_weight_matrix(S["bank"].get(i), wname) for i in range(F)]
    lam, vec = _eig_decompose(S["R"])             # (F,M), (F,M,M)

    n_eig_max = int(n_sources or lam.shape[1])
    n_eig_max = max(1, min(n_eig_max, lam.shape[1]))
    step = _grid_step_deg(dirs)
    if merge_deg is None:
        merge_deg = max(1.5 * step, 5.0)
    if smooth_beam_deg is None:
        smooth_beam_deg = max(1.5 * step, 5.0)

    peaks = []            # [(index, power, freq_idx, eig_idx)]
    lam_used = []
    for k in range(F):
        l0 = max(float(lam[k, 0]), _EPS)
        for i in range(n_eig_max):
            li = float(lam[k, i])
            if li <= eig_rel_floor * l0:
                break
            mg = np.abs(W[k].conj() @ vec[k, :, i]) ** 2
            j = int(np.argmax(mg))
            peaks.append((j, li, int(k), int(i)))
            lam_used.append(li)
    if not peaks:
        return {"map": np.zeros(G), "components": [], "n_eig": 0,
                "eigenvalues": [], "peaks": [],
                "runtime_s": float(time.perf_counter() - t0)}

    # 按角距把 (频点, 特征向量) 的峰合并成源（功率加权质心用峰所在格点近似）
    clusters: list[dict] = []
    cos_merge = np.cos(np.deg2rad(float(merge_deg)))
    for j, li, k, i in peaks:
        hit = None
        for c in clusters:
            if float(dirs[j] @ dirs[c["index"]]) >= cos_merge:
                hit = c
                break
        if hit is None:
            clusters.append({"index": j, "power": li, "n": 1,
                             "freqs": [int(k)], "eigs": [int(i)],
                             "members": [(j, li)]})
        else:
            hit["power"] += li
            hit["n"] += 1
            hit["freqs"].append(int(k))
            hit["eigs"].append(int(i))
            hit["members"].append((j, li))
            # 功率加权更新位置（在成员格点里挑累计功率最大的）
            acc: dict[int, float] = {}
            for jj, ll in hit["members"]:
                acc[jj] = acc.get(jj, 0.0) + ll
            hit["index"] = max(acc.items(), key=lambda kv: kv[1])[0]
    clusters.sort(key=lambda c: -c["power"])
    if n_sources:
        clusters = clusters[:int(n_sources)]

    total = np.zeros(G)
    comp_map = np.zeros(G)
    components = []
    for c in clusters:
        j, p = int(c["index"]), float(c["power"])
        beam = _clean_beam(dirs, dirs[j], float(smooth_beam_deg))
        total += p * beam
        comp_map[j] += p
        azp, elp = ac.angles_from_dir(dirs[j])
        components.append({"index": j, "az": _finite(azp), "el": _finite(elp),
                           "dir": [float(v) for v in dirs[j]],
                           "power": _finite(p),
                           "power_rel": _finite(p / max(float(clusters[0]["power"]), _EPS)),
                           "n_peaks": int(c["n"]),
                           "eig": int(c["eigs"][0])})
    return {"map": _sanitize(total), "components": components,
            "components_map": _sanitize(comp_map),
            "n_eig": int(n_eig_max), "n_peaks": len(peaks),
            "eigenvalues": [float(v) for v in lam_used],
            "smooth_beam_deg": _finite(smooth_beam_deg),
            "runtime_s": float(time.perf_counter() - t0)}


def orthogonal_beamforming(X, fs, mics, grid_dirs, n_sources: int | None = None,
                           freqs=None, n_fft: int = 512, hop=None,
                           c: float = C_DEFAULT, weights: str = "ds",
                           max_freqs: int | None = 64, f_min=None, f_max=None,
                           eig_rel_floor: float = 1e-3, merge_deg=None,
                           smooth_beam_deg=None) -> dict:
    """正交波束成形（Orthogonal Beamforming, Sarradj）。

    对每个频点的协方差做特征分解 R_f = Σ λ_i v_i v_i^H。特征向量两两正交，
    对应互相"去相干"的声场成分：单个点源 → 一个主特征向量；两个非相干源
    → 两个主特征向量，因此天然分离"谱上重叠"的多个源。

    每个显著特征向量的波束图 `|w_g^H v_i|²` 只取峰值位置、把特征值 λ_i 记在
    那里，最后按角距把各频点的峰合并成源。结果图是"每个源一个峰"的稀疏图，
    主瓣宽度由 `smooth_beam_deg`（默认 1.5 个网格步长）决定，而不是阵列孔径
    —— 这也是正交波束成形看起来比常规图"干净"的原因。

    限制：**完全相干**的两个源在 R_f 里只占一个特征向量（秩一），
    正交波束成形与 CLEAN-SC 一样分不开；此时要靠 DAMAS 这类基于 PSF 形状的
    反卷积（见 `damas`）。

    `n_sources` 给定则每个频点最多取前 n_sources 个特征向量、并只保留
    功率最大的 n_sources 个源。
    """
    S = _imaging_setup(X, fs, mics, grid_dirs, freqs=freqs, n_fft=n_fft, hop=hop,
                       c=c, weights=weights, max_freqs=max_freqs,
                       f_min=f_min, f_max=f_max)
    core = _ob_core(S, n_sources=n_sources, eig_rel_floor=eig_rel_floor,
                    merge_deg=merge_deg, smooth_beam_deg=smooth_beam_deg)
    out = _pack_map(core["map"], S, "orthogonal", extra={
        "components": core["components"],
        "components_map": core.get("components_map"),
        "n_components": len(core["components"]),
        "n_eig": core["n_eig"],
        "n_peaks": core.get("n_peaks", 0),
        "eigenvalues": core["eigenvalues"],
        "smooth_beam_deg": core.get("smooth_beam_deg"),
        "runtime_s": core["runtime_s"],
    })
    return out


def _fb_core(S: dict, exponent: float = 0.5, eig_rel_floor: float = 1e-6,
             weights: str | None = None) -> dict:
    """函数波束成形主体：把 R 换成 R^p（特征值取 p 次幂）再走常规波束成形。"""
    t0 = time.perf_counter()
    G, F = S["dirs"].shape[0], S["freqs_used"].size
    wname = weights or S["weights"]
    p = float(exponent)
    if not (0.0 < p <= 1.0):
        raise ValueError("functional_beamforming 的 exponent 必须落在 (0, 1]")
    lam, vec = _eig_decompose(S["R"])
    Rp = np.empty_like(S["R"])
    for k in range(F):
        lk = lam[k]
        floor = eig_rel_floor * max(float(lk[0]), _EPS)
        lk = np.where(lk > floor, lk, 0.0) ** p        # 噪声特征值直接置零
        Rp[k] = (vec[k] * lk[None, :]) @ vec[k].conj().T
    P = np.zeros(G)
    for k in range(F):
        W = _weight_matrix(S["bank"].get(k), wname)
        x = np.einsum("gm,mg->g", W.conj(), Rp[k] @ W.T, optimize=True).real
        P += np.maximum(x, 0.0) ** (1.0 / p)
    P /= max(F, 1)
    return {"map": _sanitize(P), "exponent": p,
            "runtime_s": float(time.perf_counter() - t0)}


def functional_beamforming(X, fs, mics, grid_dirs, exponent: float = 0.5,
                           freqs=None, n_fft: int = 512, hop=None,
                           c: float = C_DEFAULT, weights: str = "ds",
                           max_freqs: int | None = 64, f_min=None, f_max=None,
                           eig_rel_floor: float = 1e-6) -> dict:
    """函数波束成形（Functional Beamforming, Dougherty 2014）。

    把协方差替换成它的幂：`P(g) = [ Σ_f (w_f^H R_f^p w_f)^{1/p} ] / F`，
    特征分解后即 `R^p = Σ λ_i^p v_i v_i^H`（小特征值可置零抑制噪声）。

    `exponent = 1` 退化为常规波束成形；`exponent = 0.5`（默认）即 γ=2 的经典
    函数波束成形：单源时 `P ∝ |w^H a|^4`，相当于把 PSF 平方 —— 主瓣变窄
    （约 √2 倍）、旁瓣大幅下降、动态范围提升，代价是主瓣形状对失配更敏感。
    """
    S = _imaging_setup(X, fs, mics, grid_dirs, freqs=freqs, n_fft=n_fft, hop=hop,
                       c=c, weights=weights, max_freqs=max_freqs,
                       f_min=f_min, f_max=f_max)
    core = _fb_core(S, exponent=exponent, eig_rel_floor=eig_rel_floor)
    return _pack_map(core["map"], S, "functional", extra={
        "exponent": float(core["exponent"]),
        "runtime_s": core["runtime_s"],
    })


# ==========================================================================
# 9. 多方法对比
# ==========================================================================
def compare_methods(X, fs, mics, grid_dirs,
                    methods=("dirty", "damas", "clean_sc", "orthogonal", "functional"),
                    freqs=None, n_fft: int = 512, hop=None, c: float = C_DEFAULT,
                    weights: str = "ds", max_freqs: int | None = 64,
                    f_min=None, f_max=None, damas_kw: dict | None = None,
                    clean_kw: dict | None = None, ob_kw: dict | None = None,
                    fb_kw: dict | None = None, truth_dirs=None) -> dict:
    """对同一份数据跑多种成像方法，返回各自的 map 与对比指标。

    前置处理（STFT / 频点选择 / 协方差）只做一次，所有方法复用，
    因此对比 5 种方法的代价 ≈ 最贵的那一种 + 一点零头。

    指标（`metrics[name]`）：
      main_lobe_width_deg （等效 -3dB 主瓣宽度）、psl_db（峰值旁瓣级）、
      dynamic_range_db、n_peaks、peak_az/el、runtime_s。
    给了 `truth_dirs` 时额外给出 `resolution`（两两分辨情况）与峰值角度误差。

    返回
    ----
    {results: {name: 方法原始返回}, metrics: {name: {...}}, shape, az_axis,
     el_axis, freqs_used, truth_dirs, resolution: {name: {...}}}
    """
    t_all = time.perf_counter()
    S = _imaging_setup(X, fs, mics, grid_dirs, freqs=freqs, n_fft=n_fft, hop=hop,
                       c=c, weights=weights, max_freqs=max_freqs,
                       f_min=f_min, f_max=f_max)
    methods = [str(m).lower() for m in methods]
    results: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    resolution: dict[str, dict] = {}

    # 脏图只算一次，DAMAS / CLEAN-SC 都基于它
    dirty_power = None
    if any(m in methods for m in ("dirty", "damas")):
        dirty_power = _dirty_power(S["R"], S["bank"], weights)
    psf = None
    if "damas" in methods:
        psf = point_spread_function(S["mics"], S["freqs_used"], S["dirs"],
                                    None, S["c"], weights)

    for name in methods:
        t0 = time.perf_counter()
        if name == "dirty":
            res = _pack_map(dirty_power, S, "dirty")
            res["power"] = _sanitize(dirty_power)
        elif name == "damas":
            dm = _pack_map(dirty_power, S, "dirty")
            dm["dirs"] = S["dirs"]
            kw = dict(damas_kw or {})
            res = damas(dm, psf, **kw)
            res["method"] = "damas"
        elif name in ("clean_sc", "cleansc", "clean-sc"):
            core = _clean_sc_core(S, **(clean_kw or {}))
            res = _pack_map(core["clean_map"], S, "clean_sc", extra={
                "components": core["components"],
                "n_components": core["n_components"],
                "components_map": core["components_map"],
                "history": core["history"],
                "stop_reason": core["stop_reason"],
            })
        elif name in ("orthogonal", "ob", "orthogonal_beamforming"):
            core = _ob_core(S, **(ob_kw or {}))
            res = _pack_map(core["map"], S, "orthogonal", extra={
                "components": core["components"], "n_eig": core["n_eig"]})
        elif name in ("functional", "fb", "functional_beamforming"):
            core = _fb_core(S, **(fb_kw or {}))
            res = _pack_map(core["map"], S, "functional", extra={
                "exponent": core["exponent"]})
        else:
            raise ValueError(f"未知方法: {name}")
        res.setdefault("runtime_s", float(time.perf_counter() - t0))
        results[name] = res
        m = dict(res.get("metrics") or map_metrics(
            res["map"], S["shape"], S["dirs"], S["az_axis"], S["el_axis"]))
        m["runtime_s"] = _finite(res.get("runtime_s", 0.0))
        m["n_components"] = int(res.get("n_components", 0) or 0)
        metrics[name] = m
        if truth_dirs is not None:
            resolution[name] = resolution_check(
                res["map"], S["dirs"], S["shape"], S["az_axis"], S["el_axis"],
                truth_dirs)

    return {
        "results": results,
        "metrics": metrics,
        "resolution": resolution,
        "shape": tuple(S["shape"]),
        "az_axis": None if S["az_axis"] is None else np.asarray(S["az_axis"], float),
        "el_axis": None if S["el_axis"] is None else np.asarray(S["el_axis"], float),
        "freqs_used": np.asarray(S["freqs_used"], float),
        "dirs": S["dirs"],
        "truth_dirs": None if truth_dirs is None else np.asarray(truth_dirs, float),
        "runtime_s": float(time.perf_counter() - t_all),
    }


def to_jsonable(result: dict, drop=("R", "dirs", "Xf", "residual_map")) -> dict:
    """把成像结果转成可 `json.dumps(..., allow_nan=False)` 的纯 Python 结构。

    CONTRACT 第 8 节要求大数组用「扁平数组 + shape」传、且不得出现 NaN/Inf。
    本函数负责：ndarray → list、numpy 标量 → float/int、非有限值 → None，
    并按需丢弃内部大对象（协方差 R、方向表 dirs 等 GUI 用不到的字段）。
    """
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items() if k not in drop}
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "fc":
                v = _sanitize(v)
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, (np.floating, float)):
            f = float(v)
            return f if np.isfinite(f) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        return v

    return conv(dict(result))


__all__ = [
    "point_spread_function", "psf_columns", "psf_column", "PointSpreadFunction",
    "dirty_map", "damas", "clean_sc",
    "orthogonal_beamforming", "functional_beamforming", "compare_methods",
    "find_peaks", "main_lobe_width", "map_metrics", "resolution_check",
    "to_jsonable",
]
