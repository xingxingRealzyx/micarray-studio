"""阵列几何：拓扑构建、指标分析、JSON 序列化。

所有约定见 docs/CONTRACT.md 第 1、2 节。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Sequence

import numpy as np

C_DEFAULT = 343.0

Spacing = Literal["equal", "log", "quadratic"]
Kind = Literal[
    "ula", "uca", "grid", "spiral_arm", "vogel", "cantilever", "nested",
    "multi_ring", "custom",
]


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def pairwise_distances(mics: np.ndarray) -> np.ndarray:
    """所有阵元对的欧氏距离，长度 M*(M-1)/2。"""
    M = mics.shape[0]
    i, j = np.triu_indices(M, k=1)
    return np.linalg.norm(mics[i] - mics[j], axis=1)


def aperture(mics: np.ndarray) -> float:
    """最大阵元间距（米）。决定低频分辨率与无混叠上限。"""
    d = pairwise_distances(mics)
    return float(d.max()) if d.size else 0.0


def min_spacing(mics: np.ndarray) -> float:
    """最近阵元间距（米）。决定高频无混叠上限：f_max = c/(2*d_min)。"""
    d = pairwise_distances(mics)
    return float(d.min()) if d.size else 0.0


def grating_free_freq(mics: np.ndarray, c: float = C_DEFAULT) -> float:
    """由最近间距给出的无空间混叠上限频率（Hz）。"""
    d = min_spacing(mics)
    return float(c / (2.0 * d)) if d > 0 else float("inf")


def redundancy_stats(mics: np.ndarray, tol: float = 1e-9) -> dict:
    """间距冗余度分析。

    "非冗余阵列" 指所有阵元对间距互不相同 —— 每个间距只对应一个阵元对，
    在给定阵元数下给出最干净的空间谱。这里量化它。
    """
    d = pairwise_distances(mics)
    if d.size == 0:
        return {"n_pairs": 0, "n_unique": 0, "redundancy": 1.0,
                "max_multiplicity": 0, "coincident_pairs": 0, "histogram": []}

    # 聚类：排序后按容差分组
    ds = np.sort(d)
    groups, start = [], 0
    for k in range(1, ds.size + 1):
        if k == ds.size or ds[k] - ds[start] > tol:
            groups.append(k - start)
            start = k
    mult = np.asarray(groups)

    return {
        "n_pairs": int(d.size),
        "n_unique": int(mult.size),
        "redundancy": float(d.size / mult.size),   # 1.0 = 完全非冗余
        "max_multiplicity": int(mult.max()),
        "coincident_pairs": int((d < 1e-6).sum()),  # 阵元重合的对数
        # 供前端画直方图：间距分箱计数
        "histogram": _spacing_histogram(d, bins=48),
    }


def _spacing_histogram(d: np.ndarray, bins: int = 48) -> list[dict]:
    if d.size == 0 or d.max() <= 0:
        return []
    edges = np.linspace(0.0, d.max(), bins + 1)
    counts, _ = np.histogram(d, bins=edges)
    return [{"d": round(float(0.5 * (edges[k] + edges[k + 1])), 6),
             "count": int(counts[k])} for k in range(bins)]


def array_metrics(mics: np.ndarray, c: float = C_DEFAULT) -> dict:
    return {
        "n_mics": int(mics.shape[0]),
        "aperture": round(aperture(mics), 6),
        "min_spacing": round(min_spacing(mics), 6),
        "grating_free_freq": round(grating_free_freq(mics, c), 1),
        "centroid": [round(float(v), 6) for v in mics.mean(axis=0)],
    }


# --------------------------------------------------------------------------
# 阵元排布辅助
# --------------------------------------------------------------------------
def _arm_radii(n: int, r0: float, r1: float, spacing: Spacing) -> np.ndarray:
    """单臂上 n 个阵元的半径分布。"""
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.array([0.5 * (r0 + r1)])
    t = np.linspace(0.0, 1.0, n)
    if spacing == "equal":
        r = r0 + (r1 - r0) * t
    elif spacing == "log":
        # 对数分布：半径按几何级数增长 -> 相邻间距随半径增大，
        # 天然形成"多尺度基线"，且每对间距各不相同（非冗余）。
        r = r0 * (r1 / r0) ** t
    elif spacing == "quadratic":
        r = r0 + (r1 - r0) * t ** 2
    else:
        raise ValueError(f"unknown spacing: {spacing}")
    return r


def _place_on_arms(
    n_arms: int, radii: np.ndarray, z: np.ndarray | float = 0.0,
    arm_phase: float = 0.0, stagger: float = 0.0,
) -> np.ndarray:
    """把每条臂上的半径序列按 az = arm_phase + 360*k/n_arms 摆放。

    stagger: 每条臂整体旋转的微小偏置（度），用于打散跨臂的重复间距。
    """
    pts = []
    for k in range(n_arms):
        ang = np.deg2rad(arm_phase + 360.0 * k / n_arms + stagger * k)
        ca, sa = np.cos(ang), np.sin(ang)
        zz = np.full(radii.shape, z) if np.isscalar(z) else z
        pts.append(np.stack([radii * ca, radii * sa, zz], axis=1))
    return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


def _dedup(mics: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """去掉重合阵元，保持顺序。"""
    keep = []
    for p in mics:
        if not any(np.linalg.norm(p - q) < tol for q in keep):
            keep.append(p)
    return np.asarray(keep) if keep else np.zeros((0, 3))


def _recenter(mics: np.ndarray, recenter: bool) -> tuple[np.ndarray, list[float]]:
    if not recenter or mics.size == 0:
        return mics, [0.0, 0.0, 0.0]
    c = mics.mean(axis=0)
    return mics - c, [round(float(v), 9) for v in c]


# --------------------------------------------------------------------------
# 几何构建器
# --------------------------------------------------------------------------
def ula(n: int, spacing: float = 0.04, axis: str = "x", z: float = 0.0) -> np.ndarray:
    """均匀直线阵。默认沿 x 轴（boresight 为 +x，即端射方向为 y）。"""
    pos = (np.arange(n) - (n - 1) / 2.0) * spacing
    mics = np.zeros((n, 3))
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    mics[:, idx] = pos
    if axis != "z":
        mics[:, 2] = z
    return mics


def uca(n: int, radius: float = 0.1, z: float = 0.0, phase: float = 0.0) -> np.ndarray:
    """均匀圆阵。"""
    ang = np.deg2rad(phase) + 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(ang), radius * np.sin(ang),
                     np.full(n, z)], axis=1)


def multi_ring(radii: Sequence[float], counts: Sequence[int], z: float = 0.0,
               stagger: bool = True) -> np.ndarray:
    """同心多环（常见天花板阵）。"""
    rings = []
    for k, (r, n) in enumerate(zip(radii, counts)):
        ph = 360.0 * k / (2.0 * len(radii)) if stagger else 0.0
        rings.append(uca(int(n), float(r), z=z, phase=ph))
    return np.concatenate(rings, axis=0)


def grid(nx: int, ny: int, dx: float = 0.04, dy: float | None = None,
         z: float = 0.0) -> np.ndarray:
    """矩形网格阵。"""
    dy = dx if dy is None else dy
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dy
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)


def spiral_arm(n_arms: int = 4, mics_per_arm: int = 4, r0: float = 0.03,
               r1: float = 0.25, spacing: Spacing = "log",
               z: float = 0.0, arm_phase: float = 0.0,
               stagger: float = 0.0) -> np.ndarray:
    """阿基米德螺旋臂阵：等弧长采样 + 半径线性增长。

    每条臂上阵元沿螺旋曲线等弧长分布，因此相邻间距近似相等，
    而不同臂之间形成大量互不相同的长基线 —— 这是"多尺度基线"的经典做法。
    """
    per = max(1, mics_per_arm)
    r_lin = np.linspace(r0, r1, per)
    # 等弧长修正：弧长 ∝ ∫sqrt(1+(r/b)^2)dr，对线性采样的半径做一次重分布
    s = np.cumsum(np.sqrt(1.0 + (r_lin / max(r1 - r0, 1e-9) * per) ** 2))
    s = (s - s[0]) / max(s[-1] - s[0], 1e-12)
    radii = r0 + (r1 - r0) * s

    pts = []
    for k in range(n_arms):
        base = np.deg2rad(arm_phase + 360.0 * k / n_arms + stagger * k)
        # 螺旋的角增量：从 r0 到 r1 转 total_turn 圈
        theta = base + 2.0 * np.pi * 1.25 * (radii - r0) / max(r1 - r0, 1e-12)
        pts.append(np.stack([radii * np.cos(theta), radii * np.sin(theta),
                             np.full(radii.shape, z)], axis=1))
    return np.concatenate(pts, axis=0)


def vogel(n: int, radius: float = 0.15, z: float = 0.0,
          phase: float = 0.0) -> np.ndarray:
    """Vogel / 向日葵螺旋：R = r*sqrt(i/n), theta = i*137.507...°

    在给定阵元数下给出近似最优的非冗余二维布点，常用于声源定位阵列。
    """
    golden = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    r = radius * np.sqrt((i + 0.5) / n)
    th = np.deg2rad(phase) + i * golden
    return np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], axis=1)


def cantilever(n_arms: int = 4, mics_per_arm: int = 4, r0: float = 0.035,
               r1: float = 0.24, spacing: Spacing = "log",
               board_radius: float = 0.035, rod_radius: float = 0.0025,
               arm_phase: float = 0.0, stagger: float = 0.0,
               z: float = 0.0) -> np.ndarray:
    """悬臂式（外伸臂）阵列：细杆从中央板伸出，麦克风装在杆上。

    与 spiral_arm 的区别是**强调机械形态**：中央板半径 board_radius，
    细杆半径 rod_radius 远小于最高频波长（减少散射与遮挡），
    臂从板边缘 r0 伸到 r1，麦位按 spacing 分布在杆上。
    """
    radii = _arm_radii(mics_per_arm, max(r0, board_radius), r1, spacing)
    return _place_on_arms(n_arms, radii, z=z, arm_phase=arm_phase, stagger=stagger)


def nested(n_inner: int = 4, n_outer: int = 4, d_inner: float = 0.02,
           d_outer: float = 0.05, axis: str = "x") -> np.ndarray:
    """嵌套阵：内层密排保证高频，外层稀疏保证低频与自由度。

    交换子间距 (d_inner, d_outer) 互质时差集近似连续，可显著扩展虚拟孔径。
    """
    inner = ula(n_inner, d_inner, axis=axis)
    outer = ula(n_outer, d_outer, axis=axis)
    offset = (n_inner * d_inner + d_outer) / 2.0 + d_outer * (n_outer - 1) / 2.0
    outer = outer - np.array([offset if axis == "x" else 0.0,
                              offset if axis == "y" else 0.0,
                              offset if axis == "z" else 0.0])
    return _dedup(np.concatenate([inner, outer], axis=0))


# --------------------------------------------------------------------------
# 统一入口
# --------------------------------------------------------------------------
@dataclass
class ArraySpec:
    kind: Kind = "uca"
    params: dict = field(default_factory=dict)
    recenter: bool = True
    c: float = C_DEFAULT
    name: str | None = None

    def build(self) -> "ArrayGeometry":
        fn = {
            "ula": ula, "uca": uca, "grid": grid, "spiral_arm": spiral_arm,
            "vogel": vogel, "cantilever": cantilever, "nested": nested,
            "multi_ring": multi_ring,
        }.get(self.kind)
        if fn is None:
            raise ValueError(f"unknown kind: {self.kind}")
        mics = np.asarray(fn(**self.params), dtype=np.float64)
        mics = _dedup(mics)
        mics, off = _recenter(mics, self.recenter)
        name = self.name or self.kind
        return ArrayGeometry(mics=mics, name=name, kind=self.kind,
                             center_offset=off, params=dict(self.params),
                             c=self.c)


@dataclass
class ArrayGeometry:
    mics: np.ndarray                 # (M,3)
    name: str = "array"
    kind: str = "custom"
    center_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    params: dict = field(default_factory=dict)
    c: float = C_DEFAULT

    def __post_init__(self) -> None:
        self.mics = np.asarray(self.mics, dtype=np.float64).reshape(-1, 3)

    @property
    def n_mics(self) -> int:
        return int(self.mics.shape[0])

    def metrics(self) -> dict:
        m = array_metrics(self.mics, self.c)
        m["redundancy"] = redundancy_stats(self.mics)
        return m

    def to_json(self) -> dict:
        m = self.metrics()
        return {
            "name": self.name,
            "kind": self.kind,
            "mics": [[round(float(v), 9) for v in p] for p in self.mics],
            "center": [0.0, 0.0, 0.0],
            "center_offset": self.center_offset,
            "params": _plain(self.params),
            "meta": {
                "n_mics": m["n_mics"],
                "aperture": m["aperture"],
                "min_spacing": m["min_spacing"],
                "grating_free_freq": m["grating_free_freq"],
                "speed_of_sound": self.c,
                "redundancy": m["redundancy"],
            },
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, allow_nan=False)


def _plain(obj):
    """把 numpy 标量/数组递归转成原生类型，便于 json。"""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# --------------------------------------------------------------------------
# 预设
# --------------------------------------------------------------------------
PRESETS: dict[str, ArraySpec] = {
    "uca-8":        ArraySpec("uca", {"n": 8, "radius": 0.06}, name="圆阵 8 元 Ø120mm"),
    "uca-16":       ArraySpec("uca", {"n": 16, "radius": 0.10}, name="圆阵 16 元 Ø200mm"),
    "ula-8":        ArraySpec("ula", {"n": 8, "spacing": 0.04}, name="线阵 8 元 40mm"),
    "spiral-4x4":   ArraySpec("spiral_arm", {"n_arms": 4, "mics_per_arm": 4,
                                             "r0": 0.03, "r1": 0.25, "spacing": "log"},
                              name="螺旋臂 4×4 (r 3→25cm, 对数)"),
    "spiral-4x4-broad": ArraySpec("spiral_arm", {"n_arms": 4, "mics_per_arm": 4,
                                                 "r0": 0.010, "r1": 0.50, "spacing": "log"},
                                  name="螺旋臂 4×4 宽带 (r 1→50cm, 对数)"),
    "spiral-6x3":   ArraySpec("spiral_arm", {"n_arms": 6, "mics_per_arm": 3,
                                             "r0": 0.035, "r1": 0.20, "spacing": "quadratic"},
                              name="螺旋臂 6×3 (二次)"),
    "ula-16-equal": ArraySpec("ula", {"n": 16, "spacing": 0.031},
                              name="线阵 16 元 31mm (等间距对照)"),
    "vogel-16":     ArraySpec("vogel", {"n": 16, "radius": 0.15}, name="Vogel 螺旋 16 元"),
    "vogel-32":     ArraySpec("vogel", {"n": 32, "radius": 0.20}, name="Vogel 螺旋 32 元"),
    "cantilever-4x4": ArraySpec("cantilever", {"n_arms": 4, "mics_per_arm": 4,
                                               "r0": 0.035, "r1": 0.24, "spacing": "log",
                                               "board_radius": 0.035, "rod_radius": 0.0025},
                                name="悬臂 4 臂×4 麦"),
    "grid-4x4":     ArraySpec("grid", {"nx": 4, "ny": 4, "dx": 0.045}, name="方阵 4×4"),
    "nested-4-4":   ArraySpec("nested", {"n_inner": 4, "n_outer": 4, "d_inner": 0.02,
                                         "d_outer": 0.05}, name="嵌套阵 4+4"),
    "ring-3":       ArraySpec("multi_ring", {"radii": [0.05, 0.10, 0.15],
                                             "counts": [6, 10, 14]},
                              name="同心三环 30 元"),
}


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def build_preset(name: str) -> ArrayGeometry:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name}")
    return PRESETS[name].build()


# --------------------------------------------------------------------------
# 方向网格（供 GUI / 定位 / 成像使用）
# --------------------------------------------------------------------------
def direction_grid(az_range=(-180.0, 180.0), el_range=(-90.0, 90.0),
                   az_bins: int = 72, el_bins: int = 37):
    """返回 (dirs (G,3), shape (n_el,n_az), az_axis, el_axis)。

    dirs 按 el 外层、az 内层展平（C 序），与 shape 一致。见 CONTRACT 第 2 节。
    """
    az = np.linspace(az_range[0], az_range[1], az_bins)
    el = np.linspace(el_range[0], el_range[1], el_bins)
    AZ, EL = np.meshgrid(az, el, indexing="xy")     # (n_el, n_az)
    a, e = np.deg2rad(AZ.ravel()), np.deg2rad(EL.ravel())
    dirs = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=1)
    return dirs, (el_bins, az_bins), az, el


def uv(dirs: np.ndarray, u_axis: str = "x", v_axis: str = "y") -> np.ndarray:
    """把方向向量投到观察平面的 (u,v) 坐标（声场成像绘图用）。"""
    idx = {"x": 0, "y": 1, "z": 2}
    return dirs[:, [idx[u_axis], idx[v_axis]]]


__all__ = [
    "C_DEFAULT", "ArraySpec", "ArrayGeometry", "PRESETS",
    "ula", "uca", "grid", "multi_ring", "spiral_arm", "vogel", "cantilever",
    "nested", "direction_grid", "uv", "pairwise_distances", "aperture",
    "min_spacing", "grating_free_freq", "redundancy_stats", "array_metrics",
    "preset_names", "build_preset",
]
