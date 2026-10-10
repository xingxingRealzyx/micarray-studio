"""几何声学：传播时延、球面扩散、阵列几何参数。

约定见 docs/CONTRACT.md 第 1-3 节。
"""
from __future__ import annotations

import numpy as np

C_DEFAULT = 343.0


def distance(mics: np.ndarray, point: np.ndarray) -> np.ndarray:
    """mics (M,3) 与 point (3,) 的距离 -> (M,)。"""
    return np.linalg.norm(mics - np.asarray(point, dtype=float)[None, :], axis=1)


def unit_dir(az_deg, el_deg) -> np.ndarray:
    """角度 -> 单位方向向量（**阵列坐标系**，见 CONTRACT 第 2 节）。

    阵元位于 xy 平面、boresight 为 +z，因此：
      az 在 xy 平面内从 +x 逆时针量取
      el 从 xy 平面量向 +z，el=0 是阵面切线方向，el=+90 是正对 boresight
    """
    a, e = np.deg2rad(float(az_deg)), np.deg2rad(float(el_deg))
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def dirs_from_angles(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """向量化版本 -> (...,3)。"""
    a, e = np.deg2rad(np.asarray(az_deg, float)), np.deg2rad(np.asarray(el_deg, float))
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def angles_from_dir(u: np.ndarray) -> tuple[float, float]:
    """单位方向向量 -> (az_deg, el_deg)，与 unit_dir 互逆。"""
    u = np.asarray(u, float)
    n = np.linalg.norm(u)
    if n > 0:
        u = u / n
    az = float(np.rad2deg(np.arctan2(u[1], u[0])))
    el = float(np.rad2deg(np.arcsin(np.clip(u[2], -1.0, 1.0))))
    return az, el


def delay_from_dir(mics: np.ndarray, u: np.ndarray, c: float = C_DEFAULT) -> np.ndarray:
    """远场平面波：阵元 m 相对质心的传播时延 tau_m = -dot(p_m, u)/c。

    符号约定：u 方向来波在 dot(p_m,u) 大的阵元上**先**到达，故 tau 为负。
    """
    return -(mics @ np.asarray(u, float)) / c


def delay_from_point(mics: np.ndarray, p: np.ndarray, c: float = C_DEFAULT) -> np.ndarray:
    """近场球面波：tau_m = (||p_m - p|| - ||p||)/c，以阵列质心为参考。"""
    p = np.asarray(p, float)
    return (distance(mics, p) - np.linalg.norm(p)) / c


def delays_grid(mics: np.ndarray, dirs: np.ndarray, c: float = C_DEFAULT) -> np.ndarray:
    """(G,3) 方向 -> (G,M) 时延矩阵，供方向扫描使用。"""
    return -(np.asarray(dirs, float) @ mics.T) / c


def spread_loss(mics: np.ndarray, p: np.ndarray, ref_distance: float) -> np.ndarray:
    """球面扩散幅度因子 r_ref / ||p_m - p||，归一化到参考距离。"""
    return float(ref_distance) / distance(mics, p)


def is_near_field(distance_m: float, aperture_m: float, freq_hz: float,
                  c: float = C_DEFAULT) -> bool:
    """Fraunhofer 判据：r < 2*D^2/lambda 时须用近场模型。"""
    if freq_hz <= 0:
        return False
    lam = c / freq_hz
    return distance_m < 2.0 * aperture_m ** 2 / lam


def fresnel_distance(aperture_m: float, freq_hz: float, c: float = C_DEFAULT) -> float:
    """临界距离 r = 2*D^2/lambda (m)。"""
    if freq_hz <= 0:
        return float("inf")
    return 2.0 * aperture_m ** 2 / (c / freq_hz)


def alias_limit(min_spacing_m: float, c: float = C_DEFAULT) -> float:
    """由最近阵元间距给出的无空间混叠上限频率 f = c/(2d)。"""
    return float(c / (2.0 * min_spacing_m)) if min_spacing_m > 0 else float("inf")


def angular_resolution(aperture_m: float, freq_hz: float,
                       c: float = C_DEFAULT) -> float:
    """均匀加权线阵的 -3dB 波束宽度近似（度）：theta ≈ 0.886*lambda/D。"""
    if aperture_m <= 0 or freq_hz <= 0:
        return float("nan")
    lam = c / freq_hz
    return float(np.rad2deg(0.886 * lam / aperture_m))
