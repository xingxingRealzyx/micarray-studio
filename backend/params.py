"""请求参数归一化。

后端与前端（以及外部脚本）对同一个概念会有不同的写法。这里把两种常见
形态统一收口，避免"字段名不一致导致静默走默认值"这类最难查的问题：

  扁平形态（脚本/curl 友好）          嵌套形态（前端 reqBody 的写法）
  ---------------------------------  -------------------------------------
  room_enabled / room.bounds         room.enabled / room.size / room.order
  noise_kind / snr_db                noise.snr_db
  steer_az / steer_el                steer=[az, el]
  source_az ...                      sources[0].az ...
  freqs=[...]                        freqs={fmin, fmax, n}
  az_bins / el_bins                  grid.az_bins / grid.el_bins
  mismatch.gain_db_std               mismatch.gain_std
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "fs": 16000,
    "duration": 0.8,
    "n_fft": 512,
    "az_bins": 72,
    "el_bins": 37,
    "snr_db": 25.0,
    "fmin": 300.0,
    "fmax": 6000.0,
    "nfreq": 5,
}


def _sub(p: dict, key: str) -> dict:
    v = p.get(key)
    return v if isinstance(v, dict) else {}


def _first(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


# --------------------------------------------------------------------------
def room_enabled(p: dict) -> bool:
    r = _sub(p, "room")
    return bool(_first(p.get("room_enabled"), r.get("enabled"), default=False))


def room_bounds(p: dict) -> tuple:
    """返回 (bounds6, center3)。

    前端的 `bounds` 是 [[0,0,0],[sx,sy,sz]] 两元素形式（房间占第一象限），
    后端引擎用的是 [xmin,xmax,ymin,ymax,zmin,zmax] 六元素形式。
    两种都接受，并把阵列默认放在房间中心、朝下（吊顶装法）。
    """
    r = _sub(p, "room")
    b = _first(p.get("bounds"), r.get("bounds"))
    if b is not None:
        b = np.asarray(b, float)
        if b.shape == (2, 3):
            lo, hi = b[0], b[1]
            return ((float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
                     float(lo[2]), float(hi[2])),
                    tuple((lo + hi) / 2.0))
        if b.size == 6:
            v = b.ravel()
            return (tuple(float(x) for x in v),
                    ((v[0] + v[1]) / 2, (v[2] + v[3]) / 2, (v[4] + v[5]) / 2))
    size = _first(p.get("room_size"), r.get("size"), default=(6.0, 4.0, 3.0))
    sx, sy, sz = (float(v) for v in size)
    return ((0.0, sx, 0.0, sy, 0.0, sz), (sx / 2, sy / 2, sz / 2))


def room_spec(p: dict):
    from engine import sim
    if not room_enabled(p):
        return None, None
    r = _sub(p, "room")
    bounds, center = room_bounds(p)
    order = int(_first(p.get("max_order"), p.get("room_order"), r.get("order"),
                        r.get("max_order"), default=2))
    absorption = _first(p.get("absorption"), r.get("absorption"),
                        default=(0.3,) * 6)
    spec = sim.RoomSpec(bounds=bounds, absorption=tuple(float(v) for v in absorption),
                        max_order=order)
    return spec, center


def array_center(p: dict) -> tuple:
    _, center = room_bounds(p)
    return tuple(float(v) for v in _first(p.get("center"), default=center))


def noise_spec(p: dict):
    from engine import sim
    n = _sub(p, "noise")
    snr = _first(p.get("snr_db"), n.get("snr_db"), default=DEFAULTS["snr_db"])
    if snr is None:
        return None
    return sim.NoiseSpec(snr_db=float(snr),
                         kind=_first(p.get("noise_kind"), n.get("kind"),
                                     default="white"),
                         coherent=bool(_first(p.get("noise_coherent"),
                                              n.get("coherent"), default=False)),
                         seed=int(n.get("seed", 7)))


def mismatch_spec(p: dict):
    from engine import sim
    m = _sub(p, "mismatch")
    return sim.MismatchSpec(
        gain_db_std=float(_first(p.get("gain_db_std"), m.get("gain_db_std"),
                                 m.get("gain_std"), default=0.0)),
        phase_deg_std=float(_first(p.get("phase_deg_std"), m.get("phase_deg_std"),
                                   m.get("phase_std_deg"), default=0.0)),
        pos_std_m=float(_first(p.get("pos_std_m"), m.get("pos_std_m"),
                               default=0.0)),
        seed=int(m.get("seed", 5)))


# --------------------------------------------------------------------------
def steer(p: dict, default=(0.0, 90.0)) -> tuple:
    """指向：支持 [az,el] / {az,el} / steer_az+steer_el / sources[0]。"""
    s = _first(p.get("steer"), p.get("steer_dir"))
    if isinstance(s, dict):
        return (float(s.get("az", default[0])), float(s.get("el", default[1])))
    if s is not None and not isinstance(s, str):
        a = np.asarray(s, float).ravel()
        if a.size >= 2:
            return (float(a[0]), float(a[1]))
    if p.get("steer_az") is not None or p.get("steer_el") is not None:
        return (float(_first(p.get("steer_az"), default=default[0])),
                float(_first(p.get("steer_el"), default=default[1])))
    srcs = p.get("sources")
    if srcs:
        return (float(srcs[0].get("az", default[0])),
                float(srcs[0].get("el", default[1])))
    return default


def freqs(p: dict) -> np.ndarray:
    """设计频点：支持显式数组，或 {fmin,fmax,n} / freq_range 形式。"""
    f = p.get("freqs")
    if isinstance(f, (list, tuple)) and f and not isinstance(f[0], (list, dict)):
        return np.asarray(f, float)
    fr = f if isinstance(f, dict) else _sub(p, "freq_range")
    fmin = float(_first(p.get("fmin"), fr.get("fmin"), default=DEFAULTS["fmin"]))
    fmax = float(_first(p.get("fmax"), fr.get("fmax"), default=DEFAULTS["fmax"]))
    n = int(_first(p.get("nfreq"), fr.get("n"), default=DEFAULTS["nfreq"]))
    if fmax <= fmin:
        fmax = fmin * 2.0
    return np.linspace(fmin, fmax, max(int(n), 2))


def grid_bins(p: dict) -> tuple:
    g = _sub(p, "grid")
    return (int(_first(p.get("az_bins"), g.get("az_bins"),
                      default=DEFAULTS["az_bins"])),
            int(_first(p.get("el_bins"), g.get("el_bins"),
                      default=DEFAULTS["el_bins"])))


def fs(p: dict) -> int:
    return int(_first(p.get("fs"), default=DEFAULTS["fs"]))


def duration(p: dict) -> float:
    return float(_first(p.get("duration"), default=DEFAULTS["duration"]))


def n_fft(p: dict) -> int:
    return int(_first(p.get("n_fft"), default=DEFAULTS["n_fft"]))


def preset_params(p: dict) -> dict:
    """阵列参数：显式 kind+params 优先，其次是预设名，最后用 mics 坐标。"""
    if p.get("mics"):
        return {"mics": p["mics"]}
    if p.get("preset"):
        return {"preset": p["preset"]}
    return {"kind": p.get("kind", "uca"), "params": dict(p.get("params") or {})}


def sources(p: dict) -> list:
    srcs = p.get("sources")
    if srcs:
        return [dict(s) for s in srcs]
    az, el = steer(p)
    return [{"az": az, "el": el,
             "distance": float(_first(p.get("distance"), default=1.5)),
             "amplitude": float(_first(p.get("amplitude"), default=1.0)),
             "signal": _first(p.get("signal"), default="speech")}]


__all__ = [
    "room_enabled", "room_bounds", "room_spec", "array_center", "noise_spec",
    "mismatch_spec", "steer", "freqs", "grid_bins", "fs", "duration", "n_fft",
    "preset_params", "sources", "DEFAULTS",
]
