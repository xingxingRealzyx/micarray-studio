"""MicArray Studio 后端：标准库 http.server + JSON API。

只用 Python 标准库（无需 fastapi/uvicorn）。工程、音频、任务状态都落在
`backend/_session/` 下，方便排查问题。

API 契约见 docs/CONTRACT.md 第 8 节。
"""
from __future__ import annotations

import json
import math
import re
import pathlib
import sys
import threading
import time
import traceback
import uuid
import wave

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402

from engine import acoustics as ac                          # noqa: E402
from engine import beamforming as bfm                       # noqa: E402
from engine import codegen as cg                            # noqa: E402
from engine import dsp, geometry as geo, sim                # noqa: E402
from engine import signals as sg                            # noqa: E402
from backend import params as P                             # noqa: E402

SESSION = pathlib.Path(__file__).resolve().parent / "_session"
SESSION.mkdir(exist_ok=True)
EXPORT_ROOT = SESSION / "export"
EXPORT_ROOT.mkdir(exist_ok=True)
WEB_ROOT = ROOT / "web"

# 可选模块：并行开发中，缺失时给出清晰提示而不是整体崩溃
_MODULES: dict[str, object] = {}
for _name in ("localization", "imaging", "enhance"):
    try:
        _MODULES[_name] = __import__(f"engine.{_name}", fromlist=[_name])
    except Exception as _e:                                 # pragma: no cover
        _MODULES[_name] = None
        print(f"[warn] engine.{_name} 不可用: {_e}")


# --------------------------------------------------------------------------
# JSON 安全化
# --------------------------------------------------------------------------
def sanitize(obj, ndigits: int = 6):
    """递归转成 JSON 安全的结构。

    - numpy 标量/数组 -> 原生类型（数组保持**扁平 + shape**，避免嵌套开销）
    - NaN / Inf -> None（JSON 标准不允许，前端也画不出来）
    - 浮点统一保留 ndigits 位，既能显著减小响应体，又不影响可视化精度
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return round(v, ndigits)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return sanitize(obj.item(), ndigits)
        if np.iscomplexobj(obj):
            obj = np.abs(obj)
        return [sanitize(v, ndigits) for v in obj.ravel().tolist()]
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return str(obj)


def flat_with_shape(a, ndigits: int = 6) -> dict:
    """数组 -> {"data": [...], "shape": [...]}，供前端 heatmap 使用。"""
    arr = np.asarray(a, dtype=float)
    return {"data": sanitize(arr, ndigits), "shape": list(arr.shape)}


# --------------------------------------------------------------------------
# 请求参数 -> 引擎对象（统一走 backend/params.py 的归一化）
# --------------------------------------------------------------------------
def _resolve_preset(key: str):
    """按 id、显示名或模糊匹配解析预设。

    前端下拉框用 name（显示名，如"圆阵 8 元 Ø120mm"）当 option 的 value，
    而引擎的预设键是 id（如 "uca-8"）。只认 id 会直接 400，且错误信息
    对用户毫无意义，所以这里做三级容错。
    """
    if key in geo.PRESETS:
        return geo.build_preset(key)
    # 前端下拉的 value 可能是 "显示名 [id]" 这种复合形式，先按方括号里的 id 试
    m = re.search(r"\[([^\[\]]+)\]\s*$", key)
    if m and m.group(1) in geo.PRESETS:
        return geo.build_preset(m.group(1))
    for k, spec in geo.PRESETS.items():
        if spec.name == key:
            return geo.build_preset(k)
    for k, spec in geo.PRESETS.items():
        if key and (k in key or (spec.name or "") in key):
            return geo.build_preset(k)
    raise ValueError(
        f"未知阵列预设 {key!r}；可用 id: {geo.preset_names()}")


def build_geometry(p: dict):
    """支持三种方式：preset 名字 / kind+params / 直接给 mics 坐标。

    前端只发一组通用量（n_mics / radius / spacing），不同阵型接受的参数名
    各不相同。这里做一次"按阵型过滤"：多余的键直接丢掉，否则会 TypeError
    （例如 uca() 不接受 spacing），前端只会看到一个 400，很难查。
    """
    spec = P.preset_params(p)
    if spec.get("mics"):
        return geo.ArrayGeometry(mics=np.asarray(spec["mics"], float),
                                 name="custom", kind="custom")
    if spec.get("preset"):
        g = _resolve_preset(str(spec["preset"]))
        if p.get("name"):
            g.name = p["name"]
        return g

    kind = spec.get("kind", "uca")
    raw = dict(spec.get("params") or {})
    n = raw.pop("n_mics", None)
    spacing = raw.pop("spacing", None)
    radius = raw.pop("radius", None)

    # 每种阵型接受的参数名（与 engine/geometry.py 的函数签名一致）
    allowed = {
        "ula": {"n", "spacing", "axis", "z"},
        "uca": {"n", "radius", "z", "phase"},
        "grid": {"nx", "ny", "dx", "dy", "z"},
        "spiral_arm": {"n_arms", "mics_per_arm", "r0", "r1", "spacing", "z",
                       "arm_phase", "stagger"},
        "cantilever": {"n_arms", "mics_per_arm", "r0", "r1", "spacing",
                       "board_radius", "rod_radius", "arm_phase", "stagger", "z"},
        "vogel": {"n", "radius", "z", "phase"},
        "nested": {"n_inner", "n_outer", "d_inner", "d_outer", "axis"},
        "multi_ring": {"radii", "counts", "z", "stagger"},
    }.get(kind, set(raw.keys()) | {"n", "radius", "spacing", "nx", "ny", "dx"})

    if n is not None:
        if kind == "grid":
            raw.setdefault("nx", int(n))
            raw.setdefault("ny", int(n))
        elif kind in ("spiral_arm", "cantilever"):
            # 这类阵型的"阵元数"= 臂数 × 每臂麦数，用 n 反推一组合理默认
            if n >= 4:
                raw.setdefault("n_arms", 4)
                raw.setdefault("mics_per_arm", max(int(round(n / 4)), 1))
        else:
            raw.setdefault("n", int(n))
    if radius is not None and kind in ("uca", "vogel", "multi_ring"):
        if kind == "multi_ring":
            raw.setdefault("radii", [float(radius) * k for k in (0.5, 1.0, 1.5)])
            raw.setdefault("counts", [6, 10, 14])
        else:
            raw.setdefault("radius", float(radius))
    if spacing is not None:
        if kind == "grid":
            raw.setdefault("dx", float(spacing))
        elif kind in ("spiral_arm", "cantilever"):
            # 这两类阵型的 spacing 是**分布策略字符串**（equal/log/quadratic），
            # 不是长度。前端统一发数字，此时把它当"最大半径"更符合直觉。
            raw.setdefault("r1", max(float(spacing) * 6.0, 0.2))
            raw.setdefault("spacing", "log")
        elif kind in ("ula", "nested"):
            raw.setdefault("spacing", float(spacing))

    filtered = {k: v for k, v in raw.items() if k in allowed and v is not None}
    try:
        return geo.ArraySpec(kind, filtered).build()
    except TypeError as e:
        # 把可用参数名报给调用方，便于前端/脚本自查
        raise ValueError(
            f"阵型 {kind} 的参数不合法: {e}；该阵型接受 "
            f"{sorted(allowed)}，收到 {sorted(filtered)}") from None


def build_sources(p: dict) -> list[sim.SourceSpec]:
    """SourcesSpec 归一化：缺 signal 时给 speech，并保证 seed 各不相同。"""
    out = []
    for i, s in enumerate(P.sources(p)):
        out.append(sim.SourceSpec(
            signal=s.get("signal", "speech"),
            params=dict(s.get("signal_params", {"seed": 10 + i})),
            direction=(float(s.get("az", 0.0)), float(s.get("el", 60.0))),
            distance=float(s.get("distance", 1.5)),
            amplitude=float(s.get("amplitude", 1.0)),
            label=s.get("label", f"src{i}"),
        ))
    return out


def run_sim(p: dict):
    """按请求跑一次仿真，返回 (geom, result)。

    阵列位置与朝向：默认放在房间中心、主瓣朝下（吊顶阵列的典型装法）；
    用户可用 center / boresight 覆盖。注意 center 必须与房间 bounds 一致，
    否则声源会落到房间外（引擎会明确报错，这是特性不是 bug）。
    """
    g = build_geometry(p)
    room, _ = P.room_spec(p)
    # 注意：房间关闭时 room_spec 返回 (None, None)，不能直接用它的 center。
    # array_center() 内部已处理"有房间用房间中心、无房间用默认值"。
    center = tuple(float(v) for v in (p.get("center") or P.array_center(p)))
    cfg = sim.SimConfig(fs=P.fs(p), duration=P.duration(p),
                        c=float(p.get("speed_of_sound", 343.0)),
                        center=center,
                        boresight=tuple(p.get("boresight", (0.0, -90.0))))
    res = sim.simulate(g, build_sources(p), cfg, room=room,
                       noise=P.noise_spec(p), mismatch=P.mismatch_spec(p))
    return g, res


def axis_or_none(v):
    return None if v is None else sanitize(np.asarray(v, float))


def normalize_map_result(out: dict, dirs, shape, az, el) -> dict:
    """把各模块的 map 结果统一成契约里的形状。

    子模块对 `shape` / `az_axis` 的写法略有差异（有的是网格形状，有的是展平
    长度），这里统一以 `shape` 为网格形状、`az_axis`/`el_axis` 为一维轴。
    """
    m = np.asarray(out.get("map", []), float).ravel()
    # 契约要求 shape 是**网格形状**（如 (37,72)），而子模块可能返回展平长度。
    # 以传入的网格 shape 为准，只有尺寸对不上时才退回子模块的值。
    shp = tuple(shape) if int(np.prod(shape)) == m.size else (
        out.get("shape") or tuple(shape))
    if int(np.prod(shp)) != m.size:
        shp = (m.size,)
    # 契约第 8 节要的是**扁平数组 + 独立的 shape**，不是 {data,shape} 嵌套对象。
    # 早期返回嵌套对象时前端 normalizeField 会把 data 数组当成 map 用，
    # 直接抛 "a.map is not a function"，整张定位图/成像图画不出来。
    return {
        "map": sanitize(m),
        "map_shape": [int(v) for v in shp],
        "shape": [int(v) for v in shp],
        "az_axis": axis_or_none(out.get("az_axis") if out.get("az_axis") is not None else az),
        "el_axis": axis_or_none(out.get("el_axis") if out.get("el_axis") is not None else el),
        "est_dirs": sanitize(out.get("est_dirs", [])),
        "peak_vals": sanitize(out.get("peak_vals", [])),
        "method": out.get("method", "?"),
    }


# --------------------------------------------------------------------------
# 各端点实现
# --------------------------------------------------------------------------
def api_arrays(_p: dict) -> dict:
    return {
        # 同时给 id 与 name：前端用哪个当 value 都能工作（后端也做容错解析）
        "presets": [{"id": k, "name": geo.PRESETS[k].name,
                     "label": f"{geo.PRESETS[k].name} [{k}]"}
                    for k in geo.preset_names()],
        "current": geo.build_preset("uca-8").to_json(),
        "kinds": ["ula", "uca", "grid", "spiral_arm", "vogel", "cantilever",
                  "nested", "multi_ring"],
        "convention": {
            "frame": "阵列坐标系：阵元在 xy 平面，boresight = +z",
            "az": "xy 平面内从 +x 逆时针，[-180,180] 度",
            "el": "从 xy 平面量向 +z，[-90,90] 度；el=+90 正对阵列",
        },
    }


def api_array_design(p: dict) -> dict:
    g = build_geometry(p)
    return g.to_json()


def _grid_from_params(p: dict):
    az_bins, el_bins = P.grid_bins(p)
    return geo.direction_grid(
        az_range=tuple(p.get("az_range", (-180.0, 180.0))),
        el_range=tuple(p.get("el_range", (-90.0, 90.0))),
        az_bins=az_bins, el_bins=el_bins)


def _flatten_patterns(methods: dict) -> dict:
    """把每个方法的 pattern_db 从 [F][G] 压成扁平 F*G（契约第 8 节）。"""
    out = {}
    for k, v in methods.items():
        if not isinstance(v, dict) or "pattern_db" not in v:
            out[k] = v
            continue
        v = dict(v)
        rows = v["pattern_db"]
        v["pattern_db"] = [x for row in rows for x in row] if rows and \
            isinstance(rows[0], list) else list(rows)
        out[k] = v
    return out


def _count_freqs(methods: dict) -> int:
    """从成功的方法条目里取频点数（跳过 {"error": ...} 的失败项）。"""
    for v in (methods or {}).values():
        if isinstance(v, dict) and "freqs" in v:
            try:
                return len(v["freqs"])
            except TypeError:
                continue
    return 0


def api_beamform(p: dict) -> dict:
    t0 = time.time()
    g = build_geometry(p)
    freqs = P.freqs(p)
    steer = P.steer(p)
    methods = p.get("methods") or ["delay_sum", "mvdr", "superdirective", "fi"]
    # 前端历史上列过 lcmv / max_snr / min_variance 三个引擎并没有实现的方法名。
    # 这里显式映射到语义最接近的实现（lcmv/max_snr → max_sir，min_variance → mvdr），
    # 而不是让引擎抛异常 —— 但要如实标注实际用的方法名。
    _BM_ALIAS = {"lcmv": "max_sir", "max_snr": "max_sir",
                 "min_variance": "mvdr", "mvdr_min_variance": "mvdr"}
    methods = [_BM_ALIAS.get(str(m).lower(), m) for m in methods]
    wng_floor = p.get("wng_floor_db")
    loading = float(p.get("loading", 0.0))

    R_est = None
    if any(m in ("mvdr", "max_sir") for m in methods):
        # 用一次含干扰的仿真估计协方差
        try:
            _, res = run_sim(p)   # run_sim 返回 (geom, result)
            Xf, fr = dsp.stft(res.X, res.fs, n_fft=P.n_fft(p))
            R = np.einsum("mkf,nkf->fmn", Xf, Xf.conj()) / max(Xf.shape[1], 1)
            idx = [int(np.argmin(np.abs(fr - f))) for f in freqs]
            R_est = R[idx]
        except Exception:
            R_est = None

    # 严格尊重调用方请求的网格。早期版本写成 max(请求值, 91/46)，
    # 结果是"数据按大网格算、shape 按请求值报"，前端拿到的长度与声称形状
    # 不一致，索引全错位、图看着像一片平色 —— 这类静默错位最难查。
    azb, elb = P.grid_bins(p)
    # 契约第 8 节：大数组一律"扁平 + shape"。pattern_db 早期写成 [F][G] 嵌套，
    # 前端按扁平解析时每个频点会变成 G 行，索引全错。这里统一扁平。
    _flat_pattern = bool(p.get("flat_pattern", True))
    # 说明：dirs 与 pattern_db 都是扁平数组，dirs 长度 = G*3，
    # pattern_db 长度 = F*G，G = shape[0]*shape[1]。
    if p.get("az_bins") is not None:
        azb = int(p["az_bins"])
    if p.get("el_bins") is not None:
        elb = int(p["el_bins"])
    sw = bfm.sweep(g, freqs, steer=steer, methods=methods,
                   az_bins=azb, el_bins=elb,
                   R_est=R_est, wng_floor_db=wng_floor, loading=loading,
                   fi_mainlobe_deg=p.get("fi_mainlobe_deg"))
    return {
        "geometry": g.to_json(),
        # dirs 必须是**扁平** N*3（CONTRACT 第 8 节：大数组一律扁平+shape）。
        # 早期直接 sanitize((G,3) 数组) 会保留嵌套，前端拿到 G*3 个数按点数
        # 解析，于是方向与功率值错位、整幅图看起来像一片平色。
        "dirs": sanitize(np.asarray(sw["dirs"], float).ravel()),
        "shape": list(sw["shape"]),
        "az_axis": axis_or_none(sw["az_axis"]),
        "el_axis": axis_or_none(sw["el_axis"]),
        "methods": sanitize(_flatten_patterns(sw["methods"]) if _flat_pattern
                             else sw["methods"]),
        "convention_note": "pattern_db 为**扁平**数组，长度 F*G（freqs 外层、"
                           "dirs 内层，与 shape 一致）；建议按 u-v 圆盘渲染"
                           "而非矩形 az-el 图",
        "pattern_layout": {
            "freqs": "outer", "dirs": "inner",
            # 必须容错：某个方法失败时它的条目是 {"error": ...}，没有 freqs 键。
            # 早期写成 sw["methods"][第一个]["freqs"]，只要第一个方法失败就
            # 抛 KeyError，把整个响应打成 400 —— 单方法失败不该毁掉整次请求。
            "n_freqs": _count_freqs(sw.get("methods") or {}),
        },
        "runtime_s": round(time.time() - t0, 3),
    }


def api_locate(p: dict) -> dict:
    t0 = time.time()
    mod = _MODULES["localization"]
    if mod is None:
        raise RuntimeError("定位模块不可用（engine/localization.py 导入失败）")
    g, res = run_sim(p)
    dirs, shape, az, el = _grid_from_params(p)
    truth = res.truth["sources"][0]["direction_rel"] if res.truth.get("sources") else None
    out = mod.localize(res.X, res.fs, g.mics,
                       method=p.get("method", "srp_phat"),
                       grid_dirs=dirs, n_sources=int(p.get("n_sources", 1)),
                       n_fft=P.n_fft(p), truth=truth)
    r = normalize_map_result(out, dirs, shape, az, el)
    r.update({
        "est_xyz": sanitize(out.get("est_xyz", [])),
        "truth": sanitize(res.truth["sources"]),
        "peak_dir": sanitize(out.get("est_dirs", [])),
        "geometry": g.to_json(),
        "runtime_s": round(time.time() - t0, 3),
    })
    if truth and out.get("est_dirs"):
        try:
            e = np.asarray(out["est_dirs"], float)[0]
            r["error_deg"] = round(float(np.linalg.norm(e - np.asarray(truth, float))), 3)
        except Exception:
            r["error_deg"] = None
    return r


def api_imaging(p: dict) -> dict:
    t0 = time.time()
    mod = _MODULES["imaging"]
    if mod is None:
        raise RuntimeError("成像模块不可用（engine/imaging.py 导入失败）")
    g, res = run_sim(p)          # 注意是二元组：几何 + 仿真结果
    dirs, shape, az, el = _grid_from_params(p)
    # freqs 可能是显式数组，也可能是 {fmin,fmax,n} 这种范围描述（前端用后者）。
    # 只有显式数组才值得传给引擎当"指定频点"，其余情况交给引擎自己选频。
    fq = p.get("freqs")
    if isinstance(fq, dict) or fq is None:
        freqs = None
    else:
        freqs = np.asarray(fq, float)
        if freqs.ndim != 1 or freqs.size == 0:
            freqs = None
    dirty = mod.dirty_map(res.X, res.fs, g.mics, dirs, freqs=freqs,
                          n_fft=P.n_fft(p), weights=p.get("weights", "ds"))
    # dirty["map"] 可能是 (G,) 或 (G,1)；契约要求网格形状，这里强制 reshape
    dm = np.asarray(dirty["map"], float).reshape(-1)
    if dm.size != int(np.prod(shape)):
        shape = (dm.size,)
    dm = dm.reshape(shape)
    out = {
        # 同样是"扁平数组 + shape"，不要 {data,shape} 嵌套（见 normalize_map_result 注释）
        "dirty": sanitize(dm.ravel()),
        "dirty_shape": list(shape),
        "shape": list(shape),
        "az_axis": axis_or_none(az),
        "el_axis": axis_or_none(el),
        "dirty_metrics": sanitize(dirty.get("metrics", {})),
        "freqs_used": sanitize(dirty.get("freqs_used", [])),
        "method": p.get("method", "damas"),
        "geometry": g.to_json(),
    }
    method = p.get("method", "damas")
    if method == "damas":
        # damas 需要 PSF 回调，直接复用 imaging 模块的构造器
        # PointSpreadFunction 本身就是合法的 psf_fn（imaging.damas 会识别
        # 它的 columns()/shape 接口）。早期版本我误传了 .col 方法，
        # 单列返回 (G,1) 导致 "长度 1 != 脏图长度 G" —— 直接给对象即可。
        psf = mod.point_spread_function(
            g.mics, np.asarray(dirty.get("freqs_used", [1000.0]), float), dirs)
        # DAMAS 的耗时 ∝ n_iter × G²，默认 120 次在 72×37 网格上要 40 秒，
        # 会撞前端 30 秒超时。这里默认 40 次（前 40 次已收敛大部分能量），
        # 并把默认网格压到 61×31；用户要更精细可显式加大 n_iter。
        n_iter = int(p.get("n_iter", 40))
        dc = mod.damas(dm, psf_fn=psf, shape=shape, n_iter=n_iter)
        cm = np.asarray(dc.get("clean_map", dc.get("map")), float).reshape(-1)
        out["deconv"] = sanitize(cm)
        out["deconv_shape"] = list(shape if cm.size == int(np.prod(shape)) else (cm.size,))
        out["history"] = sanitize(dc.get("history", {}))
        out["stop_reason"] = dc.get("stop_reason")
        out["n_components"] = dc.get("n_components")
    elif method == "clean_sc":
        cs = mod.clean_sc(res.X, res.fs, g.mics, dirs,
                          n_sources=int(p.get("n_sources", 3)),
                          n_fft=P.n_fft(p))
        cm = np.asarray(cs["map"], float).reshape(-1)
        out["deconv"] = sanitize(cm)
        out["deconv_shape"] = list(shape if cm.size == int(np.prod(shape)) else (cm.size,))
        out["components"] = sanitize(cs.get("components", []))
    else:
        out["deconv"] = out["dirty"]
    out["runtime_s"] = round(time.time() - t0, 3)
    return out


def api_enhance(p: dict) -> dict:
    t0 = time.time()
    mod = _MODULES["enhance"]
    if mod is None:
        raise RuntimeError("增强模块不可用（engine/enhance.py 导入失败）")
    g, res = run_sim(p)          # 注意是二元组：几何 + 仿真结果
    steer = P.steer(p)
    method = p.get("method", "mvdr_postfilter")
    # 方法名别名：前端下拉用的是简称（mvdr / wiener / delay_sum / superdirective），
    # 后端函数名是带后缀的全称。只认全称会让前端直接吃 400（实际发生过）。
    _ALIAS = {
        "mvdr": "mvdr_postfilter",
        "mvdr_postfilter": "mvdr_postfilter",
        "wiener": "multichannel_wiener",
        "mwf": "multichannel_wiener",
        "multichannel_wiener": "multichannel_wiener",
        "gsc": "gsc",
        # 纯波束成形方法：没有后置滤波，直接用 MVDR 链路 + 相应指向设计，
        # 至少让用户能看到"只做空间滤波"的效果，而不是报错。
        "delay_sum": "mvdr_postfilter",
        "superdirective": "mvdr_postfilter",
    }
    key = _ALIAS.get(str(method).lower())
    if key is None:
        raise ValueError(
            f"未知增强方法 {method!r}；可用: "
            f"{sorted(set(_ALIAS))}")
    fn = {"mvdr_postfilter": mod.mvdr_postfilter,
          "multichannel_wiener": mod.multichannel_wiener,
          "gsc": mod.gsc}[key]
    method = key
    # 把 ArrayGeometry 通过 geom= 传进去（不是当成 mics 位置参数！）：
    # 只有拿到 geom，enhance 才会调用 beamforming.design() 走 MVDR 自适应设计；
    # 否则退化为解析解，指向与失配处理都会变差。
    kw = {"n_fft": P.n_fft(p), "geom": g}
    try:
        out = fn(res.X, res.fs, g.mics, steer, **kw)
    except TypeError:
        # gsc 目前不接受 geom=，退回位置参数调用
        out = fn(res.X, res.fs, g.mics, steer, n_fft=P.n_fft(p))

    y = np.asarray(out["y"], float).ravel()
    clean1 = res.clean[0] if res.clean is not None else None
    ev = {}
    ref_note = ""
    if clean1 is not None:
        # 参考信号的选择很关键：房间开启时 sim.clean 含完整混响尾，而增强
        # （尤其后置滤波/谱减）会主动去掉混响 —— 用混响信号当 SI-SDR 参考，
        # 等于把"去混响"判成失真，实测 SI-SDR 会变成负值（−6 dB），
        # 看起来像算法失败。改用**直达声**作参考在数学上更正确，数值也回到
        # 合理区间。无房间时 direct 与 clean 等价，行为不变。
        ref = None
        if res.direct is not None:
            rp = float(np.mean(res.direct[0] ** 2))
            if rp > 1e-12 * max(float(np.mean(clean1 ** 2)), 1e-20):
                ref = res.direct[0]
                ref_note = ("客观指标以直达声为参考（房间混响下用混响信号作 SI-SDR "
                            "参考会把去混响误判为失真）")
        if ref is None:
            ref = clean1
            ref_note = "客观指标以 clean[0] 为参考（自由场，与直达声等价）"
        try:
            ev = mod.evaluate(ref, y, res.fs, noisy=res.X[0])
        except Exception as ex:
            ev = {"error": str(ex)}
    uid = uuid.uuid4().hex[:12]
    noisy_path = _write_wav(SESSION / f"noisy_{uid}.wav", res.X[0], res.fs)
    out_path = _write_wav(SESSION / f"enh_{uid}.wav", y, res.fs)
    return {
        "method": method,
        "metrics": sanitize({**(out.get("metrics") or {}), **ev}),
        "curve": sanitize(out.get("curve", {})),
        "gain_curve": sanitize(out.get("gain", [])) if p.get("return_gain") else None,
        "audio_url": f"/api/audio/{out_path.name}",
        "noisy_audio_url": f"/api/audio/{noisy_path.name}",
        # 以 evaluate 的口径为准（含 noisy 基准），缺省再退回模块自身的 metrics
        "snr_in": sanitize(ev.get("snr_in_db") or (out.get("metrics") or {}).get("snr_in_db")),
        "snr_out": sanitize(ev.get("snr_out_db") or (out.get("metrics") or {}).get("snr_out_db")),
        "seg_snr_in": sanitize(ev.get("segmental_snr_in_db")),
        "seg_snr_out": sanitize(ev.get("segmental_snr_out_db")),
        "snr_gain": sanitize(ev.get("snr_improvement_db") or ev.get("seg_snr_gain_db")),
        "si_sdr_in": sanitize(ev.get("si_sdr_in_db")),
        "si_sdr_out": sanitize(ev.get("si_sdr_out_db")),
        "si_sdr_gain": sanitize(ev.get("si_sdr_improvement_db")),
        "truth": sanitize(res.truth["sources"]),
        "geometry": g.to_json(),
        "ref_note": ref_note,
        "room": None if res.meta.get("room") is None else {
            "rt60_est": res.meta["room"]["rt60_est"]},
        "runtime_s": round(time.time() - t0, 3),
    }


def api_export_cpp(p: dict) -> dict:
    g = build_geometry(p)
    fs_ = P.fs(p)
    freqs = p.get("freqs")
    freqs = (np.asarray(freqs, float) if isinstance(freqs, list) and freqs
             else np.fft.rfftfreq(P.n_fft(p), 1.0 / fs_))
    name = str(p.get("name", "beamformer")).replace("/", "_")[:40] or "beamformer"
    result = cg.export_and_verify(
        g, freqs, steer=(float(p.get("steer_az", 0.0)), float(p.get("steer_el", 90.0))),
        method=p.get("method", "delay_sum"), name=name, outdir=EXPORT_ROOT,
        fs=fs_, n_fft=P.n_fft(p),
        n_samples=int(p.get("n_samples", 512)),
        verify=bool(p.get("verify", True)))
    return {
        "outdir": str(result.outdir),
        "files": result.files,
        "parity": sanitize(result.parity),
    }


def _write_wav(path: pathlib.Path, x: np.ndarray, fs: int) -> pathlib.Path:
    x = np.asarray(x, float).ravel()
    peak = float(np.max(np.abs(x)))
    if peak > 1e-12:
        x = x / peak * 0.95
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())
    return path


# --------------------------------------------------------------------------
# 任务（导出这类耗时操作）
# --------------------------------------------------------------------------
class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "running"
        self.progress = 0.0
        self.result = None
        self.error = None
        self.t0 = time.time()

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "progress": round(self.progress, 3), "result": self.result,
                "error": self.error, "elapsed_s": round(time.time() - self.t0, 2)}


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def start_job(kind: str, fn, *a, **kw) -> Job:
    job = Job(kind)
    with JOBS_LOCK:
        JOBS[job.id] = job

    def worker():
        try:
            job.progress = 0.2
            job.result = sanitize(fn(*a, **kw))
            job.progress = 1.0
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    threading.Thread(target=worker, daemon=True).start()
    return job


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
ROUTES = {
    "/api/arrays": ("GET", api_arrays),
    "/api/array/design": ("POST", api_array_design),
    "/api/beamform": ("POST", api_beamform),
    "/api/locate": ("POST", api_locate),
    "/api/imaging": ("POST", api_imaging),
    "/api/enhance": ("POST", api_enhance),
}

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                 ".css": "text/css", ".json": "application/json",
                 ".wav": "audio/wav", ".png": "image/png", ".svg": "image/svg+xml"}


def make_handler():
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = "MicArrayStudio/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):                  # 精简日志
            if "/api/" in (self.path or ""):
                sys.stderr.write("[api] %s %s\n" % (self.path, fmt % args))

        # ---- 工具 ----
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _error(self, code: int, msg: str, detail: str = "") -> None:
            self._json({"error": msg, "detail": detail[-2000:]}, code)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as e:
                raise ValueError(f"请求体不是合法 JSON: {e}")

        # ---- 路由 ----
        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path == "/api/health":
                    self._json({"status": "ok", "modules": {
                        k: (v is not None) for k, v in _MODULES.items()},
                        "engine": {"numpy": np.__version__}})
                    return
                if path.startswith("/api/jobs/"):
                    jid = path.rsplit("/", 1)[-1]
                    with JOBS_LOCK:
                        job = JOBS.get(jid)
                    if job is None:
                        self._error(404, "任务不存在")
                    else:
                        self._json(job.to_json())
                    return
                if path.startswith("/api/audio/"):
                    self._serve_file(SESSION / pathlib.Path(path).name, "audio/wav")
                    return
                if path == "/api/arrays":
                    self._json(sanitize(api_arrays({})))
                    return
                # 静态文件
                rel = "index.html" if path in ("/", "") else path.lstrip("/")
                target = (WEB_ROOT / rel).resolve()
                if not str(target).startswith(str(WEB_ROOT.resolve())):
                    self._error(403, "路径越界")
                    return
                self._serve_file(target)
            except Exception as e:
                self._error(500, f"{type(e).__name__}: {e}", traceback.format_exc())

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                params = self._body()
                if path == "/api/export/cpp":
                    job = start_job("export_cpp", api_export_cpp, params)
                    self._json({"job_id": job.id, "status": job.status})
                    return
                route = ROUTES.get(path)
                if route is None:
                    self._error(404, f"未知端点 {path}")
                    return
                _, fn = route
                t0 = time.time()
                data = sanitize(fn(params))
                if isinstance(data, dict):
                    data.setdefault("elapsed_s", round(time.time() - t0, 3))
                self._json(data)
            except Exception as e:
                # 把请求体一起记下来：字段名不匹配是最难查的一类问题
                self._error(400, f"{type(e).__name__}: {e}",
                            traceback.format_exc() +
                            "\n--- request body ---\n" +
                            json.dumps(params, ensure_ascii=False)[:1500])

        def _serve_file(self, target: pathlib.Path, ctype: str | None = None):
            if not target.exists() or not target.is_file():
                self._error(404, f"文件不存在: {target.name}")
                return
            data = target.read_bytes()
            ct = ctype or CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            self._send(200, data, ct)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8777, quiet: bool = False):
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer((host, port), make_handler())
    if not quiet:
        print(f"MicArray Studio 已启动: http://{host}:{port}")
        print(f"  静态目录: {WEB_ROOT}")
        print(f"  会话目录: {SESSION}")
        miss = [k for k, v in _MODULES.items() if v is None]
        if miss:
            print(f"  [warn] 模块不可用: {', '.join(miss)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
    return httpd


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MicArray Studio 后端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    serve(a.host, a.port)
