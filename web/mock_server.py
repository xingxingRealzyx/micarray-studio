#!/usr/bin/env python3
"""MicArray Studio —— 假后端（仅 Python 标准库）。

用途：在真正的后端（backend/）还没写好时，让 web/index.html 能被真实地跑起来自测。
它不依赖 numpy / engine 包，所有数据用 math + random 现造，但 **JSON schema 与
docs/CONTRACT.md 第 8 节完全一致**（包括二维数组形状、扁平数组 + shape 等）。

角度约定（CONTRACT 第 2 节，严格遵守）：
  阵列坐标系，阵元在 xy 平面 (z=0)，boresight = +z；
  az: xy 平面内从 +x 逆时针转向 +y，[-180, 180] 度；
  el: 从 xy 平面量向 +z，[-90, 90] 度（el=+90 表示正对阵列）；
  单位方向 u = [cos(el)cos(az), cos(el)sin(az), sin(el)]。

跑法：
    python3 web/mock_server.py --port 8765
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
import traceback
import wave
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import urlparse

C_SOUND = 343.0
HERE = os.path.dirname(os.path.abspath(__file__))
DB_FLOOR = -300.0


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def dir_vec(az: float, el: float) -> list:
    a, e = math.radians(az), math.radians(el)
    return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]


def dot3(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def pairwise(mics) -> list:
    out = []
    for i in range(len(mics)):
        for j in range(i + 1, len(mics)):
            dx = mics[i][0] - mics[j][0]
            dy = mics[i][1] - mics[j][1]
            dz = mics[i][2] - mics[j][2]
            out.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    return out


def redundancy_stats(mics) -> dict:
    """与 engine/geometry.py 的 redundancy_stats() 同定义（含 48 bin 直方图）。"""
    d = pairwise(mics)
    if not d:
        return {"n_pairs": 0, "n_unique": 0, "redundancy": 1.0, "max_multiplicity": 0,
                "coincident_pairs": 0, "histogram": []}
    ds = sorted(d)
    groups, start = [], 0
    for k in range(1, len(ds) + 1):
        if k == len(ds) or ds[k] - ds[start] > 1e-9:
            groups.append(k - start)
            start = k
    dmax = ds[-1]
    bins = 48
    hist = []
    if dmax > 0:
        for k in range(bins):
            lo = dmax * k / bins
            hi = dmax * (k + 1) / bins
            if k == bins - 1:
                cnt = sum(1 for v in ds if lo <= v <= hi)
            else:
                cnt = sum(1 for v in ds if lo <= v < hi)
            hist.append({"d": round(0.5 * (lo + hi), 6), "count": int(cnt)})
    return {"n_pairs": int(len(d)), "n_unique": int(len(groups)),
            "redundancy": round(len(d) / float(len(groups)), 6),
            "max_multiplicity": int(max(groups)),
            "coincident_pairs": int(sum(1 for v in d if v < 1e-6)),
            "histogram": hist}


def dirs_grid(az_bins: int, el_bins: int, az_range=(-180.0, 180.0), el_range=(-90.0, 90.0)):
    """返回 (dirs(G,3), shape(n_el,n_az), az_axis, el_axis)，C 序：el 外层、az 内层。"""
    az_bins = max(1, int(az_bins))
    el_bins = max(1, int(el_bins))
    if az_bins == 1:
        az_axis = [0.5 * (az_range[0] + az_range[1])]
    else:
        az_axis = [az_range[0] + (az_range[1] - az_range[0]) * i / (az_bins - 1)
                   for i in range(az_bins)]
    if el_bins == 1:
        el_axis = [0.5 * (el_range[0] + el_range[1])]
    else:
        el_axis = [el_range[0] + (el_range[1] - el_range[0]) * i / (el_bins - 1)
                   for i in range(el_bins)]
    dirs = []
    for e in el_axis:
        for a in az_axis:
            dirs.append(dir_vec(a, e))
    return dirs, (el_bins, az_bins), az_axis, el_axis


# ---------------------------------------------------------------------------
# 阵列几何
# ---------------------------------------------------------------------------
def build_mics(kind: str, params: dict):
    p = dict(params or {})
    n = int(p.get("n_mics", 8))
    kind = (kind or "ula").lower()
    pts = []
    if kind == "ula":
        d = float(p.get("spacing", 0.04))
        pts = [[(i - (n - 1) / 2.0) * d, 0.0, 0.0] for i in range(n)]
    elif kind == "uca":
        r = float(p.get("radius", 0.1))
        for i in range(n):
            t = 2 * math.pi * i / n
            pts.append([r * math.cos(t), r * math.sin(t), 0.0])
    elif kind == "grid":
        nx = max(1, int(p.get("nx", 4)))
        ny = max(1, int(p.get("ny", 4)))
        dx, dy = float(p.get("dx", 0.05)), float(p.get("dy", 0.05))
        for iy in range(ny):
            for ix in range(nx):
                pts.append([(ix - (nx - 1) / 2.0) * dx, (iy - (ny - 1) / 2.0) * dy, 0.0])
    elif kind == "spiral_arm":
        arms = max(1, int(p.get("n_arms", 4)))
        r = float(p.get("radius", 0.2))
        per = max(1, n // arms)
        for a in range(arms):
            base = 2 * math.pi * a / arms
            for k in range(per):
                t = (k + 1) / float(per)
                rr = r * t
                th = base + 1.6 * math.pi * t
                pts.append([rr * math.cos(th), rr * math.sin(th), 0.0])
    elif kind == "vogel":
        r = float(p.get("radius", 0.25))
        ga = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n):
            rr = r * math.sqrt((i + 0.5) / n)
            th = i * ga
            pts.append([rr * math.cos(th), rr * math.sin(th), 0.0])
    elif kind == "multi_ring":
        r = float(p.get("radius", 0.25))
        rings = max(1, int(p.get("n_rings", 3)))
        idx = 0
        for ri in range(rings):
            rr = r * (ri + 1) / rings
            cnt = max(1, n // rings) if ri < rings - 1 else max(1, n - idx)
            for k in range(cnt):
                if idx >= n:
                    break
                t = 2 * math.pi * k / cnt
                pts.append([rr * math.cos(t), rr * math.sin(t), 0.0])
                idx += 1
    else:  # custom 兜底：圆环
        r = float(p.get("radius", 0.1))
        for i in range(n):
            t = 2 * math.pi * i / n
            pts.append([r * math.cos(t), r * math.sin(t), 0.0])
    if not pts:
        pts = [[0.0, 0.0, 0.0]]
    return pts


def center_mics(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    return [[p[0] - cx, p[1] - cy, p[2] - cz] for p in pts], [cx, cy, cz]


def geometry_json(kind: str, params: dict, name=None, mics=None) -> dict:
    if mics is None:
        mics = build_mics(kind, params)
    mics, off = center_mics([[float(v) for v in m] for m in mics])
    mics = [[round(v, 9) for v in m] for m in mics]
    d = pairwise(mics)
    ap = max(d) if d else 0.0
    dmin = min(d) if d else 0.0
    red = redundancy_stats(mics)
    return {
        "name": name or kind,
        "kind": kind or "custom",
        "mics": mics,
        "center": [0.0, 0.0, 0.0],
        "center_offset": [round(v, 9) for v in off],
        "params": {k: v for k, v in (params or {}).items()},
        "meta": {
            "n_mics": len(mics),
            "aperture": round(ap, 6),
            "min_spacing": round(dmin, 6),
            "grating_free_freq": round(C_SOUND / (2.0 * dmin), 1) if dmin > 0 else None,
            "speed_of_sound": C_SOUND,
            "redundancy": red,
        },
    }


PRESETS = [
    {"name": "ula-8", "label": "均匀线阵 ULA-8", "kind": "ula", "n_mics": 8,
     "params": {"n_mics": 8, "spacing": 0.04}},
    {"name": "uca-16", "label": "均匀圆阵 UCA-16", "kind": "uca", "n_mics": 16,
     "params": {"n_mics": 16, "radius": 0.1}},
    {"name": "grid-16", "label": "矩形栅格 4x4", "kind": "grid", "n_mics": 16,
     "params": {"nx": 4, "ny": 4, "dx": 0.05, "dy": 0.05}},
    {"name": "spiral-arm-16", "label": "螺旋臂 16", "kind": "spiral_arm", "n_mics": 16,
     "params": {"n_mics": 16, "n_arms": 4, "radius": 0.2}},
    {"name": "vogel-32", "label": "Vogel 螺旋 32", "kind": "vogel", "n_mics": 32,
     "params": {"n_mics": 32, "radius": 0.25}},
    {"name": "multi-ring-24", "label": "多环阵 24", "kind": "multi_ring", "n_mics": 24,
     "params": {"n_mics": 24, "radius": 0.25, "n_rings": 3}},
]
PRESET_BY_NAME = {p["name"]: p for p in PRESETS}


def geom_from_body(body: dict) -> dict:
    """从请求 body 里解析几何：优先 mics，其次 preset，其次 kind+params。"""
    if isinstance(body.get("geometry"), dict):
        g = body["geometry"]
        if g.get("mics"):
            return geometry_json(g.get("kind", "custom"), g.get("params", {}),
                                 name=g.get("name"), mics=g["mics"])
        body = {**g, **{k: v for k, v in body.items() if k != "geometry"}}
    if body.get("mics"):
        return geometry_json("custom", {}, name=body.get("name") or "自定义", mics=body["mics"])
    pre = body.get("preset")
    if pre and pre in PRESET_BY_NAME:
        p = PRESET_BY_NAME[pre]
        return geometry_json(p["kind"], p["params"], name=p["name"])
    if pre:
        for p in PRESETS:
            if p["label"] == pre:
                return geometry_json(p["kind"], p["params"], name=p["name"])
    return geometry_json(body.get("kind", "uca"), body.get("params", {}))


# ---------------------------------------------------------------------------
# 波束图模型（延迟求和 + 方法相关的旁瓣抑制/主瓣锐化）
# ---------------------------------------------------------------------------
METHOD_SHAPE = {          # sharp: 主瓣锐化指数; floor_db: 旁瓣地板
    "delay_sum": (1.0, -13.5),
    "mvdr": (1.9, -26.0),
    "superdirective": (2.6, -32.0),
    "fi": (1.5, -20.0),
    "lcmv": (2.0, -28.0),
    "max_snr": (2.1, -27.0),
    "min_variance": (2.0, -26.0),
}


def array_factor_db(mics, kp, d: list, kps: list, sharp: float, floor_db: float,
                    mismatch: float = 0.0) -> float:
    """|sum_m exp(-j (k p_m·d - k p_m·u0))|^2 / M^2 的 dB，归一到指向处 0 dB。

    kp/kps 由调用方预先乘好 k，避免在 F×G×M 内层循环里重复做乘法和 list 取值。
    """
    re = im = 0.0
    for i in range(len(kp)):
        p = kp[i]
        ph = -(p[0] * d[0] + p[1] * d[1] + p[2] * d[2]) + kps[i]
        re += math.cos(ph)
        im += math.sin(ph)
    n = float(len(kp))
    pw = (re * re + im * im) / (n * n)
    if pw < 1e-12:
        pw = 1e-12
    db = 10.0 * math.log10(pw) * sharp
    if mismatch > 0:
        db -= mismatch * 3.0
    if db < floor_db:
        db = floor_db + (db - floor_db) * 0.15
    return max(db, DB_FLOOR)


def _coarse_axes(n, lo, hi, target, max_pts=20):
    """把 n 点的轴降采样到不超过 max_pts 点（至少覆盖同样的 [lo,hi]）。"""
    m = max(2, min(n, max_pts))
    return [lo + (hi - lo) * i / (m - 1) for i in range(m)]


def _resample(coarse, caz, cel, az_axis, el_axis):
    """把粗网格 (len(cel), len(caz)) 双线性插值到目标网格（C 序，el 外层）。"""
    na, ne = len(caz), len(cel)
    daz = (caz[-1] - caz[0]) / (na - 1) if na > 1 else 1.0
    dele = (cel[-1] - cel[0]) / (ne - 1) if ne > 1 else 1.0
    out = []
    for e in el_axis:
        fe = 0.0 if ne == 1 else min(max((e - cel[0]) / dele, 0.0), ne - 1.0)
        e0 = int(fe)
        e1 = min(ne - 1, e0 + 1)
        we = fe - e0
        for a in az_axis:
            fa = 0.0 if na == 1 else min(max((a - caz[0]) / daz, 0.0), na - 1.0)
            a0 = int(fa)
            a1 = min(na - 1, a0 + 1)
            wa = fa - a0
            v00 = coarse[e0 * na + a0]
            v01 = coarse[e0 * na + a1]
            v10 = coarse[e1 * na + a0]
            v11 = coarse[e1 * na + a1]
            out.append((v00 * (1 - wa) + v01 * wa) * (1 - we) + (v10 * (1 - wa) + v11 * wa) * we)
    return out


def base_pattern_db(mics, f: float, dirs, steer_u, mismatch: float = 0.0,
                    az_axis=None, el_axis=None):
    """延迟求和基准方向图（未加方法相关的锐化/地板）——所有方法共用，省 4× 计算量。

    为了在纯 Python 里跑得动，先在一张不超过 20×20 的粗网格上算，再双线性插值到
    请求的网格。方向图本身是光滑的，插值带来的误差对"假数据"完全可以接受。
    """
    k = 2.0 * math.pi * f / C_SOUND
    kp = [(k * m[0], k * m[1], k * m[2]) for m in mics]
    kps = [p[0] * steer_u[0] + p[1] * steer_u[1] + p[2] * steer_u[2] for p in kp]
    mm = mismatch * (f / 4000.0) ** 2 * 3.0
    n = float(len(mics))

    def af(d):
        re = im = 0.0
        for i in range(len(kp)):
            p = kp[i]
            ph = -(p[0] * d[0] + p[1] * d[1] + p[2] * d[2]) + kps[i]
            re += math.cos(ph)
            im += math.sin(ph)
        pw = (re * re + im * im) / (n * n)
        return 10.0 * math.log10(pw if pw > 1e-12 else 1e-12) - mm

    if not az_axis or not el_axis or len(dirs) <= 400:
        out = [af(d) for d in dirs]
    else:
        caz = _coarse_axes(len(az_axis), az_axis[0], az_axis[-1], 0, 20)
        cel = _coarse_axes(len(el_axis), el_axis[0], el_axis[-1], 0, 20)
        coarse = []
        for e in cel:
            for a in caz:
                coarse.append(af(dir_vec(a, e)))
        out = _resample(coarse, caz, cel, az_axis, el_axis)
    # 归一到峰值 0 dB：粗网格 + 插值会让峰值比理想指向低零点几 dB，
    # 归一化后"主瓣顶端 = 0 dB"这件事在任何网格密度下都成立（也更贴近真实后端）。
    mx = max(out)
    return [v - mx for v in out]


def beamwidth_az_deg(mics, f: float, steer_az: float, steer_el: float, sharp: float) -> float:
    """在 steer 的 el 上做 1° 精细 az 切割，找 -3 dB 主瓣宽度（度）。"""
    su = dir_vec(steer_az, steer_el)
    k = 2.0 * math.pi * f / C_SOUND
    kp = [[k * m[0], k * m[1], k * m[2]] for m in mics]
    kps = [p[0] * su[0] + p[1] * su[1] + p[2] * su[2] for p in kp]
    n = float(len(mics))
    prof = []
    best = -1e9
    for i in range(-90, 91):
        d = dir_vec(steer_az + i, steer_el)
        re = im = 0.0
        for j in range(len(kp)):
            q = kp[j]
            ph = -(q[0] * d[0] + q[1] * d[1] + q[2] * d[2]) + kps[j]
            re += math.cos(ph)
            im += math.sin(ph)
        pw = max((re * re + im * im) / (n * n), 1e-12)
        db = 10.0 * math.log10(pw) * sharp
        prof.append((i, db))
        if db > best:
            best = db
    thr = best - 3.0
    left = right = 0
    for i, db in prof:
        if i <= 0 and db >= thr:
            left = i
        if i >= 0 and db >= thr:
            right = i
    return float(max(0.5, right - left))


def parse_freqs(body: dict):
    fr = body.get("freqs")
    if isinstance(fr, list) and fr:
        f = [float(v) for v in fr][:128]
        return f, len(f) < len(fr)
    if isinstance(fr, dict):
        fmin = float(fr.get("fmin", 300.0))
        fmax = float(fr.get("fmax", 4000.0))
        n = max(1, min(128, int(fr.get("n", 24))))
        return [fmin + (fmax - fmin) * i / max(1, n - 1) for i in range(n)], False
    tr = body.get("freq_range") or {}
    fmin = float(tr.get("fmin", 300.0))
    fmax = float(tr.get("fmax", 4000.0))
    n = max(1, min(128, int(tr.get("n", 24))))
    return [fmin + (fmax - fmin) * i / max(1, n - 1) for i in range(n)], False


def parse_grid(body: dict):
    g = body.get("grid") or {}
    az_bins = max(1, min(361, int(g.get("az_bins", 72))))
    el_bins = max(1, min(181, int(g.get("el_bins", 37))))
    return az_bins, el_bins


def sources_of(body: dict):
    src = body.get("sources")
    if isinstance(src, list) and src:
        out = []
        for s in src:
            if isinstance(s, dict):
                out.append({"az": float(s.get("az", 0.0)), "el": float(s.get("el", 0.0)),
                            "distance": float(s.get("distance", 1.5)),
                            "amplitude": float(s.get("amplitude", 1.0))})
            elif isinstance(s, (list, tuple)) and len(s) >= 2:
                out.append({"az": float(s[0]), "el": float(s[1]), "distance": 1.5,
                            "amplitude": 1.0})
        return out
    if body.get("az") is not None or body.get("el") is not None:
        return [{"az": float(body.get("az", 0.0)), "el": float(body.get("el", 0.0)),
                 "distance": float(body.get("distance", 1.5)), "amplitude": 1.0}]
    return [{"az": 30.0, "el": 60.0, "distance": 1.5, "amplitude": 1.0}]


# ---------------------------------------------------------------------------
# WAV（标准库 wave 写出，给 <audio> 播放）
# ---------------------------------------------------------------------------
_AUDIO_CACHE = OrderedDict()
_AUDIO_LOCK = threading.Lock()


def synth_wav(enhanced: bool, seed: int = 7) -> bytes:
    fs, n = 16000, 16000
    rnd = random.Random(seed)
    buf = bytearray()
    for i in range(n):
        t = i / fs
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 1.7 * t)      # 音节包络
        s = (math.sin(2 * math.pi * 220 * t) + 0.5 * math.sin(2 * math.pi * 440 * t)
             + 0.3 * math.sin(2 * math.pi * 880 * t)) * env / 1.8
        noise = rnd.uniform(-1.0, 1.0) * (0.05 if enhanced else 0.30)
        v = clamp(0.6 * s + noise, -1.0, 1.0)
        buf += int(v * 32000).to_bytes(2, "little", signed=True)
    bio = BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(bytes(buf))
    return bio.getvalue()


def audio_bytes(name: str) -> bytes:
    with _AUDIO_LOCK:
        if name in _AUDIO_CACHE:
            _AUDIO_CACHE.move_to_end(name)
            return _AUDIO_CACHE[name]
    data = synth_wav(enhanced=name.startswith("enh"), seed=abs(hash(name)) % 1000)
    with _AUDIO_LOCK:
        _AUDIO_CACHE[name] = data
        while len(_AUDIO_CACHE) > 32:
            _AUDIO_CACHE.popitem(last=False)
    return data


# ---------------------------------------------------------------------------
# 导出任务（后台线程推进 progress）
# ---------------------------------------------------------------------------
_JOBS = {}
_JOB_LOCK = threading.Lock()
_JOB_SEQ = [0]
EXPORT_FILES = ["include/mas_array_config.hpp", "include/mas_beamformer.hpp",
                "src/mas_beamformer.cpp", "src/mas_fft.cpp", "tests/test_parity.cpp",
                "tests/vectors/steering.json", "CMakeLists.txt"]
STEPS = ["生成 include/mas_array_config.hpp ...", "生成 src/mas_fft.cpp ...",
         "生成 src/mas_beamformer.cpp ...", "写测试向量 tests/vectors/*.json ...",
         "写 CMakeLists.txt ...", "cmake 配置 ...", "编译 mas_core ...",
         "运行 test_parity ..."]


def start_export(body: dict) -> dict:
    with _JOB_LOCK:
        _JOB_SEQ[0] += 1
        jid = "exp-%04d" % _JOB_SEQ[0]
        job = {"job_id": jid, "status": "running", "progress": 0.0,
               "message": STEPS[0], "result": None, "t0": time.time(),
               "skipped": bool(body.get("simulate_fail")) or body.get("verify") is False,
               "name": body.get("name") or "beamformer", "step": 0}
        _JOBS[jid] = job

    def run():
        for k in range(1, len(STEPS) + 1):
            time.sleep(0.15)
            with _JOB_LOCK:
                job["progress"] = round(k / float(len(STEPS)), 3)
                job["message"] = STEPS[k - 1] if k <= len(STEPS) else "完成"
                job["step"] = k
        with _JOB_LOCK:
            if job["skipped"]:
                parity = {"status": "skipped", "tol": 1e-6, "errors": {},
                          "stdout": "未编译（verify=false 或未找到 cmake）"}
            else:
                parity = {"status": "pass", "tol": 1e-6,
                          "errors": {"fft": 3.1e-15, "window_table": 1.2e-16,
                                     "steering": 8.4e-16, "beamform": 2.2e-15},
                          "stdout": "[fft] 最大相对误差 3.1e-15\n"
                                    "[window_table] 最大相对误差 1.2e-16\n"
                                    "[steering] 最大相对误差 8.4e-16\n"
                                    "[beamform] 最大相对误差 2.2e-15"}
            job["result"] = {
                "files": list(EXPORT_FILES),
                "outdir": "export/" + str(job["name"]),
                "compile_log": "[ 50%] Building CXX object CMakeFiles/mas_core.dir/src/mas_fft.cpp.o\n"
                               "[100%] Built target test_parity\n",
                "parity": parity,
                "elapsed_s": round(time.time() - job["t0"], 3),
            }
            job["status"] = "done"
            job["progress"] = 1.0
            job["message"] = "完成"

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid, "status": "running", "progress": 0.0, "files": [],
            "compile_log": "", "parity": None}


def job_json(jid: str):
    with _JOB_LOCK:
        job = _JOBS.get(jid)
        if job is None:
            return None
        return {"job_id": job["job_id"], "status": job["status"],
                "progress": job["progress"], "message": job["message"],
                "result": job["result"]}


# ---------------------------------------------------------------------------
# 各端点
# ---------------------------------------------------------------------------
def ep_beamform(body: dict) -> dict:
    geom = geom_from_body(body)
    mics = geom["mics"]
    az_bins, el_bins = parse_grid(body)
    if az_bins * el_bins > 20000:
        el_bins = max(1, 20000 // az_bins)
        truncated = True
    else:
        truncated = False
    dirs, shape, az_ax, el_ax = dirs_grid(az_bins, el_bins)
    freqs, ftrunc = parse_freqs(body)
    steer = body.get("steer") or [0.0, 90.0]
    if isinstance(steer, dict):
        steer = [float(steer.get("az", 0.0)), float(steer.get("el", 90.0))]
    saz, sel = float(steer[0]), float(steer[1])
    su = dir_vec(saz, sel)
    mm = body.get("mismatch") or {}
    mismatch = (float(mm.get("phase_std_deg", 0.0)) / 10.0
                + float(mm.get("gain_std", 0.0)) * 10.0)
    methods = body.get("methods") or ["delay_sum", "mvdr", "superdirective", "fi"]
    if isinstance(methods, str):
        methods = [methods]
    # 阵列的"低频拐点"：ka ~ 1，低于它任何超指向方法都要牺牲 WNG
    ap = geom["meta"]["aperture"] or 0.1
    f_knee = C_SOUND / (2.0 * math.pi * max(ap, 1e-3))
    m_db = 10.0 * math.log10(len(mics))
    # 先把与"方法"无关的延迟求和基准图算一遍（4 个方法共用），否则纯 Python 里
    # 每个方法都要重算 F×G×M 次三角运算，请求要 3 秒以上。
    base_rows = [base_pattern_db(mics, f, dirs, su, mismatch, az_ax, el_ax) for f in freqs]
    out_methods = {}
    for mth in methods:
        sharp, floor_db = METHOD_SHAPE.get(mth, (1.0, -14.0))
        # 方法代价：超指向方法低频 WNG 掉得多（与 CONTRACT 第 6 节的描述一致）
        wng_drop = {"delay_sum": 0.0, "fi": 8.0, "mvdr": 16.0,
                    "superdirective": 26.0}.get(mth, 12.0)
        di_boost = {"delay_sum": 0.0, "fi": 1.5, "mvdr": 2.6,
                    "superdirective": 4.0}.get(mth, 1.0)
        pat, bw, wng_db, di_db = [], [], [], []
        for fi, f in enumerate(freqs):
            base = base_rows[fi]
            row = []
            for db0 in base:
                db = db0 * sharp
                if db < floor_db:
                    db = floor_db + (db - floor_db) * 0.15
                row.append(round(clamp(db, DB_FLOOR, 40.0), 3))
            pat.append(row)
            ratio = (f / f_knee) ** 2
            wng_db.append(round(m_db - wng_drop / (1.0 + ratio) - mismatch * 1.5, 3))
            di_db.append(round(min(m_db + di_boost,
                                   (m_db + di_boost) * ratio / (1.0 + ratio)), 3))
            bw.append(round(beamwidth_az_deg(mics, f, saz, sel, sharp), 3))
        out_methods[mth] = {
            "freqs": [round(f, 3) for f in freqs],
            "pattern_db": pat,
            "beamwidth_deg": bw,
            "metrics": {"wng_db": wng_db, "di_db": di_db,
                        "wng": [round(10.0 ** (v / 10.0), 6) for v in wng_db],
                        "di": [round(10.0 ** (v / 10.0), 6) for v in di_db],
                        "wng_min_db": min(wng_db) if wng_db else None,
                        "di_max_db": max(di_db) if di_db else None},
            "extra": {"note": "mock 数据：非物理仿真"},
        }
    res = {"dirs": [[round(v, 9) for v in d] for d in dirs],
           "shape": [shape[0], shape[1]], "az_axis": az_ax, "el_axis": el_ax,
           "freqs": [round(f, 3) for f in freqs], "steer": [saz, sel],
           "geometry": geom, "methods": out_methods}
    if truncated or ftrunc:
        res["truncated"] = True
    return res


def _gauss_map(dirs, shape, centers, sigma_deg, base=0.02):
    """在方向网格上叠几个高斯瓣（按立体角距离），归一到 max=1。"""
    vals = [base] * len(dirs)
    inv = 1.0 / (2.0 * sigma_deg * sigma_deg)
    for c in centers:
        cu = dir_vec(c["az"], c["el"])
        w = c.get("amplitude", 1.0)
        for i, d in enumerate(dirs):
            dot = clamp(dot3(d, cu), -1.0, 1.0)
            ang = math.degrees(math.acos(dot))
            vals[i] += w * math.exp(-ang * ang * inv)
    mx = max(vals) or 1.0
    return [clamp(v / mx, 0.0, 1.0) for v in vals]


def ep_locate(body: dict) -> dict:
    geom = geom_from_body(body)
    az_bins, el_bins = parse_grid(body)
    dirs, shape, az_ax, el_ax = dirs_grid(az_bins, el_bins)
    srcs = sources_of(body)
    snr = float((body.get("noise") or {}).get("snr_db", 15.0))
    method = body.get("method") or "srp_phat"
    rnd = random.Random(int(abs(snr) * 97) + len(srcs))
    sigma = clamp(4.0 - 0.05 * snr, 1.2, 4.0)
    mp = _gauss_map(dirs, shape, [{"az": s["az"], "el": s["el"],
                                   "amplitude": s["amplitude"]} for s in srcs], sigma)
    est = []
    for s in srcs:
        jitter = clamp(2.5 - 0.05 * snr, 0.3, 2.5)
        est.append([round(((s["az"] + rnd.uniform(-jitter, jitter)) + 180) % 360 - 180, 4),
                    round(clamp(s["el"] + rnd.uniform(-jitter, jitter), -90, 90), 4)])
    est_xyz = []
    for (a, e), s in zip(est, srcs):
        u = dir_vec(a, e)
        r = s["distance"]
        est_xyz.append([round(u[0] * r, 6), round(u[1] * r, 6), round(u[2] * r, 6)])
    return {"map": [round(v, 6) for v in mp], "shape": [shape[0], shape[1]],
            "az_axis": az_ax, "el_axis": el_ax, "map_scale": 1.0, "method": method,
            "est_dirs": est, "est_xyz": est_xyz,
            "peak_vals": [1.0 for _ in est], "n_sources": len(est),
            "truth": [[round(s["az"], 4), round(s["el"], 4)] for s in srcs],
            "front_back_ambiguous": True,
            "dirs": [[round(v, 9) for v in d] for d in dirs]}


def ep_imaging(body: dict) -> dict:
    geom = geom_from_body(body)
    az_bins, el_bins = parse_grid(body)
    dirs, shape, az_ax, el_ax = dirs_grid(az_bins, el_bins)
    srcs = sources_of(body)
    method = body.get("method") or "damas"
    dirty = _gauss_map(dirs, shape, srcs, 7.5)
    deconv = _gauss_map(dirs, shape, srcs, 3.0)
    psf = _gauss_map(dirs, shape, [{"az": 0.0, "el": 90.0}], 6.0)
    return {"dirty": [round(v, 6) for v in dirty], "deconv": [round(v, 6) for v in deconv],
            "psf": [round(v, 6) for v in psf], "shape": [shape[0], shape[1]],
            "az_axis": az_ax, "el_axis": el_ax, "method": method,
            "dynamic_range_db": 30.0,
            "dirs": [[round(v, 9) for v in d] for d in dirs]}


def ep_enhance(body: dict) -> dict:
    geom = geom_from_body(body)
    srcs = sources_of(body)
    method = body.get("method") or "mvdr"
    snr_in = float((body.get("noise") or {}).get("snr_db", 3.0))
    gain = 6.5 + 2.0 * math.log10(max(1, geom["meta"]["n_mics"])) - abs(srcs[0]["el"] - 90) * 0.02
    snr_out = snr_in + gain
    sdr_in = snr_in - 4.6
    sdr_out = sdr_in + gain * 0.97
    npts = 40
    curve = {"t": [], "snr_in_db": [], "snr_out_db": [], "si_sdr_in_db": [], "si_sdr_out_db": []}
    for i in range(npts):
        x = i / float(npts - 1)
        conv = 1.0 - math.exp(-3.2 * x)
        curve["t"].append(round(x, 4))
        curve["snr_in_db"].append(round(snr_in + 0.15 * math.sin(6 * x), 3))
        curve["snr_out_db"].append(round(snr_in + gain * conv, 3))
        curve["si_sdr_in_db"].append(round(sdr_in + 0.15 * math.cos(5 * x), 3))
        curve["si_sdr_out_db"].append(round(sdr_in + gain * 0.97 * conv, 3))
    sid = abs(hash(method + str(round(snr_in, 2)) + str(len(geom["mics"])))) % 100000
    return {"snr_in": round(snr_in, 3), "snr_out": round(snr_out, 3),
            "snr_gain_db": round(snr_out - snr_in, 3),
            "si_sdr_in": round(sdr_in, 3), "si_sdr_out": round(sdr_out, 3),
            "si_sdr_gain_db": round(sdr_out - sdr_in, 3),
            "stoi_proxy_in": round(clamp(0.45 + 0.02 * snr_in, 0.05, 0.98), 3),
            "stoi_proxy_out": round(clamp(0.55 + 0.03 * snr_in + 0.012 * gain, 0.05, 0.99), 3),
            "stoi_proxy": round(clamp(0.55 + 0.03 * snr_in + 0.012 * gain, 0.05, 0.99), 3),
            "method": method, "fs": 16000, "n_samples": 16000, "curve": curve,
            "audio_url": "/api/audio/enh_%d.wav" % sid,
            "noisy_audio_url": "/api/audio/noisy_%d.wav" % sid,
            "metrics": {method: {"n_mics": geom["meta"]["n_mics"], "snr_gain_db": round(gain, 3)}}}


def ep_enhance_demo(body: dict) -> dict:
    """无参考的"听感 demo"分支：只给可播放音频 + 电平，指标为 null（与引擎 _no_ref_metrics 对齐）。"""
    r = ep_enhance(body)
    r.update({"snr_in": None, "snr_out": None, "si_sdr_in": None, "si_sdr_out": None,
              "stoi_proxy": None, "output_level_db": -18.4, "demo": True,
              "noise_reduction_est_db": 8.0})
    return r


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "MicArrayMock/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # 精简日志
        if os.environ.get("MAS_MOCK_VERBOSE"):
            BaseHTTPRequestHandler.log_message(self, fmt, *args)

    # ---- 输出 helpers ----
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _err(self, code: int, msg: str):
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return {}
        try:
            d = json.loads(raw.decode("utf-8"))
        except Exception:
            raise ValueError("请求体不是合法 JSON")
        return d if isinstance(d, dict) else {"value": d}

    # ---- 路由 ----
    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method: str):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._static(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
            if path == "/api/health":
                return self._json({"status": "ok", "service": "micarray-studio-mock",
                                   "version": "1.0", "engine": "mock-stdlib", "numpy": False})
            if path == "/api/arrays":
                cur = geometry_json(PRESETS[0]["kind"], PRESETS[0]["params"],
                                    name=PRESETS[0]["name"])
                return self._json({"presets": PRESETS, "current": cur})
            if path.startswith("/api/audio/") and path.endswith(".wav"):
                name = os.path.basename(path)[:-4]
                return self._send(200, audio_bytes(name), "audio/wav")
            if path.startswith("/api/jobs/"):
                jid = path[len("/api/jobs/"):].strip("/")
                j = job_json(jid)
                if j is None:
                    return self._err(404, "job not found: %s" % jid)
                return self._json(j)
            if method != "POST":
                return self._err(404, "未知路径: %s（POST 端点需要 POST）" % path)

            body = self._body()
            if path == "/api/array/design":
                return self._json(geom_from_body(body))
            if path == "/api/beamform":
                return self._json(ep_beamform(body))
            if path == "/api/locate":
                return self._json(ep_locate(body))
            if path == "/api/imaging":
                return self._json(ep_imaging(body))
            if path == "/api/enhance":
                mode = (body.get("mode") or "").lower()
                if mode in ("demo", "no_ref", "noref"):
                    return self._json(ep_enhance_demo(body))
                return self._json(ep_enhance(body))
            if path == "/api/export/cpp":
                return self._json(start_export(body))
            return self._err(404, "未知 API 路径: %s" % path)
        except ValueError as e:
            self._err(400, str(e))
        except BrokenPipeError:
            pass
        except Exception as e:                                     # noqa: BLE001
            self._json({"error": "%s: %s" % (type(e).__name__, e),
                        "traceback": traceback.format_exc()[-2000:]}, 500)

    def _static(self, fp: str, ctype: str):
        if not os.path.isfile(fp):
            return self._err(404, "找不到 %s —— 请确认 web/index.html 存在" % fp)
        with open(fp, "rb") as fh:
            self._send(200, fh.read(), ctype)


def main():
    ap = argparse.ArgumentParser(description="MicArray Studio 假后端（标准库实现）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print("MicArray Studio mock server -> http://%s:%d  (Ctrl-C 退出)" % (a.host, a.port),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
