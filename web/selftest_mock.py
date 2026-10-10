#!/usr/bin/env python3
"""web/mock_server.py 的自测脚本（仅标准库）。

用法：
    python3 web/selftest_mock.py          # 自己拉起 mock server，跑完自动关掉
    python3 web/selftest_mock.py --url http://127.0.0.1:8765   # 测已经在跑的服务

校验重点：CONTRACT 第 8 节的 schema、二维/扁平数组形状、数值有限性、job 轮询状态机。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
CHECKS = [0]


def ck(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL  " + msg)
    return bool(cond)


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read()
        ctype = r.headers.get("Content-Type", "")
        return r.status, raw, ctype


def jget(url, timeout=10):
    """GET + 解析 JSON；4xx/5xx 也解析 body（用来验证错误响应仍是 JSON）。"""
    try:
        st, raw, _ = get(url, timeout)
    except urllib.error.HTTPError as e:
        st, raw = e.code, e.read()
    txt = raw.decode("utf-8", "replace")
    ck("NaN" not in txt and "Infinity" not in txt, "响应含非法 JSON 数值: " + url)
    return st, json.loads(txt)


def post(url, body, timeout=30):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            st = r.status
    except urllib.error.HTTPError as e:
        raw, st = e.read(), e.code
    txt = raw.decode("utf-8", "replace")
    return st, json.loads(txt), raw


def finite_flat(seq):
    for v in seq:
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
    return True


def geom_checks(g, label):
    ck(isinstance(g.get("mics"), list) and len(g["mics"]) > 0, label + ": mics 为空")
    mics = g["mics"]
    ck(all(len(m) == 3 for m in mics), label + ": mics 不是 (M,3)")
    ck(all(abs(m[2]) < 1e-9 for m in mics), label + ": 阵元不在 xy 平面 (z!=0)")
    cx = sum(m[0] for m in mics) / len(mics)
    cy = sum(m[1] for m in mics) / len(mics)
    ck(abs(cx) < 1e-6 and abs(cy) < 1e-6, label + ": 质心未平移到原点")
    meta = g.get("meta") or {}
    ck(meta.get("n_mics") == len(mics), label + ": meta.n_mics 与 len(mics) 不一致")
    d = [math.dist(mics[i], mics[j]) for i in range(len(mics)) for j in range(i + 1, len(mics))]
    ck(abs(meta.get("aperture", -1) - max(d)) < 1e-5, label + ": aperture 不等于最大阵元间距")
    ck(abs(meta.get("min_spacing", -1) - min(d)) < 1e-5, label + ": min_spacing 不等于最近间距")
    gff = meta.get("grating_free_freq")
    ck(gff is None or abs(gff - 343.0 / (2 * min(d))) < 0.2, label + ": grating_free_freq 不对")
    red = meta.get("redundancy") or {}
    hist = red.get("histogram")
    ck(isinstance(hist, list) and len(hist) == 48, label + ": histogram 不是 48 bin")
    ck(sum(b["count"] for b in hist) == red.get("n_pairs"), label + ": histogram 计数 != n_pairs")
    ck(red.get("n_pairs") == len(d), label + ": n_pairs 不对")
    ck(0.9 <= red.get("redundancy", 0) <= len(d) + 0.01, label + ": redundancy 越界")
    ck(finite_flat([b["d"] for b in hist]), label + ": histogram d 含非有限值")


def shape_of(res, key_map, label, shape):
    n = shape[0] * shape[1]
    for key in key_map:
        arr = res.get(key)
        ck(isinstance(arr, list) and len(arr) == n, "%s: %s 长度 != n_el*n_az" % (label, key))
        ck(finite_flat(arr), "%s: %s 含非有限值" % (label, key))


def uv_spread(dirs, shape, vals):
    """以最大值为中心的角度二阶矩（度）——用来比较主瓣宽度。"""
    i0 = max(range(len(vals)), key=lambda i: vals[i])
    u0 = dirs[i0]
    num = den = 0.0
    for d, v in zip(dirs, vals):
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(d, u0))))
        ang = math.degrees(math.acos(dot))
        num += v * ang * ang
        den += v
    return math.sqrt(num / max(den, 1e-12))


def half_power_cells(vals):
    """半功率（>=0.5，即 -3 dB）以上的网格点数：主瓣立体角的代理量。

    不要用"全图角度二阶矩"做主瓣宽度的判据：它会被均匀的旁瓣地板主导，
    得出"主瓣越窄离散度越大"的反直觉结论。
    """
    return sum(1 for v in vals if v >= 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    proc = None
    base = a.url
    if base is None:
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_server.py"),
                                 "--port", str(a.port)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        base = "http://127.0.0.1:%d" % a.port
        for _ in range(60):
            try:
                st, _, _ = get(base + "/api/health", timeout=1)
                if st == 200:
                    break
            except Exception:
                time.sleep(0.15)
    base = base.rstrip("/")
    print("self-test against", base)
    try:
        run_all(base)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
    print("\n%d checks, %d failures" % (CHECKS[0], len(FAILS)))
    if FAILS:
        print("FAILED:")
        for f in FAILS:
            print(" -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


def run_all(base):
    # 1) health
    st, h = jget(base + "/api/health")
    ck(st == 200 and h.get("status") == "ok", "/api/health 未返回 200/ok")

    # 2) arrays
    st, ar = jget(base + "/api/arrays")
    presets = ar.get("presets")
    ck(isinstance(presets, list) and len(presets) >= 6, "/api/arrays presets < 6")
    for p in presets:
        ck(all(k in p for k in ("name", "label", "kind", "n_mics", "params")),
           "preset 字段缺失: %s" % p)
    ck(len({p["n_mics"] for p in presets}) >= 3, "presets 的 n_mics 缺少多样性")
    geom_checks(ar.get("current") or {}, "arrays.current")

    # 3) array/design（每个 preset 都过一遍）
    geoms = {}
    for p in presets:
        st, g, _ = post(base + "/api/array/design", {"preset": p["name"]})
        ck(st == 200, "array/design %s HTTP %d" % (p["name"], st))
        geom_checks(g, "design[%s]" % p["name"])
        ck(len(g["mics"]) == p["n_mics"], "design[%s] 阵元数与 preset 不符" % p["name"])
        geoms[p["name"]] = g
    g = geoms[presets[0]["name"]]
    # mics 直传分支
    st, g2, _ = post(base + "/api/array/design",
                     {"mics": [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]],
                      "name": "自定义"})
    ck(st == 200 and len(g2["mics"]) == 3, "array/design 的 mics 直传分支失败")
    geom_checks(g2, "design[custom]")

    # 4) beamform
    body = {"geometry": g, "methods": ["delay_sum", "mvdr", "superdirective", "fi"],
            "steer": [30.0, 60.0], "freqs": {"fmin": 500, "fmax": 4000, "n": 8},
            "grid": {"az_bins": 72, "el_bins": 37}}
    st, bf, _ = post(base + "/api/beamform", body)
    ck(st == 200, "beamform HTTP %d" % st)
    shape = bf.get("shape")
    ck(isinstance(shape, list) and len(shape) == 2, "beamform shape 不是 [n_el,n_az]")
    G = shape[0] * shape[1]
    dirs = bf["dirs"]
    ck(len(dirs) == G, "beamform dirs 长度 != n_el*n_az")
    ck(all(len(d) == 3 for d in dirs), "beamform dirs 不是 (G,3)")
    for d in dirs:
        ck(abs(sum(v * v for v in d) - 1.0) < 1e-6, "dirs 不是单位向量")
        break
    ck(len(bf["az_axis"]) == shape[1] and len(bf["el_axis"]) == shape[0],
       "az_axis/el_axis 长度与 shape 不符")
    ck(len(bf["freqs"]) == 8, "beamform freqs 长度不对")
    for mth, m in bf["methods"].items():
        pat = m.get("pattern_db")
        ck(isinstance(pat, list) and len(pat) == 8, "%s: pattern_db 不是 F 行" % mth)
        ck(all(isinstance(r, list) and len(r) == G for r in pat),
           "%s: pattern_db 不是 (F,G) 二维数组" % mth)
        ck(finite_flat([v for r in pat for v in r]), "%s: pattern_db 含非有限值" % mth)
        ck(all(v <= 1e-6 for r in pat for v in r), "%s: 存在 >0 dB 的旁瓣" % mth)
        for r in pat:
            ck(abs(max(r)) < 0.75, "%s: 峰值不在 0 dB 附近（max=%.3f）" % (mth, max(r)))
        ck(len(m["metrics"]["wng_db"]) == 8 and len(m["metrics"]["di_db"]) == 8,
           "%s: metrics 长度不对" % mth)
        ck(len(m["beamwidth_deg"]) == 8, "%s: beamwidth_deg 长度不对" % mth)
    # 主瓣应随方法变窄：delay_sum > mvdr > superdirective（用 -3dB 宽度代理）
    w = {m: sum(bf["methods"][m]["beamwidth_deg"]) for m in bf["methods"]}
    ck(w["delay_sum"] > w["superdirective"], "超指向方法的主瓣没有比延迟求和更窄: %s" % w)

    # el_bins=1 的退化网格
    st, bf1, _ = post(base + "/api/beamform", {
        "geometry": g, "methods": ["delay_sum"], "steer": [0.0, 0.0],
        "freqs": {"fmin": 1000, "fmax": 1000, "n": 1},
        "grid": {"az_bins": 181, "el_bins": 1}})
    ck(st == 200 and bf1["shape"] == [1, 181], "el_bins=1 的退化网格失败")
    ck(len(bf1["methods"]["delay_sum"]["pattern_db"][0]) == 181, "el_bins=1 的 pattern 长度不对")

    # 5) locate
    st, lo, _ = post(base + "/api/locate", {
        "geometry": g, "method": "srp_phat", "grid": {"az_bins": 72, "el_bins": 37},
        "sources": [{"az": 30.0, "el": 60.0, "distance": 1.5, "amplitude": 1.0},
                    {"az": -45.0, "el": 20.0, "distance": 2.0, "amplitude": 0.7}],
        "noise": {"snr_db": 15.0}})
    ck(st == 200, "locate HTTP %d" % st)
    sh = lo["shape"]
    shape_of(lo, ["map"], "locate", sh)
    ck(len(lo["dirs"]) == sh[0] * sh[1], "locate dirs 长度不对")
    ck(abs(max(lo["map"]) - 1.0) < 1e-6 and min(lo["map"]) >= 0.0, "locate map 未归一到 0..1")
    ck(len(lo["est_dirs"]) == 2 and len(lo["truth"]) == 2, "locate est_dirs/truth 数量不对")
    for (ea, ee), (ta, te) in zip(lo["est_dirs"], lo["truth"]):
        ck(abs(ea - ta) < 4.0 and abs(ee - te) < 4.0, "locate 估计值离真值太远")
    ck(len(lo["est_xyz"]) == 2 and len(lo["est_xyz"][0]) == 3, "locate est_xyz 形状不对")
    ck(all(k in lo for k in ("az_axis", "el_axis", "method", "peak_vals", "n_sources")),
       "locate 缺字段")

    # 6) imaging：deconv 主瓣必须明显窄于 dirty
    st, im, _ = post(base + "/api/imaging", {
        "geometry": g, "method": "damas", "grid": {"az_bins": 72, "el_bins": 37},
        "sources": [{"az": 10.0, "el": 70.0, "distance": 1.5, "amplitude": 1.0}]})
    ck(st == 200, "imaging HTTP %d" % st)
    sh = im["shape"]
    shape_of(im, ["dirty", "deconv"], "imaging", sh)
    ck(im.get("psf") is None or len(im["psf"]) == sh[0] * sh[1], "imaging psf 长度不对")
    ck(abs(max(im["dirty"]) - 1.0) < 1e-6 and abs(max(im["deconv"]) - 1.0) < 1e-6,
       "imaging dirty/deconv 未归一")
    ck(isinstance(im.get("method"), str) and im["method"], "imaging method 缺失")
    hc_d = half_power_cells(im["dirty"])
    hc_c = half_power_cells(im["deconv"])
    ck(hc_d > 0 and hc_c > 0, "imaging 没有找到半功率主瓣")
    ck(hc_c < hc_d * 0.6,
       "deconv 主瓣没有明显窄于 dirty（半功率格点 %d vs %d）" % (hc_c, hc_d))

    # 7) enhance + audio
    st, en, _ = post(base + "/api/enhance", {
        "geometry": g, "method": "mvdr", "noise": {"snr_db": 3.0},
        "sources": [{"az": 20.0, "el": 70.0, "distance": 1.5, "amplitude": 1.0}]})
    ck(st == 200, "enhance HTTP %d" % st)
    for k in ("snr_in", "snr_out", "si_sdr_in", "si_sdr_out", "curve",
              "audio_url", "noisy_audio_url", "stoi_proxy"):
        ck(k in en, "enhance 缺字段 %s" % k)
    ck(en["snr_out"] > en["snr_in"], "enhance: snr_out <= snr_in")
    ck(en["si_sdr_out"] > en["si_sdr_in"], "enhance: si_sdr_out <= si_sdr_in")
    cur = en["curve"]
    lens = {len(v) for v in cur.values() if isinstance(v, list)}
    ck(len(lens) == 1 and lens.pop() >= 5, "enhance.curve 各序列长度不一致")
    for url in (en["audio_url"], en["noisy_audio_url"]):
        st, raw, ctype = get(base + url)
        ck(st == 200 and "audio/wav" in ctype, "音频 %s 类型不对" % url)
        with wave.open(BytesIO(raw), "rb") as w:
            ck(w.getnframes() > 1000 and w.getframerate() == 16000,
               "音频 %s 内容不对" % url)

    # 8) export + jobs
    st, ex, _ = post(base + "/api/export/cpp", {"geometry": g, "method": "delay_sum",
                                                "name": "beamformer", "verify": True})
    ck(st == 200 and ex.get("job_id"), "export/cpp 未返回 job_id")
    jid = ex["job_id"]
    seen_progress, status = [], None
    result = None
    t0 = time.time()
    while time.time() - t0 < 8:
        st, j = jget(base + "/api/jobs/" + jid)
        ck(st == 200 and j.get("job_id") == jid, "jobs 返回的 job_id 不匹配")
        seen_progress.append(j.get("progress"))
        status = j.get("status")
        if status == "done":
            result = j.get("result")
            break
        time.sleep(0.12)
    ck(status == "done", "export job 未在 8s 内完成（status=%s）" % status)
    ck(len(seen_progress) >= 2 and seen_progress[-1] == 1.0, "job progress 没有推进到 1.0")
    ck(result is not None and isinstance(result.get("files"), list) and result["files"],
       "job result.files 为空")
    par = (result or {}).get("parity") or {}
    ck(par.get("status") == "pass", "parity.status != pass")
    errs = par.get("errors") or {}
    ck(errs and all(v <= par.get("tol", 1e-6) for v in errs.values()),
       "parity.errors 超过容差: %s" % errs)

    # verify=false -> skipped
    st, ex2, _ = post(base + "/api/export/cpp", {"geometry": g, "verify": False})
    jid2 = ex2["job_id"]
    t0 = time.time()
    res2 = None
    while time.time() - t0 < 8:
        st, j = jget(base + "/api/jobs/" + jid2)
        if j.get("status") == "done":
            res2 = j.get("result")
            break
        time.sleep(0.12)
    ck((res2 or {}).get("parity", {}).get("status") == "skipped", "verify=false 未返回 skipped")

    # 9) 错误处理
    st, e404 = jget(base + "/api/jobs/does-not-exist")
    ck(st == 404 and isinstance(e404.get("error"), str), "未知 job 未返回 404 JSON")
    st, e404b, _ = post(base + "/api/nope", {})
    ck(st == 404 and isinstance(e404b.get("error"), str), "未知 API 未返回 404 JSON")
    st, e400, _ = post(base + "/api/array/design", {"x": 1})
    ck(st == 200, "空 body 的 design 应当走默认分支并返回 200")

    # 10) 静态 index.html + CORS
    try:
        st, raw, ctype = get(base + "/index.html")
        ck(st == 200 and b"<html" in raw.lower(), "/index.html 未返回 HTML")
    except urllib.error.HTTPError as e:
        ck(False, "/index.html HTTP %d" % e.code)
    req = urllib.request.Request(base + "/api/health", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=5) as r:
        ck(r.status in (200, 204), "OPTIONS 未返回 204")
        ck(r.headers.get("Access-Control-Allow-Origin") == "*", "缺少 CORS 头")


if __name__ == "__main__":
    sys.exit(main())
