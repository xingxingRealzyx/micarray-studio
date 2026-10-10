"""声源定位模块验收脚本（纯标准库 + numpy，不用 pytest）。

跑法：
    cd micarray-studio
    PYTHONPATH=. <python> tests/test_localization.py

覆盖：
  1. 单声源自由场（20 dB SNR）：SRP-PHAT / MUSIC 角度误差 < 3°
  2. 32 麦 × 362 方向 × 1 s 的 SRP-PHAT 端到端耗时（向量化验收）
  3. 近场 0.6 m：tdoa_estimate 位置误差 < 5 cm、距离相对误差 < 20%
  4. 多声源：MUSIC(n_sources=2) 能分辨两个峰，各自误差 < 5°
  5. 混响（RoomSpec 中等混响）：SRP-PHAT 误差 < 8°
  6. map 无 NaN/Inf、峰值在真值附近
  7. 数值一致性：近场导向矢量 vs steering.steering_vector、向量化时延模型
     vs tdoa.delay_residual、localize 统一 schema、无随机性（同输入同输出）

设计上刻意避开的两个"物理退化点"（不是实现问题，见 engine/localization.py
的 _resolve_hemisphere 注释）：
  * 平面阵的前后模糊：镜像方向 (az,-el) 的观测与真值**逐位相同**。
    因此正 el 的源用 hemisphere='front'、负 el 的源用 'back'。
  * el≈0（声源在阵面上）：el 由 |u_xy| = cos(el) 反解，误差被 1/sin(el) 放大，
    本质上不可观测。测试用 |el| >= 15°。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import acoustics as ac            # noqa: E402
from engine import geometry as geo            # noqa: E402
from engine import localization as lc         # noqa: E402
from engine import signals as sg              # noqa: E402
from engine import sim                        # noqa: E402
from engine import steering as st             # noqa: E402
from engine import tdoa as td                 # noqa: E402

FS = 16000
RESULTS: list[tuple[str, bool, str]] = []


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def ang_err(a, b) -> float:
    """(az,el) 之间的球面角距（度）。az 用 cos(el) 折算，避免高 el 处虚高。"""
    u, v = ac.unit_dir(a[0], a[1]), ac.unit_dir(b[0], b[1])
    return float(np.rad2deg(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def min_ang_err(est_dirs, truth) -> float:
    return min(ang_err(e, truth) for e in est_dirs)


def rms_matched_amplitude(kind: str, params: dict, fs: int, n: int,
                          target_rms: float = 1.0) -> float:
    """让两个不同信号在阵列质心处有相同 RMS。

    sim.simulate() 内部按 **峰值** 归一化再乘 amplitude，所以谐波类信号
    （crest factor 高）会明显比语音弱；多源测试必须先把功率拉平，
    否则"分辨不出第二个源"其实是第二个源太小，而不是算法问题。
    """
    s = sg.generate(kind, fs, n, params)
    s = s / (np.abs(s).max() + 1e-12)
    return float(target_rms / (np.std(s) + 1e-12))


def free_field(geom, az, el, dist, fs=FS, duration=1.0, snr=20.0,
               kind="speech", params=None, seed=5):
    cfg = sim.SimConfig(fs=fs, duration=duration, center=(0.0, 0.0, 0.0),
                        boresight=(0.0, 90.0))
    src = sim.SourceSpec(signal=kind, direction=(az, el), distance=dist,
                         params=params or {"seed": 3})
    noise = None if snr is None else sim.NoiseSpec(snr, "white", False, seed)
    return sim.simulate(geom, [src], cfg, room=None, noise=noise)


def hemi_of(el) -> str:
    return "front" if el >= 0 else "back"


# --------------------------------------------------------------------------
# 1. 单声源：SRP-PHAT / MUSIC
# --------------------------------------------------------------------------
def test_single_source():
    print("\n== 1. 单声源自由场 (20 dB SNR) ==")
    geom = geo.build_preset("vogel-32")
    mics = geom.mics
    grid = geo.direction_grid()                      # 72 x 37 = 2664 -> (G,3)
    # (az, el, distance)：距离 1.0 m 已在 Fraunhofer 距离以内（vogel-32 在
    # 2 kHz 处 2D²/λ ≈ 1.7 m），刻意留一个近场距离来暴露远场模型偏差
    cases = [(37.0, 22.0, 2.0), (-120.0, 35.0, 2.0), (160.0, -15.0, 2.0),
             (90.0, 60.0, 2.0), (-45.0, -25.0, 2.0), (0.0, 45.0, 1.0),
             (135.0, 50.0, 1.0)]
    errs_srp, errs_mus = [], []
    for az_t, el_t, dist in cases:
        res = free_field(geom, az_t, el_t, dist)
        hem = hemi_of(el_t)
        o_srp = lc.srp_phat(res.X, res.fs, mics, grid, hemisphere=hem)
        o_mus = lc.music_spectrum(res.X, res.fs, mics, n_sources=1,
                                  grid_dirs=grid, hemisphere=hem)
        e1 = ang_err(o_srp["est_dirs"][0], (az_t, el_t))
        e2 = ang_err(o_mus["est_dirs"][0], (az_t, el_t))
        errs_srp.append(e1)
        errs_mus.append(e2)
        # map 健康检查
        for tag, o in (("srp", o_srp), ("music", o_mus)):
            assert np.all(np.isfinite(o["map"])), f"{tag} map 含 NaN/Inf"
            assert o["map"].shape == (grid[1][0] * grid[1][1],)
            assert abs(float(o["map"].max()) - 1.0) < 1e-9
        print("    truth=(%7.1f,%6.1f,%.1fm) srp=(%8.3f,%7.3f) err=%.3f | "
              "music=(%8.3f,%7.3f) err=%.3f"
              % (az_t, el_t, dist, o_srp["est_dirs"][0][0],
                 o_srp["est_dirs"][0][1], e1, o_mus["est_dirs"][0][0],
                 o_mus["est_dirs"][0][1], e2))
    check("单声源 SRP-PHAT 误差 < 3°", max(errs_srp) < 3.0,
          "max=%.3f° mean=%.3f°" % (max(errs_srp), np.mean(errs_srp)))
    check("单声源 MUSIC 误差 < 3°", max(errs_mus) < 3.0,
          "max=%.3f° mean=%.3f°" % (max(errs_mus), np.mean(errs_mus)))


# --------------------------------------------------------------------------
# 2. 向量化 / 性能：32 麦 × 362 方向 × 1 s
# --------------------------------------------------------------------------
def test_performance_and_schema():
    print("\n== 2. 32 麦 × 362 方向 × 1 s 的 SRP-PHAT ==")
    geom = geo.build_preset("vogel-32")
    mics = geom.mics
    assert mics.shape[0] == 32
    # 正好 362 个方向的网格：181 az × 2 el（el = 0° 与 40°，都在前半球）
    gdirs, gshape, gaz, gel = geo.direction_grid(az_bins=181, el_bins=2,
                                                 el_range=(0.0, 40.0))
    assert gdirs.shape == (362, 3) and gshape == (2, 181)

    res = free_field(geom, 40.0, 20.0, 2.0, duration=1.0)
    t0 = time.time()
    o = lc.srp_phat(res.X, res.fs, mics, (gdirs, gshape, gaz, gel))
    dt = time.time() - t0
    print("    G=%d, M=%d, N=%d -> SRP-PHAT %.3f s"
          % (gdirs.shape[0], mics.shape[0], res.X.shape[1], dt))
    for k in ("map", "shape", "az_axis", "el_axis", "est_dirs", "est_idx",
              "peak_vals"):
        assert k in o, k
    check("362 方向网格耗时 < 5 s", dt < 5.0, "%.3f s" % dt)
    check("362 网格定位误差 < 3°", ang_err(o["est_dirs"][0], (40.0, 20.0)) < 3.0,
          "err=%.3f°" % ang_err(o["est_dirs"][0], (40.0, 20.0)))
    check("map 无 NaN/Inf 且归一化", bool(np.all(np.isfinite(o["map"])))
          and abs(float(o["map"].max()) - 1.0) < 1e-9, "max=%.6f" % o["map"].max())
    check("map 是扁平数组 + shape（362 点）",
          o["map"].ndim == 1 and tuple(o["shape"]) == (2, 181)
          and o["map"].size == int(np.prod(o["shape"])),
          "len(map)=%d shape=%s" % (o["map"].size, o["shape"]))

    # 默认网格（2664 点 = 37×72）也报耗时 + 校验 shape 语义
    t0 = time.time()
    od = lc.srp_phat(res.X, res.fs, mics, geo.direction_grid())
    print("    （参考）默认 2664 方向网格耗时 %.3f s" % (time.time() - t0))
    check("默认网格 map 与 shape/轴自洽",
          tuple(od["shape"]) == (37, 72) and od["map"].size == 37 * 72
          and od["az_axis"].size == 72 and od["el_axis"].size == 37,
          "shape=%s az=%d el=%d" % (od["shape"], od["az_axis"].size,
                                    od["el_axis"].size))


# --------------------------------------------------------------------------
# 3. 近场
# --------------------------------------------------------------------------
def test_near_field():
    print("\n== 3. 近场 0.6 m：TDOA 位置 / 距离 ==")
    geom = geo.build_preset("uca-16")
    mics = geom.mics
    # |el| 不宜接近 90°：平面阵在 boresight 附近距离不可观测（见文件头注释）
    cases = [(37.0, 22.0), (-120.0, 30.0), (15.0, -25.0), (90.0, 35.0)]
    pos_err, rel_err, nfd_err = [], [], []
    for az_t, el_t in cases:
        res = free_field(geom, az_t, el_t, 0.6)
        truth = np.asarray(res.truth["sources"][0]["position_rel"], float)
        o = lc.tdoa_estimate(res.X, res.fs, mics, hemisphere=hemi_of(el_t))
        pe = float(np.linalg.norm(np.asarray(o["xyz"]) - truth))
        re = abs(o["distance"] - 0.6) / 0.6
        nd = lc.near_field_distance(res.X, res.fs, mics, az_t, el_t)
        ne = abs(nd["distance"] - 0.6) / 0.6
        pos_err.append(pe)
        rel_err.append(re)
        nfd_err.append(ne)
        assert "curve" in nd and len(nd["curve"]["distance"]) == len(nd["curve"]["cost"])
        assert np.all(np.isfinite(nd["curve"]["cost"]))
        assert np.all(np.isfinite(o["taus"]))
        print("    truth=(%7.1f,%6.1f,0.60)  xyz=(%6.3f,%6.3f,%6.3f) |Δ|=%.4f m"
              "  r=%.3f (%.1f%%)  nfd=%.3f (%.1f%%)  iters=%d/%d"
              % (az_t, el_t, o["xyz"][0], o["xyz"][1], o["xyz"][2], pe,
                 o["distance"], 100 * re, nd["distance"], 100 * ne,
                 o["iterations_far"], o["iterations_near"]))
    check("近场位置误差 < 5 cm", max(pos_err) < 0.05,
          "max=%.4f m (%s)" % (max(pos_err), np.round(pos_err, 4).tolist()))
    check("tdoa 距离相对误差 < 20%", max(rel_err) < 0.20,
          "max=%.1f%%" % (100 * max(rel_err)))
    check("near_field_distance 相对误差 < 20%", max(nfd_err) < 0.20,
          "max=%.1f%%" % (100 * max(nfd_err)))
    # 远场（2 m）也不该崩：方向 < 2°，距离数量级正确
    res = free_field(geom, 37.0, 22.0, 2.0)
    o = lc.tdoa_estimate(res.X, res.fs, mics, hemisphere="front")
    check("2 m 处 TDOA 方向误差 < 2°",
          ang_err([o["az"], o["el"]], (37.0, 22.0)) < 2.0,
          "err=%.3f° r=%.3f m" % (ang_err([o["az"], o["el"]], (37.0, 22.0)),
                                  o["distance"]))


def test_near_field_observability():
    """把"平面阵在 boresight 附近测不出距离"这一物理限制固化成可回归的事实。

    声源在阵列轴线附近时，各阵元到声源的距离几乎相同（UCA 上甚至**完全**相同），
    波前曲率信号 ~a²/(2r) 在阵元间几乎不变化 -> 距离不可观测。
    这里用 near_field_distance 的代价曲线"相对起伏"量化它：
    起伏大 = 距离可观测，起伏小 = 不可观测（此时任何距离都同样拟合得好）。
    """
    print("\n== 3b. 近场距离可观测性（物理限制的表征测试） ==")
    geom = geo.build_preset("uca-16")
    mics = geom.mics
    rows = []
    for el_t in (15.0, 30.0, 80.0):
        res = free_field(geom, 40.0, el_t, 0.6)
        nd = lc.near_field_distance(res.X, res.fs, mics, 40.0, el_t,
                                    r_min=0.2, r_max=3.0)
        c = np.asarray(nd["curve"]["cost"], float)
        contrast = float((c.max() - c.min()) / max(c.min(), 1e-30))
        rel = abs(nd["distance"] - 0.6) / 0.6
        rows.append((el_t, contrast, rel))
        print("    el=%4.1f°  代价曲线相对起伏=%.3f  距离误差=%.1f%%"
              % (el_t, contrast, 100 * rel))
    big = [r for r in rows if r[0] <= 30.0]
    near_bore = [r for r in rows if r[0] >= 80.0]
    check("el<=30° 时距离可观测（曲线起伏 > 0.3 且误差 < 20%）",
          all(r[1] > 0.3 and r[2] < 0.20 for r in big),
          "contrast=%s" % np.round([r[1] for r in big], 3).tolist())
    c_min = min(r[1] for r in big)
    check("el=80°(近 boresight) 距离**不可观测**（起伏比可观测情形小 3 倍以上 + 误差 >20%）",
          all(r[1] < 0.3 * c_min and r[2] > 0.20 for r in near_bore),
          "contrast=%.3f vs %.3f（可观测），误差=%.0f%%；"
          "这是平面阵的固有性质，不是实现缺陷"
          % (near_bore[0][1], c_min, 100 * near_bore[0][2]))


# --------------------------------------------------------------------------
# 4. 多声源
# --------------------------------------------------------------------------
def test_two_sources():
    print("\n== 4. 双声源：MUSIC(n_sources=2) ==")
    geom = geo.build_preset("vogel-32")
    mics = geom.mics
    grid = geo.direction_grid()
    fs, dur = FS, 1.5
    n = int(fs * dur)
    p1, p2 = {"seed": 3}, {"seed": 11, "f0": 150.0}
    a1 = rms_matched_amplitude("speech", p1, fs, n)
    a2 = rms_matched_amplitude("speech", p2, fs, n)
    cases = [((0.0, 30.0), (60.0, 20.0)), ((-40.0, 25.0), (70.0, 20.0)),
             ((120.0, 40.0), (-100.0, 15.0)), ((0.0, 50.0), (35.0, 30.0))]
    all_err = []
    for (az1, el1), (az2, el2) in cases:
        cfg = sim.SimConfig(fs=fs, duration=dur, center=(0.0, 0.0, 0.0),
                            boresight=(0.0, 90.0))
        srcs = [sim.SourceSpec(signal="speech", direction=(az1, el1),
                               distance=2.0, params=p1, amplitude=a1),
                sim.SourceSpec(signal="speech", direction=(az2, el2),
                               distance=2.0, params=p2, amplitude=a2)]
        res = sim.simulate(geom, srcs, cfg, room=None,
                           noise=sim.NoiseSpec(20.0, "white", False, 5))
        # hemisphere='auto'：平面阵默认只在前半球取峰。
        # 平面阵的空间谱**严格镜像对称**，用 'both' 时 n_sources=2 必然返回
        # "同一个源 + 它的镜像"，这不是 bug 而是观测数据的固有模糊。
        o = lc.music_spectrum(res.X, res.fs, mics, n_sources=2,
                              grid_dirs=grid, hemisphere="auto")
        e1 = min_ang_err(o["est_dirs"], (az1, el1))
        e2 = min_ang_err(o["est_dirs"], (az2, el2))
        sep = ang_err((az1, el1), (az2, el2))
        # 两个峰必须真的是两个不同方向（不能都指向同一个峰）
        d12 = ang_err(o["est_dirs"][0], o["est_dirs"][1]) if len(o["est_dirs"]) > 1 else 0.0
        all_err += [e1, e2]
        assert np.all(np.isfinite(o["map"]))
        print("    truth=[(%6.1f,%5.1f),(%6.1f,%5.1f)] 分离=%4.1f°  "
              "est=%s  err=(%.2f°, %.2f°)  峰间距=%.1f°  峰值=%s"
              % (az1, el1, az2, el2, sep, np.round(o["est_dirs"], 2).tolist(),
                 e1, e2, d12, np.round(o["peak_vals"], 3).tolist()))
        assert len(o["est_dirs"]) == 2, "MUSIC 未给出 2 个峰"
    check("MUSIC 双源分辨（各 < 5°）", max(all_err) < 5.0,
          "max=%.3f° mean=%.3f°" % (max(all_err), np.mean(all_err)))


# --------------------------------------------------------------------------
# 5. 混响
# --------------------------------------------------------------------------
def test_reverb():
    print("\n== 5. 混响（镜像源法 RIR）：SRP-PHAT ==")
    geom = geo.build_preset("vogel-32")
    mics = geom.mics
    grid = geo.direction_grid()
    room = sim.RoomSpec(bounds=(-3.0, 3.0, -2.0, 2.0, 0.0, 2.7),
                        absorption=(0.25, 0.25, 0.25, 0.25, 0.2, 0.25),
                        max_order=3)
    rt60 = sim.estimate_rt60(room)
    print("    room RT60(Sabine)≈%.2f s, max_order=%d" % (rt60, room.max_order))
    errs = []
    for az_t, el_t in [(30.0, 40.0), (-80.0, 25.0), (150.0, 35.0)]:
        cfg = sim.SimConfig(fs=FS, duration=1.0, center=(0.0, 0.0, 1.5),
                            boresight=(0.0, -90.0))
        src = sim.SourceSpec(signal="speech", direction=(az_t, el_t),
                             distance=1.0, params={"seed": 3})
        res = sim.simulate(geom, [src], cfg, room=room,
                           noise=sim.NoiseSpec(20.0, "white", False, 5))
        o = lc.srp_phat(res.X, res.fs, mics, grid, hemisphere="front")
        e = ang_err(o["est_dirs"][0], (az_t, el_t))
        o_m = lc.music_spectrum(res.X, res.fs, mics, n_sources=1,
                                grid_dirs=grid, hemisphere="front")
        em = ang_err(o_m["est_dirs"][0], (az_t, el_t))
        errs.append(e)
        print("    truth=(%6.1f,%5.1f)  srp=(%8.3f,%7.3f) err=%.3f | music err=%.3f"
              % (az_t, el_t, o["est_dirs"][0][0], o["est_dirs"][0][1], e, em))
        assert np.all(np.isfinite(o["map"]))
    check("中等混响下 SRP-PHAT 误差 < 8°", max(errs) < 8.0,
          "max=%.3f° mean=%.3f°" % (max(errs), np.mean(errs)))


# --------------------------------------------------------------------------
# 6. map 健全性 + 峰值在真值附近
# --------------------------------------------------------------------------
def test_map_sanity():
    print("\n== 6. map 健全性 / 峰值位置 ==")
    geom = geo.build_preset("uca-16")
    mics = geom.mics
    grid = geo.direction_grid()
    dirs, shape, az_ax, el_ax = grid
    res = free_field(geom, 55.0, 35.0, 2.0)
    for tag, o in (("srp_phat", lc.srp_phat(res.X, res.fs, mics, grid)),
                   ("music", lc.music_spectrum(res.X, res.fs, mics,
                                                n_sources=1, grid_dirs=grid))):
        m = o["map"]
        finite = bool(np.all(np.isfinite(m)))
        inrange = float(m.min()) >= 0.0 and float(m.max()) <= 1.0 + 1e-9
        # 峰值格点必须落在真值附近
        k = o["est_idx"][0]
        e_pk = ang_err(ac.angles_from_dir(dirs[k]), (55.0, 35.0))
        check("%s: map 有限且在 [0,1]" % tag, finite and inrange,
              "min=%.3g max=%.6f" % (m.min(), m.max()))
        check("%s: 峰值格点靠近真值 (<6°)" % tag, e_pk < 6.0, "%.3f°" % e_pk)
        check("%s: est_idx 与 shape 自洽" % tag,
              int(np.prod(o["shape"])) == m.size and 0 <= k < m.size,
              "shape=%s idx=%d size=%d" % (o["shape"], k, m.size))
        # map 在真值方向附近确实很高
        u = ac.unit_dir(55.0, 35.0)
        near = m[dirs @ u > np.cos(np.deg2rad(10.0))]
        check("%s: 真值 10° 邻域内 map > 0.8" % tag, near.max() > 0.8,
              "%.3f" % near.max())


# --------------------------------------------------------------------------
# 7. 数值一致性 / 确定性 / 统一入口
# --------------------------------------------------------------------------
def test_consistency():
    print("\n== 7. 数值一致性与确定性 ==")
    geom = geo.build_preset("vogel-32")
    mics = geom.mics

    # 7a. 近场导向矢量：向量化实现 vs 契约函数 steering.steering_vector
    rng = np.random.default_rng(0)          # 只用于生成测试点，不进入算法
    pts = rng.uniform(-1.5, 1.5, (12, 3))
    fs_sel = np.array([500.0, 1500.0, 3000.0])
    A = lc._near_field_manifold(mics, fs_sel, pts, 343.0)
    err = 0.0
    for gi, p in enumerate(pts):
        ref = st.steering_vector(mics, fs_sel, p, 343.0, near_field=True)
        err = max(err, float(np.abs(A[:, gi, :] - ref).max()))
    check("近场导向矢量与 steering.steering_vector 一致 (<1e-12)",
          err < 1e-12, "max|Δ|=%.3e" % err)

    # 7b. 向量化时延模型 vs tdoa.delay_residual
    res = free_field(geom, 33.0, 25.0, 1.2)
    taus, pairs = td.gcc_phat_matrix(res.X, res.fs, max_delay=0.002)
    u = ac.unit_dir(33.0, 25.0)
    r_vec = td.delay_residual(taus, pairs, mics, u, 343.0, near_field=False)
    model = lc._pair_delays_dirs(mics, u[None, :], pairs, 343.0)[0]
    r_ref = taus - model
    e1 = float(np.abs(r_vec - r_ref).max())
    pts = np.array([[0.3, 0.4, 0.5], [-0.6, 0.1, -0.2]])
    r_vec2 = td.delay_residual(taus, pairs, mics, pts[1], 343.0, near_field=True)
    model2 = lc._pair_delays_points(mics, pts[1:2], pairs, 343.0)[0]
    e2 = float(np.abs(r_vec2 - (taus - model2)).max())
    check("向量化远场时延模型 == delay_residual", e1 < 1e-15, "max|Δ|=%.3e" % e1)
    check("向量化近场时延模型 == delay_residual", e2 < 1e-15, "max|Δ|=%.3e" % e2)

    # 7c. 确定性：同输入两次调用逐位相同（C++ 逐位比对的前提）
    a = lc.srp_phat(res.X, res.fs, mics, geo.direction_grid())
    b = lc.srp_phat(res.X, res.fs, mics, geo.direction_grid())
    same = np.array_equal(a["map"], b["map"]) and a["est_dirs"] == b["est_dirs"]
    c = lc.music_spectrum(res.X, res.fs, mics, n_sources=1,
                          grid_dirs=geo.direction_grid())
    d = lc.music_spectrum(res.X, res.fs, mics, n_sources=1,
                          grid_dirs=geo.direction_grid())
    same = same and np.array_equal(c["map"], d["map"])
    t1 = lc.tdoa_estimate(res.X, res.fs, mics)
    t2 = lc.tdoa_estimate(res.X, res.fs, mics)
    same = same and np.array_equal(t1["xyz"], t2["xyz"]) and t1["az"] == t2["az"]
    check("同输入两次调用逐位一致（无随机性）", same, "")

    # 7d. localize 统一入口 schema
    truth = res.truth
    out = lc.localize(res.X, res.fs, mics, method="srp_phat", truth=truth,
                      n_sources=1)
    keys = ("map", "shape", "az_axis", "el_axis", "est_dirs", "est_idx",
            "peak_vals", "method", "truth", "n_sources", "est_xyz")
    miss = [k for k in keys if k not in out]
    check("localize 返回统一字段", not miss, "缺: %s" % miss if miss else
          "method=%s" % out["method"])
    check("localize 回填 truth", out["truth"] is truth, "")
    o_t = lc.localize(res.X, res.fs, mics, method="tdoa", truth=truth)
    o_m = lc.localize(res.X, res.fs, mics, method="music", n_sources=1,
                      truth=truth)
    ok = (o_t["est_xyz"] is not None and np.all(np.isfinite(o_t["est_xyz"]))
          and o_m["map"] is not None and np.all(np.isfinite(o_m["map"]))
          and not [k for k in keys if k not in o_t]
          and not [k for k in keys if k not in o_m])
    check("localize(tdoa/music) 字段齐全且有限", bool(ok),
          "tdoa xyz=%s" % np.round(np.asarray(o_t["est_xyz"]), 3).tolist())

    # 7e. hybrid（SRP 粗定位 + TDOA 精化）与 tdoa 的 est_xyz 一致性
    o_h = lc.localize(res.X, res.fs, mics, method="hybrid", truth=truth)
    check("localize(hybrid) 可用且与 tdoa 方向一致 (<0.5°)",
          ang_err(o_h["est_dirs"][0], o_t["est_dirs"][0]) < 0.5
          and o_h["map"] is not None,
          "hybrid=%s tdoa=%s"
          % (np.round(o_h["est_dirs"][0], 3).tolist(),
             np.round(o_t["est_dirs"][0], 3).tolist()))

    # 7f. 频点选择约定
    bins, fr = lc.select_freq_bins(FS, 512, mics, 343.0)
    top = ac.alias_limit(geo.min_spacing(mics), 343.0)
    check("自动频带上界受无混叠上限约束",
          fr[-1] <= 0.45 * FS + 1e-9 and fr[-1] <= top + 1e-9,
          "f=[%.1f, %.1f] Hz, n=%d, alias=%.1f Hz" % (fr[0], fr[-1], bins.size, top))
    b2, f2 = lc.select_freq_bins(FS, 512, mics, 343.0, freq_band=(1000.0, 2000.0))
    check("显式 freq_band 生效", 1000.0 - 1e-9 <= f2[0] and f2[-1] <= 2000.0 + 1e-9,
          "[%.1f, %.1f] Hz n=%d" % (f2[0], f2[-1], f2.size))


# --------------------------------------------------------------------------
# 8. 接口健壮性（坏输入必须报错，而不是静默返回 NaN）
# --------------------------------------------------------------------------
def _raises_valueerror(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def test_robustness():
    print("\n== 8. 接口健壮性 / 非默认参数路径 ==")
    geom = geo.build_preset("uca-16")
    mics = geom.mics
    grid = geo.direction_grid(az_bins=36, el_bins=19)
    dirs, shape, az_ax, el_ax = grid
    res = free_field(geom, 30.0, 25.0, 1.0, duration=0.4)

    check("通道数与阵元数不一致时报错",
          _raises_valueerror(lambda: lc.srp_phat(res.X[:4], res.fs, mics,
                                                 grid))
          and _raises_valueerror(lambda: lc.tdoa_estimate(res.X[0], res.fs,
                                                          mics))
          and _raises_valueerror(lambda: lc.near_field_distance(
              res.X[0], res.fs, mics, 30.0, 25.0)), "")
    check("n_sources >= M 时 MUSIC 报错",
          _raises_valueerror(lambda: lc.music_spectrum(
              res.X, res.fs, mics, n_sources=16, grid_dirs=grid)), "M=16")
    check("未知 method 报错",
          _raises_valueerror(lambda: lc.localize(res.X, res.fs, mics, "nope")),
          "")

    zero = np.zeros((16, 4000))
    zs = lc.srp_phat(zero, res.fs, mics, grid)
    zm = lc.music_spectrum(zero, res.fs, mics, n_sources=1, grid_dirs=grid)
    zt = lc.tdoa_estimate(zero, res.fs, mics)
    check("全零信号不产生 NaN/Inf",
          bool(np.all(np.isfinite(zs["map"])))
          and bool(np.all(np.isfinite(zm["map"])))
          and bool(np.all(np.isfinite(np.asarray(zt["xyz"])))), "")

    m3 = np.array([[0.10, 0.0, 0.02], [-0.10, 0.0, -0.02],
                   [0.0, 0.10, 0.05], [0.0, -0.10, -0.05]])
    o3 = lc.srp_phat(np.random.default_rng(0).standard_normal((4, 4000)),
                     res.fs, m3, az_bins=36, el_bins=19)
    check("非平面阵不做前后半球约定（hemisphere=both）",
          o3["planar_array"] is False and o3["hemisphere"] == "both"
          and o3["front_back_ambiguous"] is False,
          "planar=%s hemi=%s" % (o3["planar_array"], o3["hemisphere"]))

    # 非默认参数路径都要能跑通且结果有限
    variants = {
        "grid=dict": lambda: lc.srp_phat(res.X, res.fs, mics,
                                         {"dirs": dirs, "shape": shape,
                                          "az_axis": az_ax, "el_axis": el_ax}),
        "grid=裸(G,3)": lambda: lc.srp_phat(res.X, res.fs, mics, dirs),
        "freqs=离散频点": lambda: lc.srp_phat(res.X, res.fs, mics, grid,
                                              freqs=[500., 800., 1200., 2000.]),
        "freqs=(lo,hi) 频带": lambda: lc.srp_phat(res.X, res.fs, mics, grid,
                                                  freqs=(400., 3000.)),
        "weights=none": lambda: lc.srp_phat(res.X, res.fs, mics, grid,
                                            weights="none"),
        "beta=0.6": lambda: lc.srp_phat(res.X, res.fs, mics, grid, beta=0.6),
        "src_dist 近场导向": lambda: lc.srp_phat(res.X, res.fs, mics, grid,
                                                 src_dist=1.0),
        "n_fft=1024": lambda: lc.music_spectrum(res.X, res.fs, mics,
                                                n_sources=1, grid_dirs=grid,
                                                n_fft=1024),
        "loading=0.01": lambda: lc.music_spectrum(res.X, res.fs, mics,
                                                  n_sources=1, grid_dirs=grid,
                                                  loading=0.01),
        "pair_mode=star": lambda: lc.tdoa_estimate(res.X, res.fs, mics,
                                                   pair_mode="star"),
        "interp=2": lambda: lc.tdoa_estimate(res.X, res.fs, mics, interp=2),
        "clamp_to_aperture=False": lambda: lc.tdoa_estimate(
            res.X, res.fs, mics, clamp_to_aperture=False),
        "nfd refine=False": lambda: lc.near_field_distance(
            res.X, res.fs, mics, 30.0, 25.0, n_scan=32, refine=False),
    }
    bad = []
    for name, fn in variants.items():
        try:
            o = fn()
            arrs = [v for v in o.values() if isinstance(v, np.ndarray)]
            if not all(np.all(np.isfinite(a)) for a in arrs):
                bad.append(name + "(NaN)")
        except Exception as exc:
            bad.append("%s(%r)" % (name, exc))
    check("非默认参数路径全部可跑且有限", not bad, "问题: %s" % bad if bad else
          "%d 个变体" % len(variants))


def main():
    t0 = time.time()
    print("=" * 78)
    print("MicArray Studio —— 声源定位模块验收 (numpy %s)" % np.__version__)
    print("=" * 78)
    for fn in (test_single_source, test_performance_and_schema,
               test_near_field, test_near_field_observability,
               test_two_sources, test_reverb,
               test_map_sanity, test_consistency, test_robustness):
        try:
            fn()
        except Exception as exc:            # 单个用例炸掉不应吞掉其它用例
            import traceback
            traceback.print_exc()
            check("%s 未抛异常" % fn.__name__, False, repr(exc))

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print("汇总：%d 项检查，%d 通过，%d 失败，用时 %.1f s"
          % (len(RESULTS), len(RESULTS) - n_fail, n_fail, time.time() - t0))
    if n_fail:
        for name, ok, detail in RESULTS:
            if not ok:
                print("   FAIL: %s  %s" % (name, detail))
    print("结果：%s" % ("PASS" if n_fail == 0 else "FAIL"))
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
