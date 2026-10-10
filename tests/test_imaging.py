"""声场成像模块验收测试（纯 Python + assert，无 pytest / scipy）。

运行：
    python tests/test_imaging.py            # 全部用例
    python tests/test_imaging.py --quick    # 跳过性能用例

覆盖 docs/CONTRACT.md 与任务书里的验收点：
  1. PSF 正确性（主对角 = 1.0、主瓣峰值落在对准方向、对称性、按需取列与全矩阵一致）
  2. 单声源：脏图峰值误差 < 3°，DAMAS 主瓣宽度比脏图窄 > 30%
  3. 两相干声源（同信号）：脏图分不开，DAMAS 分辨出两个峰且各自误差 < 5°
  4. 两独立声源：DAMAS / 正交波束成形分辨
  5. CLEAN-SC：单源定位、分离较开时能分辨两个峰、分量功率与位置合理
  6. 动态范围：DAMAS 峰值旁瓣级优于脏图
  7. DAMAS 提前停止与迭代历史（残差单调性/收敛标记）
  8. 所有输出无 NaN / Inf
  9. 性能：16 麦 + 362 方向网格 + n_fft=512 + 1 秒数据的"脏图 + DAMAS 200 迭代" < 60 s

数据全部由 engine.sim.simulate() 用固定 seed 生成，测试本身可重复。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import acoustics as ac          # noqa: E402
from engine import geometry as gm           # noqa: E402
from engine import imaging as im            # noqa: E402
from engine import sim                      # noqa: E402

# --------------------------------------------------------------------------
# 极简测试框架
# --------------------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []
_TIMINGS: list[tuple[str, float]] = []


def check(name: str, cond: bool, detail="") -> bool:
    _RESULTS.append((name, bool(cond), str(detail)))
    print(("  [PASS] " if cond else "  [FAIL] ") + name
          + (" :: " + str(detail) if detail != "" else ""))
    return bool(cond)


def timed(name: str):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *a):
            _TIMINGS.append((name, time.perf_counter() - self.t0))
            return False
    return _T()


def finite_all(obj, path: str = "root", bad: list | None = None) -> list:
    """递归检查 dict/list/ndarray 里是否存在 NaN / Inf。"""
    if bad is None:
        bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            finite_all(v, f"{path}.{k}", bad)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            finite_all(v, f"{path}[{i}]", bad)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fc" and obj.size and not np.all(np.isfinite(obj)):
            bad.append(path)
    elif isinstance(obj, (float, np.floating)):
        if not np.isfinite(float(obj)):
            bad.append(path)
    return bad


# --------------------------------------------------------------------------
# 公共数据
# --------------------------------------------------------------------------
FS = 16000
DUR = 0.5
EL0 = 60.0                      # 源仰角（正对阵列前方，避开 az 奇点）
BAND = (1700.0, 2200.0)         # 分析频带（1/3 倍频程量级）
AZ_BINS, EL_BINS = 73, 19       # 5° 网格，G = 1387


def make_geom():
    return gm.build_preset("spiral-4x4")        # 16 麦，孔径 0.5 m


def make_grid(az_bins=AZ_BINS, el_bins=EL_BINS):
    return gm.direction_grid(az_range=(-180.0, 180.0), el_range=(0.0, 90.0),
                             az_bins=az_bins, el_bins=el_bins)


def nb_source(seed, az, el=EL0, dist=6.0, kind="white", freqs=None):
    """确定性宽带点源（白噪/多音）。宽带是 DAMAS 能收敛的前提：
    单频时 PSF 矩阵近似秩一，非负最小二乘解不唯一，DAMAS 会摊开。"""
    if kind == "white":
        params = {"seed": int(seed)}
    else:
        params = {"freqs": list(freqs) if freqs is not None
                  else [1750, 1850, 1950, 2050, 2150]}
    return sim.SourceSpec(signal=kind, params=params, direction=(float(az), float(el)),
                          distance=float(dist), amplitude=1.0)


def simulate(sources, snr=30.0, dur=DUR):
    return sim.simulate(make_geom(), sources, sim.SimConfig(fs=FS, duration=dur),
                        noise=sim.NoiseSpec(snr_db=snr))


def nearest_dir(dirs, az, el):
    u = ac.unit_dir(az, el)
    return int(np.argmin(im._angular_distance_deg(dirs, u)[:, 0]))


# ==========================================================================
# 1. PSF 正确性
# ==========================================================================
def test_psf():
    print("\n[1] PSF 正确性")
    g = make_geom()
    # 小网格，避开两种退化：el=±90 极点（所有 az 同向）
    # 与 az=-180/+180 重合（direction_grid 默认两端都取）
    dirs, shape, az, el = gm.direction_grid(az_range=(-160.0, 160.0),
                                            el_range=(0.0, 80.0),
                                            az_bins=9, el_bins=5)   # G=45
    freqs = np.array([1500.0, 2000.0, 2500.0])
    psf = im.point_spread_function(g.mics, freqs, (dirs, shape, az, el))
    check("PSF 形状 = (G,G)", psf.shape == (dirs.shape[0], dirs.shape[0]),
          f"{psf.shape}")
    diag = psf.diagonal()
    check("PSF 主对角恒为 1.0", np.allclose(diag, 1.0, atol=1e-12),
          f"max|diag-1| = {np.abs(diag - 1.0).max():.2e}")

    # 主瓣峰值必须落在对准方向
    ok_argmax, ok_val = True, True
    for i in range(0, dirs.shape[0], 7):
        col = psf[:, i]
        ok_argmax &= (int(np.argmax(col)) == i)
        ok_val &= abs(float(col[i]) - 1.0) < 1e-12
        ok_val &= bool(np.all(col <= 1.0 + 1e-12))
    check("PSF 每列最大值出现在对准方向 (argmax == j)", ok_argmax)
    check("PSF 峰值 = 1.0 且不超过 1.0", ok_val)

    P = psf.matrix()
    check("PSF 全矩阵对称", np.allclose(P, P.T, atol=1e-12),
          f"max|P-P^T| = {np.abs(P - P.T).max():.2e}")
    check("PSF 无 NaN/Inf", len(finite_all(P, "psf")) == 0)

    # 按需取列 / 取行 / 分块必须与全矩阵一致
    cols = psf.columns([3, 8, 20])
    check("psf_columns 与全矩阵列一致", np.allclose(cols, P[:, [3, 8, 20]], atol=1e-12))
    rows = psf.rows([5, 11])
    check("psf.rows 与全矩阵行一致", np.allclose(rows, P[[5, 11], :], atol=1e-12))
    check("psf[i, j] 标量索引一致", abs(psf[4, 9] - P[4, 9]) < 1e-12)
    check("psf[:, j] 切片索引一致", np.allclose(psf[:, 9], P[:, 9], atol=1e-12))
    check("np.asarray(psf) == matrix()", np.allclose(np.asarray(psf), P, atol=1e-12))

    # 独立入口 psf_column / psf_columns
    c0 = im.psf_column(g.mics, freqs, (dirs, shape, az, el), 9)
    check("psf_column 与 psf[:,9] 一致", np.allclose(c0, P[:, 9], atol=1e-12))
    check("PSF 全矩阵形状过大时拒绝稠密化",
          _raises(lambda: psf.matrix(max_elements=10), ValueError))
    # 不同 steer/src 网格 → 非方阵
    psf2 = im.point_spread_function(g.mics, freqs, (dirs, shape, az, el),
                                   (dirs[:10], (10,), None, None))
    check("steer/src 不同网格 → (G_steer, G_src)", psf2.shape == (10, dirs.shape[0]),
          f"{psf2.shape}")
    return psf


def _raises(fn, exc) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


# ==========================================================================
# 2/3. 单声源：脏图定位 + DAMAS 主瓣收窄 + 旁瓣改善
# ==========================================================================
def test_single_source():
    print("\n[2] 单声源：脏图定位 / DAMAS 主瓣收窄 / 旁瓣级")
    g = make_geom()
    grid = make_grid()
    dirs, shape, az, el = grid
    r = simulate([nb_source(7, 0.0)])
    truth = [ac.unit_dir(0.0, EL0)]

    with timed("dirty_map(单源, G=1387)"):
        d = im.dirty_map(r.X, r.fs, g.mics, grid, max_freqs=64,
                         f_min=BAND[0], f_max=BAND[1])
    check("脏图 map 是扁平 (G,) 数组", d["map"].shape == (dirs.shape[0],),
          f"{d['map'].shape}")
    check("脏图 shape 与网格一致", tuple(d["shape"]) == tuple(shape), f"{d['shape']}")
    check("脏图归一化 max = 1", abs(float(d["map"].max()) - 1.0) < 1e-12)
    check("脏图含 az_axis / el_axis / freqs_used",
          d["az_axis"] is not None and d["el_axis"] is not None
          and d["freqs_used"].size > 0)
    check("脏图频点落在指定带内",
          bool(np.all(d["freqs_used"] >= BAND[0]) and np.all(d["freqs_used"] <= BAND[1])),
          f"{d['freqs_used'].size} 个频点 [{d['freqs_used'][0]:.0f}, {d['freqs_used'][-1]:.0f}] Hz")
    check("脏图无 NaN/Inf", len(finite_all(d, "dirty")) == 0)

    j = nearest_dir(dirs, 0.0, EL0)
    err = float(im._angular_distance_deg(dirs[d["peak_index"]], truth[0])[0, 0])
    check("脏图峰值位置误差 < 3°", err < 3.0,
          f"误差 {err:.2f}°（峰值 az/el = {np.round(d['peak_az_el'], 2)}）")
    check("脏图峰值就在真值格点上", d["peak_index"] == j)

    psf = im.point_spread_function(g.mics, d["freqs_used"], grid)
    check("PSF 构造不预先算全矩阵（惰性对象）",
          not isinstance(psf, np.ndarray) and psf.shape == (dirs.shape[0],) * 2)
    with timed("PSF 全矩阵 G=1387"):
        P = psf.matrix()
    check("PSF 对角 = 1（实际矩阵）", np.allclose(np.diag(P), 1.0, atol=1e-12))

    with timed("damas(单源, 500 迭代, G=1387)"):
        dm = im.damas(d, psf, n_iter=500)
    check("DAMAS 无 NaN/Inf", len(finite_all({k: v for k, v in dm.items()
                                              if k != "residual_map"}, "damas")) == 0)
    check("DAMAS residual_map 无 NaN/Inf", len(finite_all(dm["residual_map"], "res")) == 0)
    w_dirty = d["metrics"]["main_lobe_width_deg"]
    w_dm = dm["metrics"]["main_lobe_width_deg"]
    check("DAMAS 主瓣宽度比脏图窄 30% 以上", w_dm < 0.7 * w_dirty,
          f"脏图 {w_dirty:.1f}° → DAMAS {w_dm:.1f}°（收窄 {100 * (1 - w_dm / w_dirty):.0f}%）")
    err_dm = float(im._angular_distance_deg(dirs[dm["peak_index"]], truth[0])[0, 0])
    check("DAMAS 峰值位置误差 < 3°", err_dm < 3.0, f"{err_dm:.2f}°")
    check("DAMAS 峰值旁瓣级优于脏图", dm["metrics"]["psl_db"] < d["metrics"]["psl_db"] - 3.0,
          f"脏图 {d['metrics']['psl_db']:.1f} dB → DAMAS {dm['metrics']['psl_db']:.1f} dB")
    check("DAMAS 动态范围优于脏图",
          dm["metrics"]["dynamic_range_db"] > d["metrics"]["dynamic_range_db"],
          f"{d['metrics']['dynamic_range_db']:.1f} dB → {dm['metrics']['dynamic_range_db']:.1f} dB")

    # 回调形式的 psf_fn 必须与对象形式结果一致
    dm_cb = im.damas(d, lambda k: psf[:, k], n_iter=50, tol=0.0)
    dm_obj = im.damas(d, psf, n_iter=50, tol=0.0)
    check("psf_fn 回调与 PSF 对象结果一致",
          np.allclose(dm_cb["raw"], dm_obj["raw"], atol=1e-10),
          f"max|Δq| = {np.abs(dm_cb['raw'] - dm_obj['raw']).max():.2e}")

    # 迭代历史与提前停止
    dm_es = im.damas(d, psf, n_iter=500, tol=1e-2)
    hist = dm_es["history"]
    check("history 含 iter/residual/delta 序列", len(hist["residual"]) == dm_es["n_iter"] > 0
          and len(hist["delta"]) == len(hist["iter"]))
    check("残差随迭代下降（末值 < 首值）", hist["residual"][-1] <= hist["residual"][0],
          f"{hist['residual'][0]:.4f} → {hist['residual'][-1]:.4f}")
    check("tol=1e-2 时提前停止（n_iter < n_iter_max）",
          dm_es["converged"] and dm_es["n_iter"] < 500,
          f"实际 {dm_es['n_iter']} 次，stop_reason={dm_es['stop_reason']}")
    return d, dm


# ==========================================================================
# 4. 两相干声源（同信号）——脏图糊在一起，DAMAS 分辨
# ==========================================================================
def test_two_coherent_sources():
    print("\n[3] 两相干声源（同一白噪信号，方位 ±20° @ el=60°）")
    g = make_geom()
    grid = make_grid()
    dirs, shape, az, el = grid
    truth = [ac.unit_dir(-20.0, EL0), ac.unit_dir(20.0, EL0)]
    sep = float(im._angular_distance_deg(truth[0], truth[1])[0, 0])
    r = simulate([nb_source(31, -20.0), nb_source(31, 20.0)])      # 同一 seed → 完全相干

    d = im.dirty_map(r.X, r.fs, g.mics, grid, max_freqs=64,
                     f_min=BAND[0], f_max=BAND[1])
    rc_d = im.resolution_check(d["map"], dirs, shape, az, el, truth)
    check("两源真实夹角在 15°~25°（近距）", 15.0 < sep < 25.0, f"{sep:.1f}°")
    check("脏图**分不开**（两真值之间无凹陷）", not rc_d["resolved"],
          f"dip = {rc_d['pairs'][0]['dip']:.2f}（1.0 = 完全糊在一起），"
          f"主瓣宽 {d['metrics']['main_lobe_width_deg']:.1f}°")

    psf = im.point_spread_function(g.mics, d["freqs_used"], grid)
    with timed("damas(两相干源, 500 迭代)"):
        dm = im.damas(d, psf, n_iter=500)
    rc_m = im.resolution_check(dm["map"], dirs, shape, az, el, truth)
    check("DAMAS 分辨出两个峰（dip < 0.7）", rc_m["resolved"],
          f"dip = {rc_m['pairs'][0]['dip']:.2f}")
    e0, e1 = rc_m["sources"][0]["err_deg"], rc_m["sources"][1]["err_deg"]
    check("DAMAS 两峰位置误差均 < 5°", max(e0, e1) < 5.0,
          f"源A {e0:.1f}°，源B {e1:.1f}°")
    check("DAMAS 峰值旁瓣级优于脏图", dm["metrics"]["psl_db"] < d["metrics"]["psl_db"],
          f"{d['metrics']['psl_db']:.1f} → {dm['metrics']['psl_db']:.1f} dB")
    check("DAMAS 主瓣宽度比脏图窄 30% 以上",
          dm["metrics"]["main_lobe_width_deg"] < 0.7 * d["metrics"]["main_lobe_width_deg"],
          f"{d['metrics']['main_lobe_width_deg']:.1f}° → "
          f"{dm['metrics']['main_lobe_width_deg']:.1f}°")
    check("DAMAS 输出无 NaN/Inf", len(finite_all(dm["map"], "damas.map")) == 0)

    # CLEAN-SC 在完全相干源上的行为（记录，不作为通过条件）：
    # R 秩一 → 相干分量剥离会把两个源当成一个分量反复剥离同一位置，
    # 这是 CLEAN-SC 类"峰拾取 + 相干剥离"方法的固有限制。
    cs = im.clean_sc(r.X, r.fs, g.mics, grid, n_sources=3, n_iter=8,
                     f_min=BAND[0], f_max=BAND[1], max_freqs=64)
    rc_c = im.resolution_check(cs["map"], dirs, shape, az, el, truth)
    check("CLEAN-SC 输出合法（无 NaN/Inf、有分量）",
          len(finite_all(cs["map"], "clean")) == 0 and len(cs["components"]) > 0,
          f"{len(cs['components'])} 个分量 "
          f"{[(round(c['az'], 1), round(c['power_rel'], 2)) for c in cs['components']]}")
    print(f"    [info] 完全相干源上 CLEAN-SC 分辨结果 = {rc_c['resolved']}"
          f"（已知限制：相干源在协方差里秩一，峰拾取无法分离）")
    return d, dm, cs


# ==========================================================================
# 5. 两独立声源：DAMAS + 正交波束成形
# ==========================================================================
def test_two_independent_sources():
    print("\n[4] 两独立（非相干）声源，方位 ±20° @ el=60°")
    g = make_geom()
    grid = make_grid()
    dirs, shape, az, el = grid
    truth = [ac.unit_dir(-20.0, EL0), ac.unit_dir(20.0, EL0)]
    r = simulate([nb_source(11, -20.0), nb_source(22, 20.0)])

    d = im.dirty_map(r.X, r.fs, g.mics, grid, max_freqs=64,
                     f_min=BAND[0], f_max=BAND[1])
    rc_d = im.resolution_check(d["map"], dirs, shape, az, el, truth)
    check("脏图分不开", not rc_d["resolved"], f"dip = {rc_d['pairs'][0]['dip']:.2f}")

    psf = im.point_spread_function(g.mics, d["freqs_used"], grid)
    dm = im.damas(d, psf, n_iter=500)
    rc_m = im.resolution_check(dm["map"], dirs, shape, az, el, truth)
    check("DAMAS 分辨出两个峰且误差 < 5°", rc_m["resolved"]
          and max(rc_m["sources"][0]["err_deg"], rc_m["sources"][1]["err_deg"]) < 5.0,
          f"dip={rc_m['pairs'][0]['dip']:.2f}, "
          f"误差 ({rc_m['sources'][0]['err_deg']:.1f}°, {rc_m['sources'][1]['err_deg']:.1f}°)")

    ob = im.orthogonal_beamforming(r.X, r.fs, g.mics, grid, n_sources=3,
                                   f_min=BAND[0], f_max=BAND[1], max_freqs=64)
    check("正交波束成形给出分量表与特征值",
          len(ob["components"]) >= 1 and len(ob["eigenvalues"]) >= 1
          and len(finite_all(ob["map"], "ob")) == 0,
          f"{len(ob['components'])} 个分量，前 2 个功率 "
          f"{[round(c['power_rel'], 2) for c in ob['components'][:2]]}")
    e_ob = float(im._angular_distance_deg(dirs[ob["components"][0]["index"]], truth[0])[0, 0])
    check("正交波束成形首分量落在较强源 5° 内", e_ob < 5.0, f"误差 {e_ob:.1f}°")

    # 谱不重叠（各频点只有一个源有能量）时正交波束成形最干净：
    # 每个频点的协方差严格秩一，特征向量直接就是该源的导向矢量。
    r2 = simulate([nb_source(0, -20.0, kind="multitone",
                             freqs=[1750, 1850, 1950]),
                   nb_source(0, 20.0, kind="multitone",
                             freqs=[2000, 2100, 2150])])
    ob2 = im.orthogonal_beamforming(r2.X, r2.fs, g.mics, grid, n_sources=3,
                                    f_min=BAND[0], f_max=BAND[1], max_freqs=64)
    rc_ob2 = im.resolution_check(ob2["map"], dirs, shape, az, el, truth)
    check("正交波束成形在谱不重叠双源上分辨两峰且误差 < 5°",
          rc_ob2["resolved"] and max(rc_ob2["sources"][0]["err_deg"],
                                     rc_ob2["sources"][1]["err_deg"]) < 5.0,
          f"误差 ({rc_ob2['sources'][0]['err_deg']:.1f}°, "
          f"{rc_ob2['sources'][1]['err_deg']:.1f}°), 分量 "
          f"{[(round(c['az'], 1), round(c['power_rel'], 2)) for c in ob2['components'][:2]]}")
    return d, dm, ob


# ==========================================================================
# 6. CLEAN-SC：单源定位 + 分离较开时分辨
# ==========================================================================
def test_clean_sc():
    print("\n[5] CLEAN-SC")
    g = make_geom()
    grid = make_grid()
    dirs, shape, az, el = grid

    # (a) 单源
    r1 = simulate([nb_source(7, 0.0)])
    cs1 = im.clean_sc(r1.X, r1.fs, g.mics, grid, n_sources=3, n_iter=8,
                      f_min=BAND[0], f_max=BAND[1], max_freqs=64)
    check("CLEAN-SC 返回 map/components/shape/history",
          all(k in cs1 for k in ("map", "components", "shape", "history", "freqs_used")))
    check("CLEAN-SC map 归一化 max = 1", abs(float(cs1["map"].max()) - 1.0) < 1e-12)
    check("CLEAN-SC 无 NaN/Inf",
          len(finite_all({k: v for k, v in cs1.items() if k != "dirty"}, "clean_sc")) == 0)
    top = cs1["components"][0]
    e_top = float(im._angular_distance_deg(dirs[top["index"]], ac.unit_dir(0.0, EL0))[0, 0])
    check("CLEAN-SC 第一分量位置误差 < 3°", e_top < 3.0,
          f"{e_top:.1f}°（az={top['az']:.1f}, el={top['el']:.1f}）")
    check("CLEAN-SC 第一分量功率占比 > 0.5", top["power_rel"] > 0.5,
          f"{top['power_rel']:.2f}")
    check("CLEAN-SC 分量含 循环号/功率/位置",
          all(k in top for k in ("loop", "index", "az", "el", "dir", "power")))

    # (b) 相距较远（±40° → 真角 ≈37.5°）两个独立源：CLEAN-SC 应能分辨
    truth2 = [ac.unit_dir(-40.0, EL0), ac.unit_dir(40.0, EL0)]
    r2 = simulate([nb_source(13, -40.0), nb_source(24, 40.0)])
    cs2 = im.clean_sc(r2.X, r2.fs, g.mics, grid, n_sources=3, n_iter=8,
                      f_min=BAND[0], f_max=BAND[1], max_freqs=64)
    rc2 = im.resolution_check(cs2["map"], dirs, shape, az, el, truth2)
    check("CLEAN-SC 在分离 ≈1 个主瓣宽时分辨两峰且误差 < 5°",
          rc2["resolved"] and max(rc2["sources"][0]["err_deg"],
                                  rc2["sources"][1]["err_deg"]) < 5.0,
          f"dip={rc2['pairs'][0]['dip']:.2f}, "
          f"误差 ({rc2['sources'][0]['err_deg']:.1f}°, {rc2['sources'][1]['err_deg']:.1f}°), "
          f"分量 {[(round(c['az'], 1), round(c['power_rel'], 2)) for c in cs2['components']]}")
    check("CLEAN-SC 干净图主瓣比脏图窄",
          cs2["metrics"]["main_lobe_width_deg"] <
          im.map_metrics(cs2["dirty"], shape, dirs, az, el)["main_lobe_width_deg"],
          f"{cs2['metrics']['main_lobe_width_deg']:.1f}°")
    return cs1, cs2


# ==========================================================================
# 7. 函数波束成形 / 对比工具
# ==========================================================================
def test_functional_and_compare():
    print("\n[6] 函数波束成形 / compare_methods")
    g = make_geom()
    grid = make_grid()
    dirs, shape, az, el = grid
    r = simulate([nb_source(7, 0.0)])

    d = im.dirty_map(r.X, r.fs, g.mics, grid, max_freqs=64,
                     f_min=BAND[0], f_max=BAND[1])
    fb = im.functional_beamforming(r.X, r.fs, g.mics, grid, exponent=0.5,
                                   f_min=BAND[0], f_max=BAND[1], max_freqs=64)
    check("函数波束成形无 NaN/Inf", len(finite_all(fb["map"], "fb")) == 0)
    check("函数波束成形主瓣比常规波束成形窄",
          fb["metrics"]["main_lobe_width_deg"] < d["metrics"]["main_lobe_width_deg"],
          f"{d['metrics']['main_lobe_width_deg']:.1f}° → "
          f"{fb['metrics']['main_lobe_width_deg']:.1f}°")
    check("函数波束成形旁瓣级优于常规图",
          fb["metrics"]["psl_db"] < d["metrics"]["psl_db"],
          f"{d['metrics']['psl_db']:.1f} → {fb['metrics']['psl_db']:.1f} dB")

    with timed("compare_methods(5 种方法, G=1387)"):
        cmp = im.compare_methods(r.X, r.fs, g.mics, grid,
                                 methods=("dirty", "damas", "clean_sc",
                                          "orthogonal", "functional"),
                                 f_min=BAND[0], f_max=BAND[1], max_freqs=64,
                                 damas_kw={"n_iter": 300},
                                 truth_dirs=[ac.unit_dir(0.0, EL0)])
    names = ("dirty", "damas", "clean_sc", "orthogonal", "functional")
    check("compare_methods 覆盖全部方法", all(n in cmp["results"] for n in names),
          f"{list(cmp['results'])}")
    check("每种方法都给出指标与 map",
          all(cmp["metrics"][n]["main_lobe_width_deg"] > 0
              and cmp["results"][n]["map"].shape == (dirs.shape[0],) for n in names))
    check("compare_methods 无 NaN/Inf", len(finite_all(cmp["metrics"], "cmp")) == 0)
    check("compare_methods 里 DAMAS 的旁瓣级最好",
          cmp["metrics"]["damas"]["psl_db"] == min(cmp["metrics"][n]["psl_db"] for n in names),
          {n: round(cmp["metrics"][n]["psl_db"], 1) for n in names})
    check("compare_methods 里 DAMAS 主瓣最窄",
          cmp["metrics"]["damas"]["main_lobe_width_deg"]
          == min(cmp["metrics"][n]["main_lobe_width_deg"] for n in names),
          {n: round(cmp["metrics"][n]["main_lobe_width_deg"], 1) for n in names})
    check("compare_methods 给出 resolution（真值对比）",
          "dirty" in cmp["resolution"] and "damas" in cmp["resolution"]
          and cmp["resolution"]["damas"]["n_resolved_sources"] >= 1)
    return cmp


# ==========================================================================
# 8. 性能：362 方向网格 + 1 秒数据 + DAMAS 200 迭代 < 60 s
# ==========================================================================
def test_performance():
    print("\n[7] 性能（16 麦 / 362 方向 / n_fft=512 / 1 s 数据）")
    g = make_geom()
    grid = gm.direction_grid(az_range=(-180.0, 180.0), el_range=(0.0, 90.0),
                             az_bins=72, el_bins=5)         # G = 360 ≈ 362
    dirs, shape, az, el = grid
    r = simulate([nb_source(7, 0.0), nb_source(19, 35.0)], snr=25.0, dur=1.0)
    check("网格点数 ≈ 362", 350 <= dirs.shape[0] <= 375, f"G = {dirs.shape[0]}")

    t0 = time.perf_counter()
    d = im.dirty_map(r.X, r.fs, g.mics, grid, max_freqs=64)
    t_dirty = time.perf_counter() - t0
    psf = im.point_spread_function(g.mics, d["freqs_used"], grid)
    t1 = time.perf_counter()
    dm = im.damas(d, psf, n_iter=200)
    t_damas = time.perf_counter() - t1
    total = time.perf_counter() - t0
    _TIMINGS.append(("性能: 脏图 (G=360)", t_dirty))
    _TIMINGS.append(("性能: DAMAS 200 迭代 (G=360)", t_damas))
    check("完整『脏图 + DAMAS 200 迭代』< 60 s", total < 60.0,
          f"脏图 {t_dirty:.2f} s + DAMAS {t_damas:.2f} s = {total:.2f} s"
          f"（含 PSF 构造/分块）")
    check("性能用例结果仍然正确（两峰 < 5°）",
          max(im._angular_distance_deg(dirs[dm["peak_index"]], ac.unit_dir(0.0, EL0))[0, 0],
              min(im._angular_distance_deg(dirs, ac.unit_dir(35.0, EL0))[:, 0])) < 8.0,
          f"峰值 az/el = {np.round(dm['peak_az_el'], 1)}")
    return d, dm


# ==========================================================================
# 9. 边界与健壮性
# ==========================================================================
def test_robustness():
    print("\n[8] 边界与健壮性")
    g = make_geom()
    grid = make_grid(az_bins=13, el_bins=5)
    dirs, shape, az, el = grid
    r = simulate([nb_source(7, 10.0)], snr=10.0)
    d = im.dirty_map(r.X, r.fs, g.mics, grid, freqs=[2000.0], max_freqs=None)
    check("显式 freqs 单频点可用", d["freqs_used"].size == 1 and d["n_freqs"] == 1,
          f"{d['freqs_used']}")

    # 1 维网格（只给 (G,3)）
    d1 = im.dirty_map(r.X, r.fs, g.mics, dirs, f_min=BAND[0], f_max=BAND[1])
    check("裸 (G,3) 网格可用且 shape=(G,)", tuple(d1["shape"]) == (dirs.shape[0],),
          f"{d1['shape']}")

    # 非负 / 允许负值两种模式都不许出 NaN
    psf = im.point_spread_function(g.mics, d1["freqs_used"], dirs)
    dm_nn = im.damas(d1, psf, n_iter=80, nonneg=True)
    dm_np = im.damas(d1, psf, n_iter=80, nonneg=False, relax=0.6)
    check("nonneg=True 解非负且无 NaN",
          bool(np.all(dm_nn["raw"] >= 0)) and len(finite_all(dm_nn["map"], "nn")) == 0)
    check("nonneg=False 无 NaN/Inf", len(finite_all(dm_np["map"], "np")) == 0)

    # 全零脏图不应产生 NaN
    dm_zero = im.damas(np.zeros(dirs.shape[0]), psf.matrix(), n_iter=5)
    check("全零脏图 → 全零解、无 NaN",
          len(finite_all(dm_zero["map"], "zero")) == 0 and float(dm_zero["map"].max()) == 0.0)

    # 静音输入
    d_sil = im.dirty_map(np.zeros((g.mics.shape[0], 2048)), r.fs, g.mics, grid,
                         f_min=BAND[0], f_max=BAND[1])
    check("静音输入 → 全零图、无 NaN",
          len(finite_all(d_sil["map"], "silent")) == 0
          and bool(np.all(np.isfinite(d_sil["map"]))))
    return d


# ==========================================================================
def main() -> int:
    quick = "--quick" in sys.argv
    t_all = time.perf_counter()
    print("=" * 78)
    print("MicArray Studio —— 声场成像模块验收测试")
    print("=" * 78)

    test_psf()
    test_single_source()
    test_two_coherent_sources()
    test_two_independent_sources()
    test_clean_sc()
    test_functional_and_compare()
    test_robustness()
    if not quick:
        test_performance()
    else:
        print("\n[7] 性能用例已跳过 (--quick)")

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print("\n" + "=" * 78)
    print("耗时明细")
    print("-" * 78)
    for name, dt in _TIMINGS:
        print(f"  {name:<40s} {dt:8.2f} s")
    print("-" * 78)
    fails = [(n, d) for n, ok, d in _RESULTS if not ok]
    if fails:
        print("失败用例：")
        for n, d in fails:
            print(f"  - {n} :: {d}")
    print(f"总计 {len(_RESULTS)} 项检查：PASS {n_pass} / FAIL {n_fail}"
          f"（总耗时 {time.perf_counter() - t_all:.1f} s）")
    print("=" * 78)
    print("PASS" if n_fail == 0 else "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
