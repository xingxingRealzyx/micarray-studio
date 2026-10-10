"""波束成形自检：解析性质 + 数值行为 + 阵列几何对比。

纯 Python + assert，运行方式：
  <runtime-python> tests/test_beamforming.py
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from engine import geometry as G, beamforming as bf, acoustics as ac, steering as st
from engine import sim, dsp

FAIL = []


def check(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def cut_beamwidth(B, dirs, u0, level_db=-3.0):
    """沿最短夹角切面找 -3dB 交点（网格不含真指向时返回 nan）。"""
    ang = np.rad2deg(np.arccos(np.clip(dirs @ u0, -1, 1)))
    o = np.argsort(ang)
    p = B[o]
    thr = 10 ** (level_db / 10)
    below = np.where(p <= thr)[0]
    return float(ang[o][below[0]] * 2) if below.size else float("nan")


print("=== 1. 解析性质：WNG(DS) == M，约束 w^H a = 1 ===")
fs = np.array([300., 1000., 3000., 6000.])
for name in ["spiral-4x4", "uca-8", "vogel-16", "grid-4x4", "nested-4-4"]:
    g = G.build_preset(name)
    M = g.n_mics
    r = bf.design(g, fs, steer=(0, 90), method="delay_sum")
    a = st.steering_vector(g.mics, fs, ac.unit_dir(0, 90))
    wHa = np.abs(np.einsum("fm,fm->f", r.weights.conj(), a.conj()))
    check(f"{name} WNG==10log10(M)",
          np.allclose(r.metrics["wng_db"], 10 * np.log10(M), atol=2e-3),
          f"WNG={r.metrics['wng_db'][0]:.3f} 期望={10*np.log10(M):.3f}")
    check(f"{name} w^H a == 1", np.abs(wHa - 1).max() < 1e-9,
          f"max|Δ|={np.abs(wHa-1).max():.1e}")

print("\n=== 2. 约束在 MVDR/超指向/频不变下同样成立 ===")
g = G.build_preset("spiral-4x4")
for mth in ["delay_sum", "superdirective", "identity", "fi"]:
    r = bf.design(g, fs, steer=(0, 90), method=mth)
    a = st.steering_vector(g.mics, fs, ac.unit_dir(0, 90))
    v = np.abs(np.einsum("fm,fm->f", r.weights.conj(), a.conj()))
    check(f"{mth} 满足 w^H a = 1", np.abs(v - 1).max() < 1e-8,
          f"max|Δ|={np.abs(v-1).max():.2e}")

print("\n=== 3. MVDR 自适应置零：用 SINR 提升衡量（正确口径）===")
# 三条教训（都踩过）：
#  1) 信号与干扰若同源（同 seed），两者完全相干，MVDR 假设被违反 -> 结果无意义
#  2) 沿某一条切面读一个点的"抑制"不可靠，必须对干扰协方差积分或直接算 SINR
#  3) 协方差必须含噪声底，否则秩亏导致解病态（真实系统总有自噪）
cfg = sim.SimConfig(fs=16000, duration=2.0)
az_sig, az_int = 0.0, 60.0
Xs = sim.simulate(g, [sim.SourceSpec(signal="white", params={"seed": 11},
                                    direction=(az_sig, 0), distance=1.0)], cfg).X
Xi = sim.simulate(g, [sim.SourceSpec(signal="white", params={"seed": 77},
                                    direction=(az_int, 0), distance=1.0,
                                    amplitude=1.6)], cfg).X
Xn = sim.simulate(g, [sim.SourceSpec(signal="white", params={"seed": 5},
                                    direction=(0, 90), distance=1.0,
                                    amplitude=0.15)], cfg).X
X = Xs + Xi + Xn
Xf, fr = dsp.stft(X, 16000, n_fft=256)
fsel = np.array([1000., 1062.5, 1125.])
idx = [int(np.argmin(np.abs(fr - v))) for v in fsel]


def cov_of(sig):
    S, _ = dsp.stft(sig, 16000, n_fft=256)
    return (np.einsum("mkf,nkf->fmn", S, S.conj()) / S.shape[1])[idx]


R, Ri = cov_of(X), cov_of(Xi)
az = np.linspace(-180, 180, 1441)
dirs = ac.dirs_from_angles(az, np.zeros(1441))
i0 = int(np.argmin(np.abs(az - az_sig)))

# 单通道基线 SINR
S_s, _ = dsp.stft(Xs, 16000, n_fft=256)
S_i, _ = dsp.stft(Xi + Xn, 16000, n_fft=256)
base = 10 * np.log10(np.mean(np.abs(S_s[0]) ** 2) / np.mean(np.abs(S_i[0]) ** 2))
print(f"    单通道基线 SINR = {base:.1f} dB")


def sinr_out(r):
    """输出信干噪比：信号方向功率 / (干扰协方差 + 噪声) 输出功率。"""
    w = r.weights
    A = st.steering_grid(g.mics, r.freqs, dirs)
    p_s = np.abs(np.einsum("fm,fm->f", w.conj(), A[:, i0, :].conj())) ** 2
    Rw = np.einsum("fij,fj->fi", Ri, w)
    p_i = np.einsum("fi,fi->f", w.conj(), Rw).real
    return 10 * np.log10(np.mean(p_s) / np.mean(p_i))


r_ds = bf.design(g, fsel, steer=(az_sig, 0), method="delay_sum")
imp_ds = sinr_out(r_ds) - base
print(f"    DS           SINR 提升 {imp_ds:6.2f} dB   WNG {r_ds.metrics['wng_min_db']:6.1f} dB")
rows = []
for ld in [0.0, 1e-4, 1e-2, 0.1, 1.0]:
    r = bf.design(g, fsel, steer=(az_sig, 0), method="mvdr", R_est=R, loading=ld)
    imp = sinr_out(r) - base
    a_ = st.steering_vector(g.mics, fsel, ac.unit_dir(az_sig, 0))
    wHa = np.abs(np.einsum("fm,fm->f", r.weights.conj(), a_))
    rows.append((ld, imp, r.metrics["wng_min_db"]))
    print(f"    MVDR ld={ld:<5} SINR 提升 {imp:6.2f} dB   WNG {r.metrics['wng_min_db']:6.1f} dB   "
          f"|w^H a|偏差 {np.abs(wHa-1).max():.1e}")
    check(f"MVDR 加载{ld} 严格满足 w^H a = 1", np.abs(wHa - 1).max() < 1e-8,
          f"max|Δ|={np.abs(wHa-1).max():.2e}")

# 诚实结论：MVDR 的价值是能形成**任意深的零陷**，而不保证在所有负载下都赢 DS。
# 在自由场 + 单干扰 + 阵元间距未混叠的几何下，DS 的天然旁瓣已把干扰压得不错
# （本例 26.9 dB），而中等加载的 MVDR 用置零深度换了鲁棒性，反而更低。
# MVDR 真正的优势场景是：干扰多/方向未知/混响场，见下面的多干扰验证。
check("轻加载 MVDR 的置零深度远超 DS", rows[1][1] > imp_ds + 10.0,
      f"MVDR(ld=1e-4)={rows[1][1]:.1f} dB vs DS={imp_ds:.1f} dB")

# 多干扰场景：DS 的旁瓣无法同时压多个方向，MVDR 可以
az_ints = [40.0, 75.0, 130.0]
Xmm = Xs.copy()
for m, a_i in enumerate(az_ints):
    Xmm = Xmm + sim.simulate(g, [sim.SourceSpec(signal="white", params={"seed": 100 + m},
                                                direction=(a_i, 0), distance=1.0,
                                                amplitude=1.2)], cfg).X
Rmm = cov_of(Xmm)
Rimm = cov_of(Xmm - Xs)
r_ds2 = bf.design(g, fsel, steer=(az_sig, 0), method="delay_sum")
r_mv2 = bf.design(g, fsel, steer=(az_sig, 0), method="mvdr", R_est=Rmm, loading=1e-3)


def sinr_ri(r, Ri_use):
    w = r.weights
    A = st.steering_grid(g.mics, r.freqs, dirs)
    p_s = np.abs(np.einsum("fm,fm->f", w.conj(), A[:, i0, :].conj())) ** 2
    Rw = np.einsum("fij,fj->fi", Ri_use, w)
    p_i = np.einsum("fi,fi->f", w.conj(), Rw).real
    return 10 * np.log10(np.mean(p_s) / np.mean(p_i))


i_ds = sinr_ri(r_ds2, Rimm)
i_mv = sinr_ri(r_mv2, Rimm)
print(f"    三干扰场景(40/75/130°): DS {i_ds:.1f} dB | MVDR {i_mv:.1f} dB")
check("多干扰下 MVDR 明显优于 DS（自适应优势的真实场景）", i_mv > i_ds + 3.0,
      f"MVDR={i_mv:.1f} dB vs DS={i_ds:.1f} dB")
check("无加载置零最深（但最不鲁棒）", rows[0][1] > rows[-1][1],
      f"ld=0: {rows[0][1]:.1f} dB -> ld=1: {rows[-1][1]:.1f} dB")
check("加载越大 WNG 越高（鲁棒性换置零深度）", rows[-1][2] > rows[0][2],
      f"WNG {rows[0][2]:.1f} -> {rows[-1][2]:.1f} dB")
check("加载增大使 SINR 单调下降（代价明确）",
      all(rows[i][1] >= rows[i+1][1] - 0.5 for i in range(len(rows)-1)),
      " > ".join(f"{v[1]:.0f}" for v in rows))

print("\n=== 4. 白噪增益下限约束：低频鲁棒化 ===")
flow = np.array([200., 300., 500.])
for floor in [-3.0, 0.0]:
    r = bf.design(g, flow, steer=(0, 90), method="superdirective", wng_floor_db=floor)
    check(f"WNG>={floor}dB 被满足", r.metrics["wng_min_db"] >= floor - 0.5,
          f"实测 min WNG={r.metrics['wng_min_db']:.2f} dB (加载 {r.extra['loading']:.2e})")
r_no = bf.design(g, flow, steer=(0, 90), method="superdirective")
check("无约束超指向 WNG 极低（说明约束有意义）", r_no.metrics["wng_min_db"] < -20,
      f"min WNG={r_no.metrics['wng_min_db']:.1f} dB")

print("\n=== 5. 频不变 vs DS：主瓣宽度一致性 ===")
dirs2, _, _, _ = G.direction_grid(az_bins=361, el_bins=181)
u0 = ac.unit_dir(0, 90)
fsb = np.array([400., 800., 1600., 3200.])
bws = {}
for mth in ["delay_sum", "fi"]:
    r = bf.design(g, fsb, steer=(0, 90), method=mth, fi_mainlobe_deg=50)
    B = bf.evaluate_beampattern(r.weights, g.mics, r.freqs, dirs2)
    bws[mth] = [cut_beamwidth(B[k], dirs2, u0) for k in range(fsb.size)]
    print(f"    {mth:10s} 主瓣宽度 {[round(v, 1) for v in bws[mth]]} 度")
r_ds = np.nanmax(bws["delay_sum"]) / np.nanmin(bws["delay_sum"])
r_fi = np.nanmax(bws["fi"]) / np.nanmin(bws["fi"])
check("频不变主瓣一致性优于 DS", r_fi < r_ds, f"FI 比值={r_fi:.2f} vs DS={r_ds:.2f}")

print("\n=== 6. 端到端：波束成形实际提升 SNR ===")
src = sim.SourceSpec(signal="speech", direction=(0, 90), distance=1.2)
r = sim.simulate(g, [src], sim.SimConfig(fs=16000, duration=1.0),
                 noise=sim.NoiseSpec(snr_db=5))
fr512 = np.fft.rfftfreq(512, 1 / 16000.0)
bfr = bf.design(g, fr512, steer=(0, 90), method="delay_sum", n_fft=512)
# 分别过 clean 与 noise，才能看出"信号保留"与"噪声抑制"两件事
ys = bf.process(bfr, r.clean, 16000, n_fft=512)
yn = bf.process(bfr, r.noise, 16000, n_fft=512)
gain_sig = np.std(ys) / np.std(r.clean[0])
gain_noi = np.std(yn) / np.std(r.noise[0])
snr_in = 20 * np.log10(np.std(r.clean[0]) / np.std(r.noise[0]))
snr_out = 20 * np.log10(np.std(ys) / np.std(yn))
imp = snr_out - snr_in
print(f"    信号保留 {gain_sig:.3f}x (期望≈1)，噪声抑制 {gain_noi:.3f}x")
print(f"    SNR {snr_in:.1f} dB -> {snr_out:.1f} dB，提升 {imp:.2f} dB "
      f"(16元理论上限 {10*np.log10(g.n_mics):.2f} dB)")
check("信号基本无损通过", abs(gain_sig - 1.0) < 0.05, f"{gain_sig:.3f}x")
check("噪声被显著抑制", gain_noi < 0.5, f"{gain_noi:.3f}x")
check("SNR 提升接近 10log10(M)", imp > 10 * np.log10(g.n_mics) - 2.0,
      f"{imp:.2f} dB vs 上限 {10*np.log10(g.n_mics):.2f} dB")
# 与解析权重逐位一致
a = st.steering_vector(g.mics, fr512, ac.unit_dir(0, 90))
yideal = bf.process(bf.BeamformerResult(freqs=fr512, weights=a.conj() / g.n_mics,
                                        mics=g.mics), r.X, 16000, n_fft=512)
yfull = bf.process(bfr, r.X, 16000, n_fft=512)
check("与解析权重实现逐位一致",
      np.abs(yfull - yideal).max() / np.abs(yideal).max() < 1e-9)

print("\n=== 7. 扩散场模型诊断（DI 可超上限的物理原因）===")
d = bf.coherence_diagnostics(g.mics, np.array([300., 3000., 6000.]))
r6 = bf.design(g, np.array([6000.]), steer=(0, 90))
print("    gamma 最小特征值:", d["gamma_eig_min"])
print(f"    各向同性上限 {d['isotropic_bound_db']:.2f} dB, 6kHz DS 实际 DI "
      f"{r6.metrics['di_max_db']:.2f} dB -> 超限标记 {r6.metrics['di_exceeds_bound']}")
check("诊断字段长度正确", len(d["gamma_eig_min"]) == 3)

print("\n=== 8. 性能 ===")
t0 = time.time()
for _ in range(5):
    bf.design(g, fr512, steer=(0, 90), method="superdirective")
t1 = time.time()
dirs3, _, _, _ = G.direction_grid(az_bins=181, el_bins=91)
r = bf.design(g, np.array([1000.]), steer=(0, 90))
bf.evaluate_beampattern(r.weights, g.mics, r.freqs, dirs3)
t2 = time.time()
print(f"    257 频点超指向设计 x5: {t1-t0:.2f}s ({(t1-t0)/5*1000:.0f} ms/次)")
print(f"    181x91 网格波束图: {t2-t1:.2f}s")
check("257 频点设计 < 0.5s", (t1 - t0) / 5 < 0.5)
check("波束图计算 < 2s", (t2 - t1) < 2.0)

print("\n" + "=" * 62)
print(f"结果: {'全部通过' if not FAIL else str(len(FAIL)) + ' 项失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
