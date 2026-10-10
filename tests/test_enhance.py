#!/usr/bin/env python3
"""多通道语音增强模块的验收测试（纯 Python + assert，无 pytest / scipy / matplotlib）。

跑法（必须用带 numpy 2.3.5 的解释器）::

    python3 \
        tests/test_enhance.py

覆盖内容
--------
1. **混响 + 噪声**（主场景）：speech 声源 + ``NoiseSpec(snr_db=5)`` + 中等混响房间
   （默认 ``RoomSpec``，Sabine 估算 RT60 ≈ 0.31 s）。验证
   MVDR+后置滤波 分段 SNR 提升 ≥ 6 dB、SI-SDR 不下降；
   MWF SI-SDR 提升 ≥ 3 dB；GSC 输出 SNR 提升 ≥ 5 dB 且收敛曲线单调不增。
2. **消融**：无混响、只有噪声时各方法仍然有效（逐个列出实测提升）。
3. **鲁棒性**：``MismatchSpec(gain_db_std=1.0, phase_deg_std=8.0)`` 下 MVDR 不应崩溃
   （列出退化多少 dB），并给出该场景下"任何线性波束成形器"的性能上限做对照。
4. **数值安全**：所有输出无 NaN/Inf、长度与输入一致、params/metrics 可
   ``json.dumps(..., allow_nan=False)``（CONTRACT 第 8 节硬要求）。
5. 单通道谱减基线与去混响的合理性检查（含一个受控的单反射 WPE 实验，
   用白噪声激励，证明去混响机制本身是正确的）。

评价口径（重要，先说清楚）
--------------------------
* **参考信号 = ``sim.clean[ref_mic]``**（含混响的干净观测），
  **输入基线 = ``sim.X[ref_mic]``**（同一通道的含噪观测）。
  这是语音增强里标准的单通道噪声抑制评价：把"混响"当作信号的一部分，
  因此各方法**不因去混响而得分**（去混响另有专门测试），
  也不因波束成形的空间选择性不同而被额外惩罚 —— 口径保守、可跨方法比较。
* ``snr_gain``  : 全局 SNR 提升 = 10log10(Σref²/Σ(y-ref)²) − 同式(输入)。
* ``seg_gain``  : ``engine.signals.segmental_snr_db`` 的提升（题目指定用它）。
  注意该实现的静音帧门限很宽松（相对 -100 dB），实测 **24/94 帧是近静音帧**
  （pr/pn ≈ −55 dB），它们的提升会主导均值，所以下表额外给出
  ``seg*_gain``（只在相对能量 > −25 dB 的"有效语音帧"上统计）做对照。
* ``si_gain``   : ``engine.signals.si_sdr`` 的提升（尺度不变）。
* ``proxy_gain``: ``engine.enhance.intelligibility_proxy``（0..1）的提升。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import enhance as en          # noqa: E402
from engine import geometry as geo        # noqa: E402
from engine import signals as sg          # noqa: E402
from engine import sim                    # noqa: E402

FS = 16000
DUR = 3.0
STEER = (0.0, 60.0)                       # (az, el) 度，阵列坐标系；el 从 xy 平面量向 +z
GEOM = geo.build_preset("uca-8")          # 8 元圆阵，半径 60 mm
CFG = sim.SimConfig(fs=FS, duration=DUR, center=(0.0, 0.0, 2.5),
                    boresight=(0.0, -90.0))
SRC = sim.SourceSpec(signal="speech", direction=STEER, distance=1.5, amplitude=1.0)
ROOM = sim.RoomSpec()                     # 默认鞋盒房间，RT60 ≈ 0.31 s（中等混响）

_RESULTS: list[tuple[str, str, bool, str]] = []


# --------------------------------------------------------------------------
# 断言收集器：不中断执行，最后统一打印 PASS/FAIL
# --------------------------------------------------------------------------
def check(group: str, name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((group, name, bool(ok), detail))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


# --------------------------------------------------------------------------
# 场景与指标
# --------------------------------------------------------------------------
def build_scenario(room: bool = True, snr_db: float | None = 5.0,
                   mismatch: sim.MismatchSpec | None = None,
                   coherent: bool = False) -> sim.SimResult:
    return sim.simulate(GEOM, [SRC], CFG,
                        room=ROOM if room else None,
                        noise=None if snr_db is None else sim.NoiseSpec(
                            snr_db=snr_db, seed=1, coherent=coherent),
                        mismatch=mismatch)


def seg_snr_active(ref: np.ndarray, test: np.ndarray, fs: int,
                   frame_ms: float = 32.0, rel_db: float = -25.0) -> float:
    """只在"有效语音帧"上统计的分段 SNR（对 engine 版本的口径补充）。

    engine.signals.segmental_snr_db 的静音门限是相对 -100 dB，几乎不过滤，
    因此近静音帧（pr/pn ≈ -55 dB）会主导均值。这里把门限提到相对最大帧能量
    ``rel_db`` dB，只统计真正的语音帧，用来交叉验证（不替代题目指定的指标）。
    """
    ref = np.asarray(ref, float).ravel()
    test = np.asarray(test, float).ravel()
    L = max(int(fs * frame_ms / 1000.0), 16)
    K = ref.size // L
    if K < 2:
        return float("nan")
    r = ref[:K * L].reshape(K, L)
    t = test[:K * L].reshape(K, L)
    pr = np.mean(r ** 2, axis=1)
    pn = np.mean((t - r) ** 2, axis=1)
    keep = pr > (10.0 ** (rel_db / 10.0)) * pr.max()
    if not np.any(keep):
        return float("nan")
    return float(np.mean(10.0 * np.log10(np.maximum(pr[keep], 1e-30)
                                         / np.maximum(pn[keep], 1e-30))))


def measure(ref: np.ndarray, inp: np.ndarray, out: np.ndarray) -> dict:
    """一组增强指标：SNR / 分段 SNR / SI-SDR / 可懂度代理 的"输出 − 输入"。"""
    n = min(ref.size, inp.size, out.size)
    ref, inp, out = ref[:n], inp[:n], out[:n]
    snr_in = sg.snr_db(ref, inp - ref)
    snr_out = sg.snr_db(ref, out - ref)
    seg_in = sg.segmental_snr_db(ref, inp, FS)
    seg_out = sg.segmental_snr_db(ref, out, FS)
    seg_a_in = seg_snr_active(ref, inp, FS)
    seg_a_out = seg_snr_active(ref, out, FS)
    si_in = sg.si_sdr(ref, inp)
    si_out = sg.si_sdr(ref, out)
    pr_in = en.intelligibility_proxy(ref, inp, FS)
    pr_out = en.intelligibility_proxy(ref, out, FS)
    return {
        "snr_gain": snr_out - snr_in,
        "seg_gain": seg_out - seg_in,
        "seg_active_gain": seg_a_out - seg_a_in,
        "si_gain": si_out - si_in,
        "proxy_gain": pr_out - pr_in,
        "snr_in": snr_in, "snr_out": snr_out,
        "seg_in": seg_in, "seg_out": seg_out,
        "si_in": si_in, "si_out": si_out,
        "proxy_in": pr_in, "proxy_out": pr_out,
    }


def run_method(result: sim.SimResult, method: str, **kw) -> tuple[dict, dict]:
    """跑一个增强方法，返回 (增强输出 dict, 指标 dict)。"""
    if method == "mvdr":
        # 默认把 ArrayGeometry 传进去，让 engine.beamforming.design() 走通
        # （CONTRACT 第 6 节：优先用 beamforming 模块；不可用时内部自动回退）
        kw.setdefault("geom", GEOM)
        out = en.mvdr_postfilter(result.X, FS, GEOM.mics, STEER, **kw)
    elif method == "mwf":
        out = en.multichannel_wiener(result.X, FS, GEOM.mics, STEER, **kw)
    elif method == "gsc":
        out = en.gsc(result.X, FS, GEOM.mics, STEER, **kw)
    elif method == "specsub":
        out = en.spectral_subtraction(result.X, FS, **kw)
    elif method == "dereverb_spectral":
        out = en.dereverberate(result.X, FS, method="spectral",
                               mics=GEOM.mics, steer_dir=STEER, **kw)
    elif method == "dereverb_wpe":
        out = en.dereverberate(result.X, FS, method="wpe",
                               mics=GEOM.mics, steer_dir=STEER, **kw)
    else:
        raise ValueError(method)
    return out, measure(result.clean[0], result.X[0], out["y"])


# --------------------------------------------------------------------------
# 通用数值安全断言
# --------------------------------------------------------------------------
def assert_sane(tag: str, result: sim.SimResult, out: dict) -> None:
    y = out["y"]
    n = result.X.shape[1]
    check(tag, "返回 dict 含 y/method/params/metrics/curve",
          all(k in out for k in ("y", "method", "params", "metrics"))
          and "curve" in out)
    check(tag, "y 无 NaN/Inf", bool(np.all(np.isfinite(y))),
          "min=%.3e max=%.3e" % (np.min(y), np.max(y)))
    check(tag, "y 长度与输入一致 (%d)" % n, y.shape == (n,),
          "shape=%s" % (y.shape,))
    try:
        json.dumps(out["params"], allow_nan=False)
        json.dumps(out["metrics"], allow_nan=False)
        ok, why = True, ""
    except Exception as exc:                                   # pragma: no cover
        ok, why = False, f"{type(exc).__name__}: {exc}"
    check(tag, "params/metrics 可 json 序列化（无 NaN/Inf）", ok, why)


# ==========================================================================
# 场景 1：混响 + 噪声（主验收场景）
# ==========================================================================
def scenario_reverb_noise() -> tuple[dict, dict]:
    print("\n" + "=" * 100)
    print("场景 1：混响 + 噪声   (RoomSpec 默认, RT60≈%.2f s, NoiseSpec(snr_db=5))"
          % sim.estimate_rt60(ROOM))
    print("=" * 100)
    res = build_scenario(room=True, snr_db=5.0)
    print("仿真: X%s fs=%d 实际SNR=%.2f dB  c=%.0f m/s  阵列=%s(M=%d)"
          % (res.X.shape, res.fs, res.meta["snr_actual_db"], res.meta["c"],
             GEOM.name, GEOM.n_mics))
    rows, outs = {}, {}

    # ---- MVDR + 后置滤波 ----
    t0 = time.time()
    out, m = run_method(res, "mvdr", n_fft=512)
    outs["mvdr"] = out
    rows["MVDR+后置滤波"] = m
    print("\n-- MVDR+后置滤波 (%.2f s) 设计路径=%s (%s)"
          % (time.time() - t0, out["params"]["design_path"],
             out["params"]["design_note"]))
    print("   输入 SNR %+.2f dB → 输出 %+.2f dB ；分段 SNR %+.2f → %+.2f ；SI-SDR %+.2f → %+.2f"
          % (m["snr_in"], m["snr_out"], m["seg_in"], m["seg_out"], m["si_in"], m["si_out"]))
    check("场景1-MVDR", "分段 SNR 提升 >= 6 dB", m["seg_gain"] >= 6.0,
          "实测 %+.2f dB（有效语音帧口径 %+.2f dB）" % (m["seg_gain"], m["seg_active_gain"]))
    check("场景1-MVDR", "SI-SDR 不下降（>= 输入）", m["si_gain"] >= 0.0,
          "实测 %+.2f dB" % m["si_gain"])
    check("场景1-MVDR", "全局 SNR 提升 > 0", m["snr_gain"] > 0.0,
          "实测 %+.2f dB" % m["snr_gain"])
    check("场景1-MVDR", "可懂度代理提升 > 0", m["proxy_gain"] > 0.0,
          "实测 %+.4f（%.3f → %.3f）" % (m["proxy_gain"], m["proxy_in"], m["proxy_out"]))
    assert_sane("场景1-MVDR", res, out)

    # ---- 设计路径一致性：beamforming 模块 vs 本模块解析式 ----
    out_a, m_a = run_method(res, "mvdr", design="analytic")
    path = out["params"]["design_path"]
    if path == "beamforming":
        check("场景1-MVDR", "beamforming 模块与解析式设计结果一致（<=1 dB）",
              abs(m["snr_gain"] - m_a["snr_gain"]) <= 1.0
              and abs(m["si_gain"] - m_a["si_gain"]) <= 1.0,
              "模块 ΔSNR %+.2f / ΔSI-SDR %+.2f ；解析式 %+.2f / %+.2f"
              % (m["snr_gain"], m["si_gain"], m_a["snr_gain"], m_a["si_gain"]))
    else:
        check("场景1-MVDR", "beamforming 模块不可用时回退解析式（信息性）", True,
              "design_path=%s note=%s" % (path, out["params"]["design_note"]))

    # ---- 多通道维纳滤波 ----
    out, m = run_method(res, "mwf")
    outs["mwf"] = out
    rows["MWF"] = m
    print("\n-- MWF  (MWF 权重|w|中位数 %.3f, 收缩增益中位数 %.1f dB)"
          % (np.median(np.linalg.norm(out["weights"], axis=1)),
             out["metrics"].get("mwf_shrinkage_median_db", float("nan"))))
    check("场景1-MWF", "SI-SDR 提升 >= 3 dB", m["si_gain"] >= 3.0,
          "实测 %+.2f dB" % m["si_gain"])
    check("场景1-MWF", "分段 SNR 提升 > 0", m["seg_gain"] > 0.0,
          "实测 %+.2f dB（有效语音帧口径 %+.2f dB）"
          % (m["seg_gain"], m["seg_active_gain"]))
    assert_sane("场景1-MWF", res, out)

    # ---- GSC ----
    t0 = time.time()
    out, m = run_method(res, "gsc", mu=0.05)
    outs["gsc"] = out
    rows["GSC"] = m
    cv = out["curve"]
    print("\n-- GSC  (%.2f s)  自适应帧占比 %.2f  |h| 均值 %.4f"
          % (time.time() - t0, out["params"]["adapt_frac"],
             float(np.mean(np.abs(out["h_final"])))))
    check("场景1-GSC", "输出 SNR 提升 >= 5 dB", m["snr_gain"] >= 5.0,
          "实测 %+.2f dB" % m["snr_gain"])
    check("场景1-GSC", "收敛曲线 err_mono_db 单调不增",
          bool(np.all(np.diff(cv["err_mono_db"]) <= 1e-9)),
          "构造保证；实测总下降 %.2f dB" % cv["converged_drop_db"])
    check("场景1-GSC", "学习曲线 learn_mono_db 单调不增",
          bool(np.all(np.diff(cv["learn_mono_db"]) <= 1e-9)),
          "实测总下降 %.2f dB" % cv["learn_drop_db"])
    check("场景1-GSC", "原始平滑曲线整体下降（首帧 > 末帧）",
          float(cv["err_smooth_db"][-1]) < float(cv["err_smooth_db"][0]),
          "%.2f → %.2f dB" % (cv["err_smooth_db"][0], cv["err_smooth_db"][-1]))
    check("场景1-GSC", "SI-SDR 不下降", m["si_gain"] >= 0.0,
          "实测 %+.2f dB" % m["si_gain"])
    assert_sane("场景1-GSC", res, out)
    return rows, outs


# ==========================================================================
# 场景 2：消融（无混响，只有噪声）
# ==========================================================================
def scenario_anechoic() -> dict:
    print("\n" + "=" * 100)
    print("场景 2：消融 —— 无混响（room=None），只有 NoiseSpec(snr_db=5)")
    print("=" * 100)
    res = build_scenario(room=False, snr_db=5.0)
    print("仿真: 实际 SNR=%.2f dB（自由场，无 RIR）" % res.meta["snr_actual_db"])
    rows = {}
    for name, method in (("MVDR+后置滤波", "mvdr"), ("MWF", "mwf"),
                         ("GSC", "gsc"), ("单通道谱减", "specsub")):
        out, m = run_method(res, method)
        rows[name] = m
        print("  %-14s SNR %+5.2f dB | 分段 SNR %+5.2f dB | SI-SDR %+5.2f dB | 代理 %+.4f"
              % (name, m["snr_gain"], m["seg_gain"], m["si_gain"], m["proxy_gain"]))
        assert_sane("场景2-" + name, res, out)
        check("场景2-" + name, "仍然有效（SNR 提升 > 0 且 SI-SDR 不下降）",
              m["snr_gain"] > 0.0 and m["si_gain"] >= 0.0,
              "SNR %+.2f dB, SI-SDR %+.2f dB" % (m["snr_gain"], m["si_gain"]))
    return rows


# ==========================================================================
# 场景 3：鲁棒性（幅相失配）
# ==========================================================================
def scenario_mismatch() -> tuple[dict, dict]:
    print("\n" + "=" * 100)
    print("场景 3：鲁棒性 —— MismatchSpec(gain_db_std=1.0, phase_deg_std=8.0, seed=3)")
    print("=" * 100)
    mm = sim.MismatchSpec(gain_db_std=1.0, phase_deg_std=8.0, seed=3)
    res_mm = build_scenario(room=True, snr_db=5.0, mismatch=mm)
    res_no = build_scenario(room=True, snr_db=5.0)
    print("失配已施加: %s" % res_mm.meta["mismatch"]["applied"])
    print("通道增益误差(dB): %s" % np.round(res_mm.meta["mismatch"]["gain_db"], 2))
    print("通道相位误差(度): %s" % np.round(res_mm.meta["mismatch"]["phase_deg"], 2))

    out_no, m_no = run_method(res_no, "mvdr")
    out_mm, m_mm = run_method(res_mm, "mvdr")
    rows = {"MVDR(无失配)": m_no, "MVDR(幅相失配)": m_mm}
    print("\n  无失配 : SNR %+5.2f | 分段 SNR %+5.2f | SI-SDR %+5.2f | 代理 %+.4f"
          % (m_no["snr_gain"], m_no["seg_gain"], m_no["si_gain"], m_no["proxy_gain"]))
    print("  有失配 : SNR %+5.2f | 分段 SNR %+5.2f | SI-SDR %+5.2f | 代理 %+.4f"
          % (m_mm["snr_gain"], m_mm["seg_gain"], m_mm["si_gain"], m_mm["proxy_gain"]))
    deg = {k: m_no[k] - m_mm[k] for k in ("snr_gain", "seg_gain", "si_gain")}
    print("  退化量 : SNR %.2f dB | 分段 SNR %.2f dB | SI-SDR %.2f dB"
          % (deg["snr_gain"], deg["seg_gain"], deg["si_gain"]))

    # 参考上限：任何"线性空间滤波 + 完美后置滤波"在这个失配下的天花板
    ceil_ = _beamform_clean_ceiling(res_mm)
    print("  该失配场景下 DAS(干净) 的天花板: SNR %+.2f dB | SI-SDR %+.2f dB "
          "（即任何线性波束成形器最好也就这样）"
          % (ceil_["snr_gain"], ceil_["si_gain"]))

    check("场景3-失配", "MVDR 分段 SNR 仍为正提升", m_mm["seg_gain"] > 0.0,
          "%+.2f dB（无失配时 %+.2f dB，退化 %.2f dB）"
          % (m_mm["seg_gain"], m_no["seg_gain"], deg["seg_gain"]))
    check("场景3-失配", "MVDR SI-SDR 不下降", m_mm["si_gain"] >= 0.0,
          "%+.2f dB（退化 %.2f dB）" % (m_mm["si_gain"], deg["si_gain"]))
    check("场景3-失配", "MVDR 输出无 NaN/Inf", bool(np.all(np.isfinite(out_mm["y"]))))
    check("场景3-失配", "退化量在 6 dB 以内（不崩溃）",
          deg["seg_gain"] <= 6.0 and deg["si_gain"] <= 6.0,
          "SNR 退化 %.2f dB / SI-SDR 退化 %.2f dB" % (deg["snr_gain"], deg["si_gain"]))
    # 失配场景下 GSC / MWF 也顺带记录
    for name, method in (("MWF", "mwf"), ("GSC", "gsc")):
        out, m = run_method(res_mm, method)
        rows[name + "(失配)"] = m
        print("  %-14s SNR %+5.2f | 分段 SNR %+5.2f | SI-SDR %+5.2f"
              % (name, m["snr_gain"], m["seg_gain"], m["si_gain"]))
        assert_sane("场景3-" + name, res_mm, out)
    return rows, {"mvdr_no": m_no, "mvdr_mm": m_mm}


def _beamform_clean_ceiling(res: sim.SimResult) -> dict:
    """对该场景做"延迟求和(干净信号)"，得到线性波束成形能达到的上限。"""
    from engine import acoustics as ac
    from engine import steering as st
    Xf, freqs = en._stft(res.X, FS, 512, 256, "blackmanharris", True)
    a = st.steering_vector(GEOM.mics, freqs, ac.unit_dir(*STEER), 343.0)
    wq = en._das_weights(a)
    Cf, _ = en._stft(res.clean, FS, 512, 256, "blackmanharris", True)
    y = en._istft(np.einsum("mkf,fm->kf", Cf, np.conj(wq)), FS, 512, 256,
                  "blackmanharris", res.X.shape[1], True)
    return measure(res.clean[0], res.X[0], y)


# ==========================================================================
# 场景 4：基线与去混响
# ==========================================================================
def scenario_baselines() -> dict:
    print("\n" + "=" * 100)
    print("场景 4：单通道谱减基线 + 去混响（含受控单反射实验）")
    print("=" * 100)
    res = build_scenario(room=True, snr_db=5.0)
    rows = {}
    for tag, kw in (("谱减 β=2.0", dict(over_subtraction=2.0)),
                    ("谱减 β=1.0", dict(over_subtraction=1.0))):
        out, m = run_method(res, "specsub", **kw)
        rows[tag] = m
        print("  %-12s SNR %+5.2f | 分段 SNR %+5.2f | SI-SDR %+5.2f | 代理 %+.4f"
              % (tag, m["snr_gain"], m["seg_gain"], m["si_gain"], m["proxy_gain"]))
        assert_sane("场景4-" + tag, res, out)
    check("场景4-谱减", "单通道基线有正向 SNR 提升",
          rows["谱减 β=2.0"]["snr_gain"] > 0.0,
          "%+.2f dB（远低于多通道方法属正常：没有阵列增益）"
          % rows["谱减 β=2.0"]["snr_gain"])

    # 去混响：用"直达声参考"评价（参考里不含混响，才是去混响的正确口径）。
    # 注意：clean[0] 口径下这两项看似无提升，是因为该口径把晚期混响当成信号。
    for tag, method in (("去混响-谱减型", "dereverb_spectral"),
                        ("去混响-WPE", "dereverb_wpe")):
        out, _ = run_method(res, method)
        assert_sane("场景4-" + tag, res, out)
        si = sg.si_sdr(res.direct[0], out["y"])
        si_in = sg.si_sdr(res.direct[0], res.X[0])
        seg_in = sg.segmental_snr_db(res.direct[0], res.X[0], FS)
        seg_out = sg.segmental_snr_db(res.direct[0], out["y"], FS)
        rows[tag] = {"si_gain": si - si_in, "snr_gain": float("nan"),
                     "seg_gain": seg_out - seg_in, "seg_active_gain": float("nan"),
                     "proxy_gain": float("nan")}
        print("  %-12s vs 直达声参考: SI-SDR %+.2f → %+.2f dB (Δ%+.2f) 分段SNR Δ%+.2f"
              % (tag, si_in, si, si - si_in, seg_out - seg_in))
        check("场景4-" + tag, "去混响相对直达声参考有正向提升（ΔSI-SDR > 0）",
              (si - si_in) > 0.0, "实测 %+.2f dB" % (si - si_in))

    # 受控实验：单次 48 ms 反射 + 白噪声激励 —— WPE 机制在这里必须显著有效
    x = np.random.default_rng(0).standard_normal(int(FS * 3.0))
    d = int(FS * 48e-3)
    wet = x.copy()
    wet[d:] += 0.7 * x[:-d]
    stft, _ = en._stft(wet, FS, 512, 256, "blackmanharris", True)
    resid, info = en._wpe_single_channel(stft, delay=3, taps=4, iters=2)
    y_wpe = en._istft(resid, FS, 512, 256, "blackmanharris", wet.size, True)
    g_in, g_out = sg.si_sdr(x, wet), sg.si_sdr(x, y_wpe)
    print("  受控单反射(48ms, r=0.7): WPE(delay=3帧,taps=4) SI-SDR %+.2f → %+.2f dB"
          % (g_in, g_out))
    check("场景4-WPE机制", "受控单反射下 WPE 显著去混响（+3 dB 以上）",
          (g_out - g_in) > 3.0, "实测提升 %+.2f dB" % (g_out - g_in))
    return rows


# ==========================================================================
# 场景 5：边界与数值稳健性
# ==========================================================================
def scenario_robustness_numeric() -> None:
    print("\n" + "=" * 100)
    print("场景 5：边界条件与数值稳健性")
    print("=" * 100)
    # 5.1 极短信号（短于 n_fft）
    short = sim.simulate(GEOM, [SRC], sim.SimConfig(fs=FS, duration=0.02,
                                                     center=(0, 0, 2.5),
                                                     boresight=(0, -90)),
                         room=None, noise=sim.NoiseSpec(snr_db=5.0, seed=1))
    for name, method in (("mvdr", "mvdr"), ("mwf", "mwf"), ("gsc", "gsc")):
        out, _ = run_method(short, method)
        check("场景5-短时(0.02s)", "%s 输出长度/有限性" % name,
              out["y"].shape == (short.X.shape[1],)
              and bool(np.all(np.isfinite(out["y"]))),
              "N=%d" % short.X.shape[1])

    # 5.2 静音输入（全零）—— 除法/求逆最容易爆的地方
    zeros = np.zeros((GEOM.n_mics, FS))
    for name, fn in (("mvdr", lambda X: en.mvdr_postfilter(X, FS, GEOM.mics, STEER)),
                     ("mwf", lambda X: en.multichannel_wiener(X, FS, GEOM.mics, STEER)),
                     ("gsc", lambda X: en.gsc(X, FS, GEOM.mics, STEER)),
                     ("specsub", lambda X: en.spectral_subtraction(X, FS))):
        out = fn(zeros)
        check("场景5-全零输入", "%s 输出有限" % name,
              bool(np.all(np.isfinite(out["y"]))),
              "max|y|=%.3e" % np.max(np.abs(out["y"])))

    # 5.3 小阵列（2 麦）+ 单通道输入给谱减
    g2 = geo.ArraySpec("uca", {"n": 2, "radius": 0.05}).build()
    r2 = sim.simulate(g2, [SRC], CFG, room=ROOM, noise=sim.NoiseSpec(snr_db=5.0, seed=1))
    for name, fn in (("mvdr", lambda: en.mvdr_postfilter(r2.X, FS, g2.mics, STEER)),
                     ("mwf", lambda: en.multichannel_wiener(r2.X, FS, g2.mics, STEER)),
                     ("gsc", lambda: en.gsc(r2.X, FS, g2.mics, STEER))):
        out = fn()
        check("场景5-M=2", "%s 正常返回" % name,
              out["y"].shape == (r2.X.shape[1],)
              and bool(np.all(np.isfinite(out["y"]))))
    out = en.spectral_subtraction(r2.X[0], FS)          # 一维输入
    check("场景5-单通道入参", "spectral_subtraction 接受 (N,) 输入",
          out["y"].shape == (r2.X.shape[1],)
          and bool(np.all(np.isfinite(out["y"]))))

    # 5.4 近场坐标点作为 steer_dir（模长 ≠ 1 时按坐标处理，CONTRACT 第 3 节）
    out = en.mvdr_postfilter(r2.X, FS, g2.mics, (0.75, 0.0, 1.30))
    check("场景5-近场指向", "steer_dir 支持近场坐标点",
          bool(np.all(np.isfinite(out["y"]))))

    # 5.5 evaluate() 在给定/不给定 noisy 时的字段完整性
    r = build_scenario(room=True, snr_db=5.0)
    m = en.evaluate(r.clean[0], r.X[0], FS, noisy=r.X[0])
    check("场景5-evaluate", "输入=输出时提升为 0（自洽性）",
          abs(m["snr_improvement_db"]) < 1e-6
          and abs(m["si_sdr_improvement_db"]) < 1e-6,
          "dSNR=%.2e dSI=%.2e" % (m["snr_improvement_db"], m["si_sdr_improvement_db"]))
    m2 = en.evaluate(r.clean[0], r.X[0], FS)
    check("场景5-evaluate", "无 noisy 时提升字段为 None（可 json 序列化）",
          m2["snr_improvement_db"] is None and m2["si_sdr_in_db"] is None)
    try:
        json.dumps(m, allow_nan=False)
        ok = True
    except Exception:                                          # pragma: no cover
        ok = False
    check("场景5-evaluate", "evaluate 结果可 json 序列化", ok)


# ==========================================================================
# 性能总表
# ==========================================================================
def print_table(rows_by_scenario: dict) -> None:
    print("\n" + "=" * 100)
    print("各方法 × 各场景性能表   参考=sim.clean[ref_mic]，输入=sim.X[ref_mic]（单位 dB）")
    print("=" * 100)
    hdr = ("%-22s %8s %8s %8s %8s %9s" %
           ("方法", "ΔSNR", "Δ分段SNR", "Δ有效帧", "ΔSI-SDR", "Δ可懂度"))
    for scen, rows in rows_by_scenario.items():
        print("\n【%s】" % scen)
        print(hdr)
        print("-" * len(hdr))
        for name, m in rows.items():
            def f(key):
                v = m.get(key, float("nan"))
                return "%8.2f" % v if isinstance(v, float) and np.isfinite(v) else "     n/a"
            print("%-22s %s %s %s %s %9s" %
                  (name, f("snr_gain"), f("seg_gain"), f("seg_active_gain"),
                   f("si_gain"),
                   ("%+.4f" % m["proxy_gain"]) if isinstance(m.get("proxy_gain"), float)
                   and np.isfinite(m["proxy_gain"]) else "      n/a"))
    print("\n说明：Δ分段SNR = engine.signals.segmental_snr_db 的提升（题目指定）；"
          "Δ有效帧 = 只在相对能量 > -25 dB 的语音帧上统计的分段 SNR 提升")


def main() -> int:
    print("MicArray Studio —— 多通道语音增强验收测试")
    print("numpy %s | python %s" % (np.__version__, sys.version.split()[0]))
    print("阵列 %s: M=%d aperture=%.3f m min_spacing=%.3f m grid_free=%.0f Hz"
          % (GEOM.name, GEOM.n_mics, geo.aperture(GEOM.mics),
             geo.min_spacing(GEOM.mics), geo.grating_free_freq(GEOM.mics)))

    table = {}
    rows1, _ = scenario_reverb_noise()
    table["场景1 混响+噪声 (RT60≈0.31s, SNR=5dB)"] = rows1
    table["场景2 消融：无混响只有噪声 (SNR=5dB)"] = scenario_anechoic()
    rows3, _ = scenario_mismatch()
    table["场景3 鲁棒性：1dB/8° 幅相失配"] = rows3
    table["场景4 基线与去混响"] = scenario_baselines()
    scenario_robustness_numeric()

    print_table(table)

    n_fail = sum(1 for _, _, ok, _ in _RESULTS if not ok)
    n_pass = len(_RESULTS) - n_fail
    print("\n" + "=" * 100)
    groups: dict[str, list] = {}
    for g, _, ok, _ in _RESULTS:
        groups.setdefault(g, []).append(ok)
    for g, oks in groups.items():
        print("  %-22s %2d/%2d PASS%s" % (g, sum(oks), len(oks),
                                          "" if all(oks) else "   <-- 有失败"))
    print("=" * 100)
    print("总计: %d PASS / %d FAIL  ->  %s"
          % (n_pass, n_fail, "PASS" if n_fail == 0 else "FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
