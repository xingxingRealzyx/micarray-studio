"""C++17 算法导出层。

设计原则（见 docs/CONTRACT.md 第 7 节）：

1. **自包含**：导出的 C++ 不依赖 Python、不依赖 Eigen（可选）、不依赖 JSON 库，
   只用 C++17 标准库。测试向量走纯文本，避免在实时路径上引入解析依赖。
2. **可验证**：导出同时生成测试向量与比对程序，`export_and_verify()` 会真的
   调用 cmake/make 编译运行，把 Python 与 C++ 的最大相对误差报出来。
   比对不过就不算导出成功——这是"导出可用"与"导出一堆看起来对的代码"的分界。
3. **数值等价**：FFT 用同一套 radix-2 蝶形、窗表按有效数字量化后传给两侧，
   保证差异只来自浮点求和顺序（量级 1e-15）。

导出产物：

    export/<name>/
      include/mas_array_config.hpp    几何 + 常量 + 窗表
      include/mas_dsp.hpp  src/mas_dsp.cpp        FFT / 卷积 / STFT
      include/mas_beamform.hpp  src/mas_beamform.cpp
      include/mas_export_api.hpp      统一入口
      tests/test_parity.cpp           读测试向量并比对
      tests/vectors/*.txt             输入 / 期望输出
      CMakeLists.txt
      README.md
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

from . import acoustics as ac
from . import beamforming as bf
from . import dsp

C_DEFAULT = ac.C_DEFAULT
SIG = 17          # 导出常量的有效数字：足够保证 float64 逐位一致


def _lit(v: float, sig: int = SIG) -> str:
    """格式化一个 double 字面量（保证可编译）。"""
    f = float(v)
    if f != f:                      # NaN 不能出现在 C++ 字面量里
        raise ValueError("导出常量中出现 NaN")
    if f in (float("inf"), float("-inf")):
        raise ValueError("导出常量中出现 Inf")
    s = f"%.{sig}g" % f
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    if s == "-0.0":
        s = "0.0"
    return s


def _arr_1d(v, sig: int = SIG) -> str:
    return ", ".join(_lit(x, sig) for x in np.asarray(v).ravel())


def _arr_2d(v, sig: int = SIG) -> str:
    """二维常量数组：每个子数组必须自带花括号。

    少了花括号编译器只会给 -Wmissing-braces 警告，但一旦行数与列数不匹配
    就会变成 "excess elements" 硬错误 —— 这里显式加括号，别依赖编译器宽容。
    """
    rows = ["{" + ", ".join(_lit(x, sig) for x in row) + "}"
            for row in np.asarray(v)]
    return ",\n    ".join(rows)


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class ExportConfig:
    name: str = "beamformer"
    mics: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    weights: np.ndarray | None = None        # (F, M) complex
    freqs: np.ndarray | None = None          # (F,)
    n_fft: int = 512
    hop: int | None = None
    fs: int = 16000
    win_name: str = "blackmanharris"
    speed_of_sound: float = C_DEFAULT
    steer_dir: np.ndarray | None = None
    method: str = "delay_sum"
    use_eigen: bool = True
    namespace: str = "mas"
    extra_notes: str = ""

    def __post_init__(self) -> None:
        self.mics = np.asarray(self.mics, dtype=np.float64).reshape(-1, 3)
        if self.weights is not None:
            self.weights = np.atleast_2d(np.asarray(self.weights, dtype=np.complex128))
            if self.freqs is None:
                raise ValueError("给了 weights 就必须给 freqs")
            self.freqs = np.atleast_1d(np.asarray(self.freqs, float))
            if self.freqs.size != self.weights.shape[0]:
                raise ValueError("freqs 与 weights 行数不一致")
        if self.hop is None:
            self.hop = self.n_fft // 2
        if self.steer_dir is None:
            self.steer_dir = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# 代码模板
# --------------------------------------------------------------------------
def _hpp_config(cfg: ExportConfig) -> str:
    M = cfg.mics.shape[0]
    F = 0 if cfg.weights is None else cfg.weights.shape[0]
    win = dsp.window_table(cfg.win_name, cfg.n_fft, sig=12)
    if cfg.weights is not None:
        w_re = ",\n    ".join("{" + ", ".join(_lit(c.real) for c in row) + "}"
                               for row in cfg.weights)
        w_im = ",\n    ".join("{" + ", ".join(_lit(c.imag) for c in row) + "}"
                               for row in cfg.weights)
    else:
        w_re = w_im = ""
    frows = "" if cfg.freqs is None else _arr_1d(cfg.freqs)
    return f"""// 由 MicArray Studio 自动生成 —— 请勿手工编辑。
// 阵列几何 / 设计参数 / 导出常量。数值与 Python 侧量化到同一精度，
// 因此两侧的差异只可能来自浮点求和顺序（量级 1e-15）。
#pragma once

#include <array>
#include <cstddef>

namespace {cfg.namespace} {{

// ---- 常量 ----------------------------------------------------------------
inline constexpr int   kNumMics      = {M};
inline constexpr int   kNumFreqs     = {F};
inline constexpr int   kNfft         = {cfg.n_fft};
inline constexpr int   kHop          = {cfg.hop};
inline constexpr int   kSampleRate   = {cfg.fs};
inline constexpr double kSpeedOfSound = {_lit(cfg.speed_of_sound)};
inline constexpr const char* kMethod  = "{cfg.method}";
inline constexpr const char* kWindowName = "{cfg.win_name}";

// ---- 麦克风坐标 (x, y, z)，单位米 ----------------------------------------
// 阵列坐标系：阵元在 xy 平面，boresight = +z（见 docs/CONTRACT.md 第 1 节）
inline constexpr double kMics[kNumMics][3] = {{
    {_arr_2d(cfg.mics)}
}};

// ---- 波束成形权重 (F, M)，实部虚部分开存，避免依赖 std::complex 的 ABI ----
inline constexpr double kWeightsRe[kNumFreqs][kNumMics] = {{
    {w_re}
}};
// 与 kWeightsRe 同形的虚部。实虚分开存是为了不依赖 std::complex 的
// 内存布局假设（不同工具链/ABI 下 constexpr complex 数组行为不一致）。
inline constexpr double kWeightsIm[kNumFreqs][kNumMics] = {{
    {w_im}
}};

// ---- 设计频点 (Hz) -------------------------------------------------------
inline constexpr double kDesignFreqs[kNumFreqs] = {{
    {frows if F else "0.0"}
}};

// ---- STFT 窗表（periodic，长度 kNfft）------------------------------------
inline constexpr double kWindow[kNfft] = {{
    {_arr_1d(win, 12)}
}};

// ---- 设计指向（阵列坐标系单位向量）--------------------------------------
inline constexpr double kSteerDir[3] = {{
    {_arr_1d(cfg.steer_dir)}
}};

}}  // namespace {cfg.namespace}
"""


def _hpp_dsp(cfg: ExportConfig) -> str:
    ns = cfg.namespace
    return f"""// 由 MicArray Studio 自动生成 —— 请勿手工编辑。
// 基础 DSP：迭代 radix-2 FFT、线性卷积、STFT/ISTFT。
// 与 Python 侧约定一致：单边谱、不做幅度补偿、窗表来自 mas_array_config.hpp。
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace {ns} {{

using cplx = std::complex<double>;

/// 原地基 2 FFT（Cooley-Tukey，迭代蝶形 + 位反转置换）。
/// n 必须是 2 的幂。inv=true 时做逆变换（含 1/n 缩放）。
void fft(std::vector<cplx>& a, bool inv);

/// 实数信号的单边 FFT -> (n/2+1) 个复数。n 必须是 2 的幂。
std::vector<cplx> rfft(const std::vector<double>& x, std::size_t n);

/// rfft 的逆变换，输出 n 点实数。
std::vector<double> irfft(const std::vector<cplx>& X, std::size_t n);

/// 线性卷积并截断到 n 点。
/// 注意：FFT 卷积必须把 nfft 取到 >= x.size()+h.size()-1，
/// 否则循环卷积会绕回污染信号开头（这个坑踩过，故在此写死断言）。
std::vector<double> convolve(const std::vector<double>& x,
                             const std::vector<double>& h, std::size_t n);

/// STFT：返回 (K, n/2+1) 的复数矩阵，按行优先展平。
/// 调用方提供窗表（长度需等于 n_fft）。
std::vector<cplx> stft(const std::vector<double>& x, int n_fft, int hop,
                       const std::vector<double>& win);

/// ISTFT：重叠相加并按窗平方和归一化，输出长度可截断。
std::vector<double> istft(const std::vector<cplx>& Xf, int n_fft, int hop,
                          const std::vector<double>& win,
                          std::size_t out_len);

/// 帧数，与 stft 内部一致。
std::size_t frame_count(std::size_t n_samples, int n_fft, int hop);

}}  // namespace {ns}
"""


def _cpp_dsp(cfg: ExportConfig) -> str:
    ns = cfg.namespace
    return f"""// 由 MicArray Studio 自动生成 —— 请勿手工编辑。
#include "mas_dsp.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {ns} {{

namespace {{
// 位反转置换：把下标 i 的比特镜像到 log2(n) 位
void bit_reverse(std::vector<cplx>& a) {{
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {{
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }}
}}
}}  // namespace

void fft(std::vector<cplx>& a, bool inv) {{
    const std::size_t n = a.size();
    if (n <= 1) return;
    if ((n & (n - 1)) != 0) throw std::invalid_argument("fft size must be power of 2");

    bit_reverse(a);

    for (std::size_t len = 2; len <= n; len <<= 1) {{
        // 旋转因子：w = exp(-2*pi*i/len)（逆变换取共轭）
        const double ang = -2.0 * M_PI / static_cast<double>(len);
        const cplx wl(std::cos(ang), inv ? -std::sin(ang) : std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {{
            cplx w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {{
                const cplx u = a[i + k];
                const cplx v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }}
        }}
    }}

    if (inv) {{
        const double s = 1.0 / static_cast<double>(n);
        for (auto& v : a) v *= s;
    }}
}}

std::vector<cplx> rfft(const std::vector<double>& x, std::size_t n) {{
    std::vector<cplx> a(n, cplx(0.0, 0.0));
    const std::size_t m = std::min(n, x.size());
    for (std::size_t i = 0; i < m; ++i) a[i] = cplx(x[i], 0.0);
    fft(a, false);
    a.resize(n / 2 + 1);
    return a;
}}

std::vector<double> irfft(const std::vector<cplx>& X, std::size_t n) {{
    std::vector<cplx> a(n, cplx(0.0, 0.0));
    const std::size_t f = n / 2 + 1;
    for (std::size_t i = 0; i < std::min(f, X.size()); ++i) a[i] = X[i];
    // 共轭对称补全
    for (std::size_t i = 1; i < f - (n % 2 == 0 ? 1 : 0); ++i) a[n - i] = std::conj(a[i]);
    fft(a, true);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i].real();
    return out;
}}

std::vector<double> convolve(const std::vector<double>& x,
                             const std::vector<double>& h, std::size_t n) {{
    if (x.empty() || h.empty()) return std::vector<double>(n, 0.0);
    const std::size_t need = x.size() + h.size() - 1;
    std::size_t nfft = 1;
    while (nfft < need) nfft <<= 1;
    assert(nfft >= need && "FFT 长度不足会导致循环卷积绕回");

    std::vector<cplx> A = rfft(x, nfft);
    std::vector<cplx> B = rfft(h, nfft);
    for (std::size_t i = 0; i < A.size(); ++i) A[i] *= B[i];
    std::vector<double> y = irfft(A, nfft);
    y.resize(n, 0.0);
    return y;
}}

std::size_t frame_count(std::size_t n_samples, int n_fft, int hop) {{
    if (n_samples == 0) return 0;
    if (n_samples <= static_cast<std::size_t>(n_fft)) return 1;
    return 1 + (n_samples - n_fft + hop - 1) / hop;
}}

std::vector<cplx> stft(const std::vector<double>& x, int n_fft, int hop,
                       const std::vector<double>& win) {{
    assert(static_cast<int>(win.size()) == n_fft);
    const std::size_t K = frame_count(x.size(), n_fft, hop);
    const std::size_t F = static_cast<std::size_t>(n_fft) / 2 + 1;
    std::vector<cplx> out(K * F, cplx(0.0, 0.0));

    std::vector<double> frame(n_fft);
    for (std::size_t k = 0; k < K; ++k) {{
        const std::size_t s = k * hop;
        for (int i = 0; i < n_fft; ++i) {{
            const std::size_t idx = s + i;
            frame[i] = (idx < x.size() ? x[idx] : 0.0) * win[i];
        }}
        std::vector<cplx> X = rfft(frame, n_fft);
        for (std::size_t f = 0; f < F; ++f) out[k * F + f] = X[f];
    }}
    return out;
}}

std::vector<double> istft(const std::vector<cplx>& Xf, int n_fft, int hop,
                          const std::vector<double>& win, std::size_t out_len) {{
    const std::size_t F = static_cast<std::size_t>(n_fft) / 2 + 1;
    if (Xf.empty()) return std::vector<double>(out_len, 0.0);
    const std::size_t K = Xf.size() / F;
    std::vector<double> y(out_len, 0.0), wsum(out_len, 0.0);
    std::vector<cplx> spec(F);

    for (std::size_t k = 0; k < K; ++k) {{
        for (std::size_t f = 0; f < F; ++f) spec[f] = Xf[k * F + f];
        std::vector<double> fr = irfft(spec, n_fft);
        const std::size_t s = k * hop;
        for (int i = 0; i < n_fft; ++i) {{
            const std::size_t idx = s + i;
            if (idx >= out_len) break;
            y[idx] += fr[i] * win[i];
            wsum[idx] += win[i] * win[i];
        }}
    }}
    // 归一化阈值与 Python 侧 engine.dsp.istft 严格保持一致（0.05*max）。
    // BH 窗端点 wsum 比稳态小 3e7 倍，用极小阈值会让两端被放大到 1e4 倍；
    // 阈值以下直接置零。两侧必须同阈值，否则 parity 会在端点处失配。
    double wmax = 0.0;
    for (std::size_t i = 0; i < out_len; ++i) wmax = std::max(wmax, wsum[i]);
    const double thr = 0.05 * wmax;
    for (std::size_t i = 0; i < out_len; ++i) {{
        if (wsum[i] > thr) y[i] /= wsum[i];
        else y[i] = 0.0;
    }}
    return y;
}}

}}  // namespace {ns}
"""


def _hpp_beamform(cfg: ExportConfig) -> str:
    ns = cfg.namespace
    return f"""// 由 MicArray Studio 自动生成 —— 请勿手工编辑。
// 波束成形：频域加权求和（设计权重来自 kWeightsRe/Im）。
#pragma once

#include <complex>
#include <vector>

#include "mas_array_config.hpp"   // kSpeedOfSound 等默认参数要用到
#include "mas_dsp.hpp"

namespace {ns} {{

/// 用导出权重对多通道信号做波束成形。
/// X: 长度 M*N，通道优先（X[m*N + n]）。
/// 返回长度 N 的单通道输出。
std::vector<double> beamform(const std::vector<double>& X, std::size_t n_samples,
                             int n_fft, int hop);

/// 延迟求和（不需预存权重，按 kMics + kSteerDir + 频率实时算相位）。
/// 用于和"导出权重"路径互相校验：两者在 DS 下应完全一致。
std::vector<double> delay_and_sum(const std::vector<double>& X,
                                  std::size_t n_samples, int n_fft, int hop);

/// 白噪增益 (F,)：WNG = |w^H a|^2 / (w^H w)。
/// 这是鲁棒性的第一指标，低频低于 0 dB 基本不可用。
std::vector<double> white_noise_gain(int n_fft, double c = kSpeedOfSound);

/// 波束图功率 (G,)，按最大值归一。dirs 是 G*3 的单位方向向量。
std::vector<double> beampattern(const std::vector<double>& dirs, double freq,
                                double c = kSpeedOfSound);

}}  // namespace {ns}
"""


def _cpp_beamform(cfg: ExportConfig) -> str:
    ns = cfg.namespace
    return f"""// 由 MicArray Studio 自动生成 —— 请勿手工编辑。
#include "mas_beamform.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>

#include "mas_array_config.hpp"

namespace {ns} {{

namespace {{
// 把设计频点的权重插值到目标频率轴（实部虚部分别线性插值），
// 与 Python 侧 beamforming.interpolate_weights 完全同构。
void interp_weights(double f, cplx* out) {{
    const double* fq = kDesignFreqs;
    if (kNumFreqs == 1 || f <= fq[0]) {{
        for (int m = 0; m < kNumMics; ++m)
            out[m] = cplx(kWeightsRe[0][m], kWeightsIm[0][m]);
        return;
    }}
    if (f >= fq[kNumFreqs - 1]) {{
        const int k = kNumFreqs - 1;
        for (int m = 0; m < kNumMics; ++m)
            out[m] = cplx(kWeightsRe[k][m], kWeightsIm[k][m]);
        return;
    }}
    int k = 0;
    while (k + 1 < kNumFreqs && fq[k + 1] < f) ++k;
    const double f0 = fq[k], f1 = fq[k + 1];
    const double t = (f1 > f0) ? (f - f0) / (f1 - f0) : 0.0;
    for (int m = 0; m < kNumMics; ++m) {{
        const double re = kWeightsRe[k][m] + t * (kWeightsRe[k + 1][m] - kWeightsRe[k][m]);
        const double im = kWeightsIm[k][m] + t * (kWeightsIm[k + 1][m] - kWeightsIm[k][m]);
        out[m] = cplx(re, im);
    }}
}}

std::vector<double> win_vector() {{
    return std::vector<double>(kWindow, kWindow + kNfft);
}}
}}  // namespace

std::vector<double> beamform(const std::vector<double>& X, std::size_t n_samples,
                             int n_fft, int hop) {{
    if (X.size() < n_samples * kNumMics)
        throw std::invalid_argument("X 长度不足 M*N");

    const std::size_t F = static_cast<std::size_t>(n_fft) / 2 + 1;
    const std::size_t K = frame_count(n_samples, n_fft, hop);
    const std::vector<double> win = win_vector();

    // 各通道 STFT；按帧累加加权和，避免一次性存 M*K*F 造成内存峰值
    std::vector<cplx> acc(K * F, cplx(0.0, 0.0));
    std::vector<double> ch(n_samples);
    std::vector<cplx> w(kNumMics);

    for (int m = 0; m < kNumMics; ++m) {{
        for (std::size_t n = 0; n < n_samples; ++n) ch[n] = X[m * n_samples + n];
        const std::vector<cplx> Xf = stft(ch, n_fft, hop, win);
        for (std::size_t k = 0; k < K; ++k) {{
            for (std::size_t f = 0; f < F; ++f) {{
                const double freq = static_cast<double>(f) * kSampleRate /
                                    static_cast<double>(n_fft);
                interp_weights(freq, w.data());
                acc[k * F + f] += std::conj(w[m]) * Xf[k * F + f];
            }}
        }}
    }}
    return istft(acc, n_fft, hop, win, n_samples);
}}

std::vector<double> delay_and_sum(const std::vector<double>& X,
                                  std::size_t n_samples, int n_fft, int hop) {{
    if (X.size() < n_samples * kNumMics)
        throw std::invalid_argument("X 长度不足 M*N");

    const std::size_t F = static_cast<std::size_t>(n_fft) / 2 + 1;
    const std::size_t K = frame_count(n_samples, n_fft, hop);
    const std::vector<double> win = win_vector();

    // 解析延迟求和权重：w_m(f) = a_m(f) / M,
    // a_m(f) = exp(-j*2*pi*f*dot(p_m, u0)/c)
    std::vector<cplx> acc(K * F, cplx(0.0, 0.0));
    std::vector<double> ch(n_samples);

    for (int m = 0; m < kNumMics; ++m) {{
        for (std::size_t n = 0; n < n_samples; ++n) ch[n] = X[m * n_samples + n];
        const std::vector<cplx> Xf = stft(ch, n_fft, hop, win);
        const double proj = kMics[m][0] * kSteerDir[0] + kMics[m][1] * kSteerDir[1] +
                            kMics[m][2] * kSteerDir[2];
        for (std::size_t k = 0; k < K; ++k) {{
            for (std::size_t f = 0; f < F; ++f) {{
                const double freq = static_cast<double>(f) * kSampleRate /
                                    static_cast<double>(n_fft);
                const double ph = -2.0 * M_PI * freq * proj / kSpeedOfSound;
                const cplx a(std::cos(ph), std::sin(ph));
                acc[k * F + f] += std::conj(a / static_cast<double>(kNumMics)) *
                                  Xf[k * F + f];
            }}
        }}
    }}
    return istft(acc, n_fft, hop, win, n_samples);
}}

std::vector<double> white_noise_gain(int n_fft, double c) {{
    const std::size_t F = static_cast<std::size_t>(n_fft) / 2 + 1;
    std::vector<double> out(F, 0.0);
    for (std::size_t f = 0; f < F; ++f) {{
        const double freq = static_cast<double>(f) * kSampleRate /
                            static_cast<double>(n_fft);
        cplx w[kNumMics];
        interp_weights(freq, w);
        // |w^H a|^2 / (w^H w)，a 由 kSteerDir 解析给出
        cplx resp(0.0, 0.0);
        double norm = 0.0;
        for (int m = 0; m < kNumMics; ++m) {{
            const double proj = kMics[m][0] * kSteerDir[0] +
                                kMics[m][1] * kSteerDir[1] +
                                kMics[m][2] * kSteerDir[2];
            const double ph = -2.0 * M_PI * freq * proj / c;
            resp += std::conj(w[m]) * cplx(std::cos(ph), std::sin(ph));
            norm += std::norm(w[m]);
        }}
        out[f] = (norm > 1e-300) ? std::norm(resp) / norm : 0.0;
    }}
    return out;
}}

std::vector<double> beampattern(const std::vector<double>& dirs, double freq,
                                double c) {{
    const std::size_t G = dirs.size() / 3;
    std::vector<double> out(G, 0.0);
    cplx w[kNumMics];
    interp_weights(freq, w);
    for (std::size_t g = 0; g < G; ++g) {{
        cplx resp(0.0, 0.0);
        for (int m = 0; m < kNumMics; ++m) {{
            const double proj = kMics[m][0] * dirs[3 * g + 0] +
                                kMics[m][1] * dirs[3 * g + 1] +
                                kMics[m][2] * dirs[3 * g + 2];
            const double ph = -2.0 * M_PI * freq * proj / c;
            resp += std::conj(w[m]) * cplx(std::cos(ph), std::sin(ph));
        }}
        out[g] = std::norm(resp);
    }}
    const double mx = *std::max_element(out.begin(), out.end());
    if (mx > 1e-300)
        for (auto& v : out) v /= mx;
    return out;
}}

}}  // namespace {ns}
"""


def _cpp_test(cfg: ExportConfig) -> str:
    ns = cfg.namespace
    mid = max(0, (0 if cfg.freqs is None else cfg.freqs.size) // 2)
    return f"""// 由 MicArray Studio 自动生成 —— 请勿手工编辑。
// 读入导出的测试向量，比对 Python 与 C++ 的逐点相对误差。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mas_array_config.hpp"
#include "mas_beamform.hpp"

namespace {{

std::vector<double> read_txt(const std::string& path) {{
    std::ifstream in(path);
    if (!in) {{ std::cerr << "无法打开 " << path << "\\n"; std::exit(2); }}
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    return v;
}}

double max_rel_err(const std::vector<double>& a, const std::vector<double>& b,
                   double* rms_out) {{
    if (a.size() != b.size()) {{
        std::cerr << "长度不一致: " << a.size() << " vs " << b.size() << "\\n";
        std::exit(2);
    }}
    double scale = 0.0, mx = 0.0, se = 0.0;
    for (double v : b) scale = std::max(scale, std::fabs(v));
    if (scale <= 0) scale = 1.0;
    for (std::size_t i = 0; i < a.size(); ++i) {{
        const double d = std::fabs(a[i] - b[i]);
        mx = std::max(mx, d);
        se += d * d;
    }}
    if (rms_out) *rms_out = std::sqrt(se / std::max<std::size_t>(a.size(), 1)) / scale;
    return mx / scale;
}}

}}  // namespace

int main(int argc, char** argv) {{
    const std::string dir = (argc > 1) ? argv[1] : "vectors";
    const double tol = (argc > 2) ? std::stod(argv[2]) : 1e-6;

    std::cout << "MicArray Studio C++ 一致性验证\\n";
    std::cout << "  阵列元数 = " << {ns}::kNumMics
              << ", n_fft = " << {ns}::kNfft
              << ", 设计频点数 = " << {ns}::kNumFreqs << "\\n";

    int fails = 0;

    // --- 1. 波束成形输出比对 ---
    {{
        const std::vector<double> X = read_txt(dir + "/input_x.txt");
        const std::vector<double> ref = read_txt(dir + "/expected_y.txt");
        const std::size_t n = ref.size();
        const std::vector<double> y = {ns}::beamform(X, n, {ns}::kNfft, {ns}::kHop);
        double rms = 0.0;
        const double e = max_rel_err(y, ref, &rms);
        std::cout << "  [beamform]      最大相对误差 " << e
                  << "  RMS " << rms << (e <= tol ? "  PASS" : "  FAIL") << "\\n";
        if (e > tol) ++fails;
    }}

    // --- 2. 解析延迟求和与导出权重一致性（仅 DS 权重时成立）---
    {{
        const std::vector<double> X = read_txt(dir + "/input_x.txt");
        const std::vector<double> ref_ds = read_txt(dir + "/expected_ds.txt");
        const std::size_t n = ref_ds.size();
        const std::vector<double> y = {ns}::delay_and_sum(X, n, {ns}::kNfft, {ns}::kHop);
        double rms = 0.0;
        const double e = max_rel_err(y, ref_ds, &rms);
        std::cout << "  [delay_and_sum] 最大相对误差 " << e
                  << "  RMS " << rms << (e <= tol ? "  PASS" : "  FAIL") << "\\n";
        if (e > tol) ++fails;
    }}

    // --- 3. 白噪增益比对 ---
    {{
        const std::vector<double> ref = read_txt(dir + "/expected_wng.txt");
        const std::vector<double> wng = {ns}::white_noise_gain({ns}::kNfft);
        std::size_t n = std::min(ref.size(), wng.size());
        double mx = 0.0;
        for (std::size_t i = 0; i < n; ++i) {{
            const double d = std::fabs(wng[i] - ref[i]) /
                             std::max(std::fabs(ref[i]), 1e-12);
            mx = std::max(mx, d);
        }}
        std::cout << "  [white_noise_gain] 最大相对误差 " << mx
                  << (mx <= 1e-9 ? "  PASS" : "  FAIL") << "\\n";
        if (mx > 1e-9) ++fails;
    }}

    // --- 4. 波束图比对 ---
    {{
        const std::vector<double> dirs = read_txt(dir + "/dirs.txt");
        const std::vector<double> ref = read_txt(dir + "/expected_pattern.txt");
        const double freq = {ns}::kDesignFreqs[{mid}];  // 与生成向量时取的中间频点一致
        const std::vector<double> pat = {ns}::beampattern(dirs, freq);
        double rms = 0.0;
        const double e = max_rel_err(pat, ref, &rms);
        std::cout << "  [beampattern]    最大相对误差 " << e
                  << "  RMS " << rms << (e <= 1e-9 ? "  PASS" : "  FAIL") << "\\n";
        if (e > 1e-9) ++fails;
    }}

    std::cout << (fails == 0 ? "全部通过\\n" : "存在失败项\\n");
    return fails == 0 ? 0 : 1;
}}
"""


def _cmake(cfg: ExportConfig) -> str:
    eigen_opt = "ON" if cfg.use_eigen else "OFF"
    return f"""cmake_minimum_required(VERSION 3.16)
project(mas_{cfg.name} CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2 -Wall -Wextra)

# 纯 C++17 实现，默认不依赖 Eigen。
# 保留该开关是为了将来接入 MVDR 这类需要解线性方程组的算法时可以直接用
# Eigen 的 Cholesky/LDLT，而不必自己写复数线性代数。
option(MAS_USE_EIGEN "使用 Eigen 做线性代数（当前导出不需要）" {eigen_opt})

add_library(mas_core STATIC
  src/mas_dsp.cpp
  src/mas_beamform.cpp
)
target_include_directories(mas_core PUBLIC include)

if(MAS_USE_EIGEN)
  find_package(Eigen3 QUIET)
  if(Eigen3_FOUND)
    target_link_libraries(mas_core PUBLIC Eigen3::Eigen)
    target_compile_definitions(mas_core PUBLIC MAS_USE_EIGEN=1)
    message(STATUS "Eigen3 已启用")
  else()
    message(STATUS "未找到 Eigen3，回退到纯 C++ 实现")
  endif()
endif()

add_executable(test_parity tests/test_parity.cpp)
target_link_libraries(test_parity PRIVATE mas_core)

enable_testing()
add_test(NAME parity COMMAND test_parity
         ${{CMAKE_CURRENT_SOURCE_DIR}}/tests/vectors 1e-6)
"""


def _readme(cfg: ExportConfig) -> str:
    M = cfg.mics.shape[0]
    F = 0 if cfg.weights is None else cfg.weights.shape[0]
    return f"""# {cfg.name} —— MicArray Studio 导出的 C++17 算法内核

自包含、无第三方依赖（Eigen 可选）。全部数值常量与 Python 侧量化到同一精度，
因此两侧差异只来自浮点求和顺序。

## 内容

| 文件 | 说明 |
|---|---|
| `include/mas_array_config.hpp` | 阵列几何（{M} 元）、设计权重（{F} 个频点）、窗表、常量 |
| `include/mas_dsp.hpp` / `src/mas_dsp.cpp` | 迭代 radix-2 FFT、线性卷积、STFT/ISTFT |
| `include/mas_beamform.hpp` / `src/mas_beamform.cpp` | 频域加权波束成形、解析延迟求和、WNG、波束图 |
| `tests/test_parity.cpp` | 读取测试向量并比对最大相对误差 |
| `tests/vectors/*.txt` | 输入与期望输出（纯文本，避免实时路径引入 JSON 依赖） |

## 坐标系约定

阵元位于 **xy 平面**，`boresight = +z`。
设计指向 `kSteerDir` 是阵列坐标系中的单位向量。
完整约定见项目 `docs/CONTRACT.md` 第 1–3 节。

## 编译与验证

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_parity tests/vectors 1e-6
```

`test_parity` 会对四件事做逐点比对：

1. `beamform()` 输出 vs Python `beamforming.process()`
2. `delay_and_sum()` 解析实现 vs Python 解析延迟求和
3. `white_noise_gain()` vs Python `beamforming.wng_db()`（容差 1e-9）
4. `beampattern()` vs Python `beamforming.evaluate_beampattern()`（容差 1e-9）

## 接口

```cpp
#include "mas_beamform.hpp"

// X 为 M*N 的通道优先缓冲（X[m*N + n]）
std::vector<double> y = {cfg.namespace}::beamform(X, n_samples, kNfft, kHop);
```

## 设计方法

`{cfg.method}`{("；" + cfg.extra_notes) if cfg.extra_notes else ""}

## 注意

- 权重已固化在 `kWeightsRe/kWeightsIm` 中。若要在目标平台上重新设计
  （例如用实测协方差做 MVDR），需要把 `beamforming.lcmv_weights` 一并移植 ——
  当前导出只含**运行时应用**，不含自适应设计。
- 采样率按 `kSampleRate` 假定；若实际采样率不同，权重对应的物理频率会偏移，
  请用相同 `fs` 重新导出。
"""


# --------------------------------------------------------------------------
# 测试向量
# --------------------------------------------------------------------------
def _write_vectors(cfg: ExportConfig, outdir: pathlib.Path,
                   X: np.ndarray, y: np.ndarray, y_ds: np.ndarray,
                   wng: np.ndarray, dirs: np.ndarray, pattern: np.ndarray,
                   fmt: str = "%.17g") -> None:
    vdir = outdir / "tests" / "vectors"
    vdir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, arr) -> None:
        np.savetxt(vdir / name, np.asarray(arr).ravel(), fmt=fmt)

    dump("input_x.txt", X)
    dump("expected_y.txt", y)
    dump("expected_ds.txt", y_ds)
    dump("expected_wng.txt", wng)
    dump("dirs.txt", dirs)
    dump("expected_pattern.txt", pattern)


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
@dataclass
class ExportResult:
    outdir: pathlib.Path
    files: list[str]
    parity: dict | None = None
    build_log: str = ""

    def summary(self) -> str:
        lines = [f"导出目录: {self.outdir}", f"文件数: {len(self.files)}"]
        if self.parity:
            lines.append("一致性验证:")
            for k, v in self.parity.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def generate(cfg: ExportConfig, outdir: pathlib.Path) -> ExportResult:
    """生成 C++ 工程（不含编译）。"""
    outdir = pathlib.Path(outdir)
    (outdir / "include").mkdir(parents=True, exist_ok=True)
    (outdir / "src").mkdir(parents=True, exist_ok=True)
    (outdir / "tests").mkdir(parents=True, exist_ok=True)

    files = {
        "include/mas_array_config.hpp": _hpp_config(cfg),
        "include/mas_dsp.hpp": _hpp_dsp(cfg),
        "src/mas_dsp.cpp": _cpp_dsp(cfg),
        "include/mas_beamform.hpp": _hpp_beamform(cfg),
        "src/mas_beamform.cpp": _cpp_beamform(cfg),
        "tests/test_parity.cpp": _cpp_test(cfg),
        "CMakeLists.txt": _cmake(cfg),
        "README.md": _readme(cfg),
    }
    for rel, content in files.items():
        (outdir / rel).write_text(content, encoding="utf-8")
    return ExportResult(outdir=outdir, files=sorted(files.keys()))


def make_vectors(cfg: ExportConfig, outdir: pathlib.Path, n_samples: int = 512,
                 seed: int = 3, fs: int | None = None) -> dict:
    """造测试向量：用 Python 侧算出期望输出，供 C++ 比对。

    输入信号刻意用宽频噪声（覆盖所有频点），这样权重插值、STFT 边界、
    卷积等所有路径都会被检验到。
    """
    fs = fs or cfg.fs
    rng = np.random.default_rng(seed)
    M = cfg.mics.shape[0]
    X = rng.standard_normal((M, n_samples))
    win = dsp.window_table(cfg.win_name, cfg.n_fft, sig=12)

    freqs_axis = np.fft.rfftfreq(cfg.n_fft, 1.0 / fs)

    # 1. 用导出权重做波束成形（与 C++ beamform() 对应）
    res = bf.BeamformerResult(freqs=cfg.freqs, weights=cfg.weights, mics=cfg.mics,
                              method=cfg.method, n_fft=cfg.n_fft)
    Xf, _ = dsp.stft(X, fs, n_fft=cfg.n_fft, hop=cfg.hop, win_table=win)
    W = bf.interpolate_weights(res, freqs_axis)
    Yf = np.einsum("fm,mkf->kf", W.conj(), Xf)
    y = dsp.istft(Yf, fs, n_fft=cfg.n_fft, hop=cfg.hop, win_table=win,
                  length=n_samples)

    # 2. 解析延迟求和（与 C++ delay_and_sum() 对应）
    A = np.exp(-2j * np.pi * freqs_axis[:, None]
               * ac.delay_from_dir(cfg.mics, cfg.steer_dir, cfg.speed_of_sound)[None, :])
    Yds = np.einsum("fm,mkf->kf", (A / M).conj(), Xf)
    y_ds = dsp.istft(Yds, fs, n_fft=cfg.n_fft, hop=cfg.hop, win_table=win,
                     length=n_samples)

    # 3. 白噪增益（与 C++ white_noise_gain() 对应）
    Afull = np.exp(-2j * np.pi * freqs_axis[:, None]
                   * ac.delay_from_dir(cfg.mics, cfg.steer_dir,
                                       cfg.speed_of_sound)[None, :])
    wng = np.abs(np.einsum("fm,fm->f", W.conj(), Afull.conj())) ** 2 \
        / np.maximum(np.einsum("fm,fm->f", W.conj(), W).real, 1e-300)

    # 4. 波束图（取设计频段的中间频点）
    from .geometry import direction_grid
    dirs, shape, az_ax, el_ax = direction_grid(az_bins=181, el_bins=91)
    f_mid = float(cfg.freqs[cfg.freqs.size // 2]) if cfg.freqs is not None else 1000.0
    k = int(np.argmin(np.abs(freqs_axis - f_mid)))
    A2 = np.exp(-2j * np.pi * f_mid * ac.delays_grid(cfg.mics, dirs,
                                                     cfg.speed_of_sound))
    pat = np.abs(np.einsum("m,gm->g", W[k].conj(), A2.conj())) ** 2
    pat = pat / max(pat.max(), 1e-300)

    _write_vectors(cfg, outdir, X, y, y_ds, wng, dirs, pat)
    return {
        "n_samples": n_samples, "n_mics": M, "n_freqs": int(cfg.freqs.size),
        "pattern_freq": f_mid, "n_dirs": int(dirs.shape[0]),
        "y_rms": float(np.sqrt(np.mean(y ** 2))),
    }


# --------------------------------------------------------------------------
# 编译 + 验证
# --------------------------------------------------------------------------
def build_and_verify(outdir: pathlib.Path, tol: float = 1e-6,
                     timeout: int = 300) -> dict:
    """真的调用 cmake 编译并运行 parity 测试。

    找不到 cmake/编译器时返回 skipped，而不是假装成功 —— 没编译过就不能
    声称"导出可用"。
    """
    outdir = pathlib.Path(outdir)
    cmake = shutil.which("cmake")
    if cmake is None:
        return {"status": "skipped", "reason": "未找到 cmake"}

    build = outdir / "build"
    build.mkdir(exist_ok=True)
    log = []

    def run(cmd, cwd):
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout)
        log.append(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
        return p

    p = run([cmake, "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], outdir)
    if p.returncode != 0:
        return {"status": "cmake_failed", "log": "\n".join(log)[-4000:]}

    p = run([cmake, "--build", "build", "-j", "4"], outdir)
    if p.returncode != 0:
        return {"status": "build_failed", "log": "\n".join(log)[-4000:]}

    exe = build / "test_parity"
    if not exe.exists():
        return {"status": "no_binary", "log": "\n".join(log)[-4000:]}

    p = run([str(exe), str(outdir / "tests" / "vectors"), str(tol)], outdir)
    out = p.stdout
    parsed = {}
    for line in out.splitlines():
        if "最大相对误差" in line:
            name = line.split("[")[1].split("]")[0]
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok == "误差":
                    parsed[name] = float(parts[i + 1])
                    break
    return {"status": "pass" if p.returncode == 0 else "fail",
            "errors": parsed, "stdout": out, "log": "\n".join(log)[-2000:]}


def export_and_verify(geom, freqs, steer=(0.0, 90.0), method: str = "delay_sum",
                      name: str = "beamformer", outdir=None, fs: int = 16000,
                      n_fft: int = 512, tol: float = 1e-6,
                      n_samples: int = 512, verify: bool = True,
                      **design_kw) -> ExportResult:
    """一行完成：设计 -> 生成 -> 造向量 -> 编译 -> 比对。

    这是导出层的验收入口。**只有 verify 通过才算导出成功。**
    """
    root = pathlib.Path(outdir) if outdir else \
        pathlib.Path(__file__).resolve().parent.parent / "export"
    outdir = pathlib.Path(root) / name

    res = bf.design(geom, freqs, steer=steer, method=method, n_fft=n_fft, **design_kw)
    cfg = ExportConfig(name=name, mics=geom.mics, weights=res.weights,
                       freqs=res.freqs, n_fft=n_fft, fs=fs,
                       steer_dir=res.steer_dir, method=res.method,
                       extra_notes=f"WNG 最小值 {res.metrics['wng_min_db']} dB，"
                                   f"DI 最大值 {res.metrics['di_max_db']} dB")
    er = generate(cfg, outdir)
    make_vectors(cfg, outdir, n_samples=n_samples, fs=fs)
    if verify:
        er.parity = build_and_verify(outdir, tol=tol)
        er.build_log = er.parity.get("log", "")
    return er


__all__ = ["ExportConfig", "ExportResult", "generate", "make_vectors",
           "build_and_verify", "export_and_verify"]
