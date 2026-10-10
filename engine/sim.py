"""声场仿真：自由场 / 镜像源法混响 / 失配 / 噪声。

数据流与电平标定见 docs/CONTRACT.md 第 4 节。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import acoustics as ac
from . import signals as sg

C_DEFAULT = ac.C_DEFAULT


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class RoomSpec:
    """矩形房间（鞋盒模型）。

    bounds : [xmin,xmax,ymin,ymax,zmin,zmax]，绝对坐标。
             阵列中心位于 (0,0,0)（见 CONTRACT 第 1 节），
             因此典型房间可写为 [-3,3,-2,2,0,2.6]（地面 z=0，阵列贴地/吊顶另设）。
    absorption : (6,) 六个面的吸声系数 [x-,x+,y-,y+,z-,z+]
    max_order  : 镜像源最大反射阶数
    """
    bounds: tuple = (-3.0, 3.0, -2.0, 2.0, 0.0, 2.7)
    absorption: tuple = (0.35, 0.35, 0.35, 0.35, 0.25, 0.35)
    max_order: int = 2

    def size(self) -> np.ndarray:
        b = np.asarray(self.bounds, float)
        return b[1::2] - b[0::2]

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        b = np.asarray(self.bounds, float)
        return bool(np.all(p >= b[0::2] + margin) and np.all(p <= b[1::2] - margin))


@dataclass
class SourceSpec:
    """声源位置，两种指定方式（**都是相对阵列**）：

    - direction(az,el) + distance : 沿阵列坐标系中该方向的球面取点
    - position(x,y,z)             : 阵列坐标系中的直角坐标

    绝对坐标 = R(boresight) @ 相对坐标 + cfg.center。
    若给了 room，绝对坐标必须落在房间内，否则报错 —— 这个报错是特性，
    它能挡住"声源跑到墙外/地板下"这类不物理的配置。
    """
    signal: str = "speech"
    params: dict = field(default_factory=dict)
    direction: tuple | None = (0.0, 0.0)      # (az, el) 度
    distance: float = 1.0                     # 米，仅 direction 模式有效
    position: tuple | None = None             # (x,y,z) 绝对坐标
    amplitude: float = 1.0                    # 阵列质心处的声压幅值
    label: str = ""

    def resolve(self) -> np.ndarray:
        if self.position is not None:
            return np.asarray(self.position, float)
        az, el = self.direction or (0.0, 0.0)
        return ac.unit_dir(az, el) * float(self.distance)

    def truth(self) -> dict:
        p = self.resolve()
        az, el = ac.angles_from_dir(p)
        return {"position": [round(float(v), 6) for v in p],
                "direction": [round(az, 4), round(el, 4)],
                "distance": round(float(np.linalg.norm(p)), 6),
                "label": self.label or self.signal}


@dataclass
class NoiseSpec:
    snr_db: float | None = 20.0
    kind: str = "white"          # white | pink
    coherent: bool = False
    seed: int = 0


@dataclass
class MismatchSpec:
    gain_db_std: float = 0.0
    phase_deg_std: float = 0.0
    pos_std_m: float = 0.0
    seed: int = 0

    def active(self) -> bool:
        return (self.gain_db_std > 0 or self.phase_deg_std > 0
                or self.pos_std_m > 0)


@dataclass
class SimConfig:
    fs: int = 16000
    duration: float = 1.0
    c: float = C_DEFAULT
    ref_distance: float = 1.0
    apply_distance_loss: bool = True
    # 阵列在房间中的位置与朝向（绝对/世界坐标）。
    # 默认：阵列贴天花板 z=3.0，主瓣朝下(-z)俯视房间 —— 对应吊顶阵列的典型装法。
    center: tuple = (0.0, 0.0, 3.0)
    boresight: tuple = (0.0, -90.0)   # (az, el) 度，阵列坐标系中指向 +z 的主瓣在世界中的方向
    # 卷积方式：auto 时按运算量选直接卷积或 FFT 卷积（两者结果一致到 1e-12）
    conv: str = "auto"
    # 分数延迟插值：sinc（默认，无幅度失真）| linear（快但有低通失真，仅用于对照）
    delay_method: str = "sinc"
    delay_taps: int = 15


@dataclass
class SimResult:
    X: np.ndarray                 # (M,N)
    fs: int
    rirs: np.ndarray | None       # (M,L) 仅房间模式
    direct: np.ndarray | None     # (M,N) 纯直达波，便于算真值 SNR
    clean: np.ndarray | None      # (M,N) 不含噪声
    noise: np.ndarray | None      # (M,N) 噪声分量
    sources: list = field(default_factory=list)
    truth: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def n_mics(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[1])

    def snr_db(self) -> float | None:
        if self.clean is None or self.noise is None:
            return None
        return sg.snr_db(self.clean, self.noise)


# --------------------------------------------------------------------------
# 镜像源法 RIR
# --------------------------------------------------------------------------
def rir_offset_samples(room: RoomSpec, src_abs: np.ndarray, mics_abs: np.ndarray,
                       center: np.ndarray, fs: int, c: float = C_DEFAULT,
                       max_order: int | None = None) -> int:
    """RIR 的全局时间偏移（采样点）。

    某些麦克风比阵列中心更靠近声源，其镜像源时延为负。直接丢弃负时延会毁掉
    那个通道。因此整个 RIR 在时间轴上右移一个**所有麦克风共用**的偏移，
    使最早到达为 0 —— 共用偏移不改变任何相对时延，波束成形结果不受影响。
    """
    imgs, _, _ = image_sources(room, src_abs, max_order)
    d_ref = max(float(np.linalg.norm(src_abs - center)), 1e-9)
    d_min = float(np.linalg.norm(imgs[:, None, :] - mics_abs[None, :, :],
                                 axis=2).min())
    return int(np.ceil(max(0.0, (d_ref - d_min) / c * fs))) + 2


def image_sources(room: RoomSpec, src: np.ndarray, max_order: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """枚举镜像源。

    return: (positions (I,3), reflect_orders (I,), atten (I,))
    atten 为各面吸声带来的幅度衰减乘积；不做空气吸收（可用 c 与距离自行折算）。
    """
    b = np.asarray(room.bounds, float)
    lo, hi = b[0::2], b[1::2]
    order = room.max_order if max_order is None else int(max_order)
    a = np.asarray(room.absorption, float)
    src = np.asarray(src, float)

    pos, att, orders = [], [], []
    rng = range(-order, order + 1)
    for idx in itertools.product(rng, repeat=3):
        n = np.asarray(idx, int)
        if int(np.abs(n).sum()) > order:
            continue
        # 反射后的镜像坐标（对每个轴独立展开，偶数次不翻转）
        p = src.copy()
        for ax in range(3):
            k = n[ax]
            if k == 0:
                continue
            if k % 2 == 0:
                p[ax] = src[ax] + k * (hi[ax] - lo[ax])
            else:
                base = lo[ax] if k > 0 else hi[ax]
                p[ax] = 2 * base - src[ax] + (k - (1 if k > 0 else -1)) * \
                    (hi[ax] - lo[ax])
        # 衰减：每个轴 |k| 次反射
        atten = 1.0
        for ax in range(3):
            for _ in range(abs(int(n[ax]))):
                atten *= np.sqrt(max(1.0 - a[2 * ax + (1 if n[ax] > 0 else 0)], 0.0))
        pos.append(p)
        att.append(atten)
        orders.append(int(np.abs(n).sum()))
    return np.asarray(pos), np.asarray(orders), np.asarray(att)


def rotation_to_boresight(boresight: tuple) -> np.ndarray:
    """把 +z 旋到指定 (az,el) 的旋转矩阵 R (3,3)。

    阵列几何按 +z 为正面设计；R 同时作用于阵元坐标与声源方向，
    因此波束图的"相对角"不变，改变的是阵列在房间中的绝对朝向。
    """
    az, el = np.deg2rad(boresight[0]), np.deg2rad(boresight[1])
    target = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, target)
    s, c = np.linalg.norm(v), float(np.dot(z, target))
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def _convolve(sig: np.ndarray, rir: np.ndarray, n: int, mode: str = "auto") -> np.ndarray:
    """确定性卷积，截断到 n 点。

    auto: 短 RIR 走 np.convolve，长 RIR 走 FFT 卷积。两者都是确定性的，
    差异仅来自浮点求和顺序（量级 1e-14），物理意义完全相同。
    """
    use_fft = mode == "fft" or (mode == "auto" and rir.size > 256)
    if use_fft:
        nl = sig.size + rir.size - 1
        # 必须取 >= nl 的 2 的幂，否则 FFT 循环卷积会绕回污染信号开头
        nfft = 1 << int(np.ceil(np.log2(max(nl, 2))))
        if nfft < nl:
            nfft *= 2
        y = np.fft.irfft(np.fft.rfft(sig, nfft) * np.fft.rfft(rir, nfft), n=nfft)
        return y[:n]
    return np.convolve(sig, rir)[:n]


def reflection_report(room: RoomSpec, src: np.ndarray, c: float = C_DEFAULT,
                      max_order: int | None = None,
                      reference: np.ndarray | None = None) -> dict:
    """相对直达波的反射到达时间统计，供 GUI 显示早期反射结构。

    时间以 `reference`（默认声源本身）处的直达波为 0；
    给 reference 时按阵列中心计算，并把时间轴平移到最早到达 = 0，
    与 shoebox_rir 的 offset_samples 约定一致。
    """
    imgs, orders, att = image_sources(room, src, max_order)
    d0 = float(np.linalg.norm(src - (src if reference is None else reference)))
    dd = np.linalg.norm(imgs - src[None, :], axis=1)
    t = (dd - d0) / c
    if reference is not None:
        t -= t.min()                      # 最早到达 = 0，与 RIR 偏移一致
    o = np.argsort(t)
    t, orders, att = t[o], orders[o], att[o]
    return {
        "n_images": int(t.size),
        "max_order": int(orders.max()) if t.size else 0,
        "t_first_ms": round(float(t[1] * 1e3), 3) if t.size > 1 else None,
        "t_last_ms": round(float(t[-1] * 1e3), 3) if t.size else 0.0,
        "early": [{"t_ms": round(float(t[k] * 1e3), 3),
                   "order": int(orders[k]),
                   "gain_db": round(float(20 * np.log10(max(att[k], 1e-12))), 2)}
                  for k in range(1, min(12, t.size))],
    }


def shoebox_rir(room: RoomSpec, mic_pos: np.ndarray, src_pos: np.ndarray,
                fs: int, c: float = C_DEFAULT, max_order: int | None = None,
                n_samples: int | None = None,
                ref_pos: np.ndarray | None = None,
                offset_samples: int = 0,
                direct_only: bool = False) -> np.ndarray:
    """单麦克风镜像源法脉冲响应。

    **时间基准**（关键）：`ref_pos`（默认阵列中心）到声源的直达波为 t=0。
    绝不可以用"本麦克风自己的直达波"做基准 —— 那样会把麦克风之间的
    相对时延抹掉，阵列的空间信息就全丢了。

    归一化约定：RIR 描述**相对传递函数**，幅度基准为直达波 1.0（按球面扩散 1/r
    相对衰减），绝对电平由 SourceSpec.amplitude 与 cfg.ref_distance 施加。
    """
    mic_pos = np.asarray(mic_pos, float)
    src_pos = np.asarray(src_pos, float)
    ref = src_pos if ref_pos is None else np.asarray(ref_pos, float)

    imgs, orders, att = image_sources(room, src_pos, max_order)

    if direct_only:
        # 只保留直达路径（最近的镜像，即源本身），走与完整 RIR **完全相同**的
        # 量化/插值路径，保证 direct 参考与 clean 在时间与幅度上逐点可比。
        d0 = float(np.linalg.norm(src_pos - mic_pos))
        i = int(np.argmin(np.linalg.norm(imgs - src_pos[None, :], axis=1)))
        imgs, att = imgs[i:i + 1], att[i:i + 1]

    d_all = np.linalg.norm(imgs - mic_pos[None, :], axis=1)
    d_ref = float(np.linalg.norm(src_pos - ref))             # 阵列中心到声源
    if d_ref <= 1e-9:
        d_ref = 1e-9

    delays = (d_all - d_ref) / c                             # 相对阵列中心直达波
    gains = att * (d_ref / np.maximum(d_all, 1e-9))

    if n_samples is None:
        n_samples = int(np.ceil((delays.max() + 0.05) * fs)) + 1
    rir = np.zeros(int(n_samples), dtype=np.float64)

    t = delays * fs + float(offset_samples)                  # 共用偏移，保持相对时延
    i0 = np.floor(t).astype(int)
    frac = t - i0
    ok = (i0 >= 0) & (i0 < rir.size)
    np.add.at(rir, i0[ok], gains[ok] * (1.0 - frac[ok]))
    ok1 = (i0 + 1 >= 0) & (i0 + 1 < rir.size)
    np.add.at(rir, (i0 + 1)[ok1], gains[ok1] * frac[ok1])
    return rir


# --------------------------------------------------------------------------
# 主仿真
# --------------------------------------------------------------------------
def simulate(geom, sources: list[SourceSpec], cfg: SimConfig | None = None,
             room: RoomSpec | None = None, noise: NoiseSpec | None = None,
             mismatch: MismatchSpec | None = None) -> SimResult:
    """生成多通道观测。

    返回 X 的第 m 通道为：sum_s amplitude_s * spread_s,m * (rir_s,m ⊛ sig_s) + noise
    """
    cfg = cfg or SimConfig()
    fs, N = int(cfg.fs), int(round(cfg.duration * cfg.fs))
    c = float(cfg.c)

    # 阵列在绝对坐标系中的位置与朝向
    R = rotation_to_boresight(tuple(cfg.boresight))
    center = np.asarray(cfg.center, float)
    mics = (R @ np.asarray(geom.mics, float).T).T + center[None, :]
    M = mics.shape[0]

    clean = np.zeros((M, N))
    direct = np.zeros((M, N))
    rirs = None
    src_truth = []
    rir_len = 0

    # 先解析所有声源（绝对坐标），并确定 RIR 长度
    resolved = []
    for si, src in enumerate(sources):
        p_rel = src.resolve()
        p = R @ p_rel + center                     # 相对阵列 -> 绝对坐标
        if room is not None and not room.contains(p, margin=1e-6):
            raise ValueError(
                f"source {si} at {np.round(p, 3)} (绝对坐标) is outside room "
                f"bounds {room.bounds}；请调整 cfg.center / distance / 房间尺寸")
        resolved.append((src, p_rel, p))

    offsets: list[int] = []
    if room is not None:
        for _, _, p in resolved:
            off = rir_offset_samples(room, p, mics, center, fs, c, room.max_order)
            offsets.append(off)
            # RIR 长度：覆盖最远镜像源 + 全局偏移 + 余量
            imgs, _, _ = image_sources(room, p, room.max_order)
            dmax = float(np.linalg.norm(imgs[:, None, :] - mics[None, :, :],
                                        axis=2).max())
            dmin = float(np.linalg.norm(imgs[:, None, :] - mics[None, :, :],
                                        axis=2).min())
            dref = float(np.linalg.norm(p - center))
            span = (dmax - dmin) / c + (dref - dmin) / c
            rir_len = max(rir_len, int(np.ceil(span * fs)) + 64)
        rir_len = max(rir_len, int(0.05 * fs))
        rirs = np.zeros((M, rir_len))
    else:
        offsets = [0] * len(resolved)

    for si, (src, p_rel, p) in enumerate(resolved):
        sig = sg.generate(src.signal, fs, N, src.params)
        sig = np.asarray(sig, float)
        sig = sig / (np.abs(sig).max() + 1e-12) * float(src.amplitude)

        r_center = float(np.linalg.norm(p_rel))
        g0 = (cfg.ref_distance / max(r_center, 1e-9)) if cfg.apply_distance_loss else 1.0

        for m in range(M):
            mic = mics[m]
            d_m = float(np.linalg.norm(mic - p))
            if room is None:
                tau = (d_m - r_center) / c
                g_m = (cfg.ref_distance / max(d_m, 1e-9)) \
                    if cfg.apply_distance_loss else 1.0
                sig_d = _delay_frac(sig, tau * fs, cfg.delay_method,
                                    cfg.delay_taps)
                clean[m] += g_m * sig_d
                direct[m] += g_m * sig_d
            else:
                rir = shoebox_rir(room, mic, p, fs, c, room.max_order,
                                  n_samples=rir_len, ref_pos=center,
                                  offset_samples=offsets[si])
                rirs[m, :rir.size] += rir
                g_m = (cfg.ref_distance / max(d_m, 1e-9)) \
                    if cfg.apply_distance_loss else 1.0
                clean[m] += g_m * _convolve(sig, rir, N, cfg.conv)
                # 纯直达波参考：与上面 RIR 同源、同量化路径，只保留直达抽头。
                # 这样 direct 与 clean 的时间/幅度基准严格一致。
                rir_d = shoebox_rir(room, mic, p, fs, c, room.max_order,
                                    n_samples=rir_len, ref_pos=center,
                                    offset_samples=offsets[si], direct_only=True)
                direct[m] += g_m * _convolve(sig, rir_d, N, cfg.conv)

        src_truth.append(_truth_abs(src, p, R, cfg.boresight))

    # 失配（注意：apply_mismatch 恒返回 tuple，不要整体覆盖 clean）
    mis_info = None
    if mismatch is not None and mismatch.active():
        u_ref = None
        if len(resolved) == 1:
            u_ref = resolved[0][1] / max(np.linalg.norm(resolved[0][1]), 1e-12)
        clean, mis_info = sg.apply_mismatch(clean, mismatch.gain_db_std,
                                            mismatch.phase_deg_std,
                                            mismatch.pos_std_m, mismatch.seed,
                                            fs=fs, c=c, direction=u_ref)
        mis_info["applied"] = True
    elif mismatch is not None:
        mis_info = {"applied": False}

    # 噪声
    noise_arr = np.zeros_like(clean)
    snr_actual = None
    if noise is not None and noise.snr_db is not None:
        X, snr_actual = sg.add_noise(clean, noise.snr_db, noise.kind, noise.seed,
                                     noise.coherent)
        noise_arr = X - clean
    else:
        X = clean.copy()

    meta = {
        "fs": fs, "n_samples": N, "n_mics": M, "c": c,
        "ref_distance": cfg.ref_distance,
        "center": [round(float(v), 6) for v in center],
        "boresight": [float(cfg.boresight[0]), float(cfg.boresight[1])],
        "room": None if room is None else {
            "bounds": [float(v) for v in room.bounds],
            "absorption": [float(v) for v in room.absorption],
            "max_order": int(room.max_order),
            "rt60_est": estimate_rt60(room),
            "rir_len": int(rir_len),
            "reflections": reflection_report(room, resolved[0][2], c,
                                             room.max_order) if resolved else None,
        },
        "snr_target_db": None if noise is None else noise.snr_db,
        "snr_actual_db": None if snr_actual is None else round(snr_actual, 4),
        "mismatch": mis_info,
        "n_sources": len(sources),
    }
    return SimResult(X=X, fs=fs, rirs=rirs, direct=direct, clean=clean,
                     noise=noise_arr, sources=[s.truth() for s in sources],
                     truth={"sources": src_truth}, meta=meta)


def _truth_abs(src: SourceSpec, p_abs: np.ndarray, R: np.ndarray,
               boresight: tuple) -> dict:
    """真值：同时给出绝对角度与相对阵列的角度。

    "相对阵列" 才是定位算法的答案空间：波束图与 DOA 估计都在阵列坐标系里。
    """
    az_abs, el_abs = ac.angles_from_dir(p_abs)
    p_rel = np.asarray(src.resolve(), float)
    az_rel, el_rel = ac.angles_from_dir(p_rel)
    return {
        "position_abs": [round(float(v), 6) for v in p_abs],
        "position_rel": [round(float(v), 6) for v in p_rel],
        "direction_abs": [round(az_abs, 4), round(el_abs, 4)],
        "direction_rel": [round(az_rel, 4), round(el_rel, 4)],
        "distance": round(float(np.linalg.norm(p_rel)), 6),
        "label": src.label or src.signal,
    }


def _delay_frac(x: np.ndarray, delay_samples: float,
                method: str = "sinc", taps: int = 15) -> np.ndarray:
    """任意分数延迟。

    method='sinc'（默认）：加窗 sinc 插值。这是**精确**的带限分数延迟，
    通带内不引入幅度失真 —— 阵列仿真必须用它。

    method='linear'：线性插值。实现简单，但线性插值等效于一个三角窗
    低通滤波器，在 6 kHz（fs=16k）处衰减约 4 dB。用它会带来一个隐蔽的
    后果：仿真的"多通道信号"被低通，之后任何对准求和都无法完全恢复，
    表现为"信号保留只有 0.83x"而不是 1.0x。这个坑踩过，故保留该模式
    仅用于对照与和 C++ 逐位比对的降级路径。

    约定：delay_samples > 0 表示信号被**延迟**（向右移）。
    """
    x = np.asarray(x, float)
    n = x.size
    if method == "linear":
        t = np.arange(n, dtype=np.float64) - float(delay_samples)
        i0 = np.floor(t).astype(int)
        frac = t - i0
        out = np.zeros(n)
        ok = (i0 >= 0) & (i0 < n)
        out[ok] += x[i0[ok]] * (1.0 - frac[ok])
        ok1 = (i0 + 1 >= 0) & (i0 + 1 < n)
        out[ok1] += x[(i0 + 1)[ok1]] * frac[ok1]
        return out

    # 加窗 sinc：y[n] = sum_k x[k] * w(k - (n - d)) * sinc(k - (n - d))
    # 用 Kaiser 窗抑制截断带来的旁瓣（旁瓣泄漏会污染弱声源，对成像不利）
    half = max(int(taps) // 2, 3)
    k = np.arange(-half, half + 1, dtype=np.float64)
    win = np.kaiser(2 * half + 1, 8.0)

    d = float(delay_samples)
    t = np.arange(n, dtype=np.float64) - d                 # 需要的源位置
    i0 = np.floor(t).astype(int)
    frac = t - i0                                          # in [0,1)
    out = np.zeros(n)
    gain = np.zeros(n)                                     # 实际插值滤波器的直流增益
    for j, kk in enumerate(k):
        idx = i0 + int(kk)
        off = frac - kk                                    # sinc 自变量
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(np.abs(off) < 1e-12, 1.0,
                         np.sin(np.pi * off) / (np.pi * np.where(
                             np.abs(off) < 1e-12, 1.0, off)))
        g = win[j] * s
        ok = (idx >= 0) & (idx < n)
        out[ok] += x[idx[ok]] * g[ok]
        gain[ok] += g[ok]
    # 逐点归一化到单位直流增益。不归一化时插值滤波器的整体增益不等于 1，
    # 表现为"信号保留莫名其妙只有 0.69x"（曾因此误判成波束成形 bug）。
    # 逐点归一化还顺带修掉了边界处有效抽头数减少导致的幅度塌陷。
    nz = np.abs(gain) > 1e-9
    out[nz] /= gain[nz]
    return out


def estimate_rt60(room: RoomSpec, c: float = C_DEFAULT) -> float:
    """Sabine 公式估算混响时间（仅作参考显示）。"""
    dims = room.size()
    V = float(np.prod(dims))
    S = float(2 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2]))
    a = np.asarray(room.absorption, float)
    areas = np.array([dims[1] * dims[2], dims[1] * dims[2],
                      dims[0] * dims[2], dims[0] * dims[2],
                      dims[0] * dims[1], dims[0] * dims[1]])
    A = float(np.sum(areas * a))
    if A <= 1e-9:
        return float("inf")
    return round(0.161 * V / A, 3)


__all__ = [
    "RoomSpec", "SourceSpec", "NoiseSpec", "MismatchSpec", "SimConfig",
    "SimResult", "simulate", "shoebox_rir", "image_sources", "estimate_rt60",
]
