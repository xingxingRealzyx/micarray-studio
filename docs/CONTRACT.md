# MicArray Studio — 引擎接口契约 (frozen v1)

> 这个文件是引擎、导出层、GUI 三者之间的唯一约定来源。
> 任何改动必须先改这里，再改实现。GUI 前端只依赖本文件描述的 JSON schema。

## 1. 坐标系与单位

- **右手系笛卡尔**，单位 **米**。
- 存在两个坐标系，务必区分：

| 坐标系 | 用途 | 定义 |
|---|---|---|
| **阵列坐标系 (array frame)** | 几何、导向矢量、波束图、DOA 答案空间 | 阵元在 **xy 平面**，`z=0`；**boresight = +z**；原点 = 阵列质心 |
| **世界坐标系 (world frame)** | 房间、镜像源、RIR | `+z` 向上（`z=0` 为地面）；原点任意，由 `RoomSpec.bounds` 给出 |

- 两者关系：`p_world = R(boresight) @ p_array + cfg.center`，其中 `R` 把阵列坐标系的 +z 旋到 `cfg.boresight` 指定的世界方向。
- **所有算法输出（波束图网格、DOA 估计、声场图）都在阵列坐标系中。** 世界坐标只用于仿真房间反射。
- 麦克风位置：`mics: (M, 3) float64`，不含朝向（全向麦假设）。
- 阵列几何中心（centroid）默认平移到原点；`metadata.center_offset` 记录被平移掉的原偏移。

## 2. 角度约定

**在阵列坐标系中**：

- **方位角 azimuth `az`**：xy 平面内，从 +x 轴逆时针转向 +y 轴，`[-180, 180]` 度。
- **俯仰角 elevation `el`**：**从 xy 平面量向 +z**，`[-90, 90]` 度。
  - `el = 0` → 阵面切线方向（阵元所在平面内）
  - `el = +90` → **正对 boresight**（阵列正前方 / 吊顶阵列的正下方）
  - `el = -90` → 阵列正后方（背对方向，用于检查后向抑制）
- 单位方向向量：

```
u(az, el) = [ cos(el)cos(az), cos(el)sin(az), sin(el) ]
```

- 方向网格 `direction_grid(...)` 返回 `dirs: (G,3)` 与 `shape: (n_el, n_az)`，
  `dirs` 按 `el` 外层、`az` 内层展平（C 序，与 `shape` 一致）。
- `(az, el)` 在 `el = ±90` 处 azimuth 退化（奇点），这是球坐标的固有问题；
  绘图时建议把网格画成 u-v 圆盘投影而非矩形 az-el 图，避免极点附近失真。

## 3. 导向矢量（steering vector）

**唯一定义**（远场平面波），符号必须与导出 C++ 完全一致：

```
a_m(f, u) = exp( -j * 2*pi*f/c * dot(mics[m], u) )        m = 0..M-1
```

- `c = 343.0` m/s（默认，可由 `SimConfig.speed_of_sound` 覆盖）。
- 约定：**阵列接收**方向为 `u` 的平面波。`a` 为列向量 `(M,) complex128`。
- 近场球面波（`r < near_field_limit`）：

```
a_m(f, p) = (r_ref / ||mics[m] - p||) * exp( -j * 2*pi*f/c * (||mics[m] - p|| - ||p||) )
```

  其中 `r_ref = ||p||` 为参考距离，相位以**阵列质心处的声源距离**为基准，保证 `p` 远离时连续退化为平面波。

**这一条是全线一致性的锚点。** Python、C++、GUI 三处的波束图必须由它推出。

## 4. 声场仿真数据流

```
SourceSpec[] ──┐
RoomSpec?   ───┼──> simulate() ──> SimResult
NoiseSpec?  ───┤                     ├── X       : (M, N) float64  多通道观测（时域）
ArraySpec   ───┘                     ├── fs      : int            采样率
                                     ├── rirs    : (M, L) | None   镜像源法 RIR（若启用）
                                     ├── sources : 回填的真实参数（方位/距离/信噪比）
                                     └── meta    : dict
```

- 时域约定：`X[m, n]`，`n` 为采样点序号，`t = n / fs`。
- 信号幅度标定：`SourceSpec.amplitude` 为**阵列质心处**的声压幅值。
  距离衰减为 `1/||p||`（近场）或 `1/||p - center||`（远场等效），
  因此**同一 amplitude 的声源在不同距离上到达阵列的绝对电平不同**——这是期望行为。
- 全局电平统一由 `SimConfig.ref_distance`（默认 1 m）与 `ref_pressure`（默认 1.0）定义。

## 5. 频域约定（波束成形/定位/成像共用）

- 统一用 `freq.py` 的 `stft()`：**rfft，单边谱**，`n_fft` 默认 512，`hop = n_fft // 2`，
  Blackman-Harris 窗（与 C++ 导出使用同一张窗表，7 位有效数字，走 `json` 传递）。
- 输出：`Xf: (M, K, F) complex128`，`K` 帧数，`F = n_fft//2 + 1` 频点数。
- 频率轴：`freqs = np.fft.rfftfreq(n_fft, 1/fs)`。
- **不做幅度补偿**（不除窗增益），保证与 C++ 逐位可比；需要物理幅度时调用方自行 `* 2 / (win.sum())`。
- 阵列协方差：`R(f) = E[ Xf[:, k, f] @ Xf[:, k, f].conj().T ]`，对 `k` 求平均。

### 5.1 逆变换的窗端陷阱（重要，改动过频谱时必须读）

`istft()` 用窗平方和 `wsum` 归一化。Blackman-Harris 窗端点的 `wsum` 比稳态
**小 3e7 倍**，因此：

- **只要动过频谱**（乘波束权重、谱增益、相位掩蔽…），WOLA 恒等式不再逐点成立，
  在端点处 `y/wsum` 会把数值噪声放大到 1e4 倍量级。实测随机相位掩蔽下
  `max|y|` 从 3.9 涨到 9121（**2339 倍**），SI-SDR 直接崩。
- 引擎的对策：`wsum <= 0.05 * max(wsum)` 的位置**直接置零**（那里有效抽头不足，
  任何"保幅度"的折中都只是放大噪声）。C++ 导出的 `istft` 用**同一阈值**，
  否则 parity 会在端点失配。
- **推荐做法**：`stft(..., center=True)` + `istft(..., center=True)`，首尾各有
  `n_fft//2` 的补零重叠区，从根本上不存在窗端问题。此时输出比输入长 `n_fft`
  点，需自行裁掉首尾各 `n_fft//2`。
- `center=False`（默认，为与 C++ 逐位可比）时：完美重构在中段精确到 1e-15，
  但**首尾各约 `n_fft` 点不可用**（被置零）。做频域修改请显式用 `center=True`。

## 6. 波束成形器统一接口

```python
w = design(geom, freqs_sel, method=..., params...)   # -> BeamformerResult
y = apply(bf, X_or_Xf)                               # -> 时域或频域输出
```

`BeamformerResult` 必须包含：

| 字段 | 形状 | 含义 |
|---|---|---|
| `freqs` | `(F,)` | 设计所用频点 |
| `weights` | `(F, M)` complex | 每频点复权重 |
| `delays` | `(F, M)` float or None | 时域 FIR 分数延迟（若走时域实现） |
| `method` | `str` | 方法名 |
| `steer_dir` | `(3,)` | 设计指向 |
| `metrics` | `dict` | 见下 |

`metrics` 必含：`wng`（白噪增益，`(F,)`）、`di`（指向性因子 dB，`(F,)`，若可算）、
`beampattern`（可选，`(F, G)`）、`nulls`（可选）。

**白噪增益**（对空间白噪声的阵列增益，标准定义）：

```
WNG(f) = |w^H a|^2 / (w^H w)        # 线性比值；dB 值为 10*log10(WNG)
```

对 M 元阵，延迟求和（w = a/M）给出 WNG = M，即上限。
MVDR/超指向在低频会牺牲 WNG 换取指向性，WNG 是**鲁棒性的第一指标**：
WNG 低于约 0 dB 意味着该频点已被失配和自噪吞掉，实用中要设 WNG 下限约束。

**指向性因子** 用扩散场噪声协方差 `Gamma_diff` 计算（标准定义）：

```
DI(f) = |w^H a|^2 / (w^H Gamma_diff w)      # 线性比值；dB 值为 10*log10(DI)
Gamma_diff[i,j] = sinc(2*pi*f*d_ij/c)       # 三维扩散场相干
```

理论上 `DI <= M`（各向同性上限）。用 `steering.diffuse_coherence()` 生成 `Gamma_diff`。

## 7. 导出层契约（C++17）

导出产物是**自包含**的算法内核，不做 I/O、不依赖 GUI：

```
export/<name>/
  include/mas_array_config.hpp    // 几何表、常数、窗表（constexpr 数组）
  include/mas_<algo>.hpp          // 算法接口（纯头文件 + 一个 .cpp）
  src/mas_<algo>.cpp
  src/mas_fft.cpp                 // 自研 radix-2 FFT，与 numpy 逐位对齐
  tests/test_parity.cpp           // 读取导出的测试向量，比对 1e-6
  tests/vectors/*.json            // Python 侧生成的输入/期望输出
  CMakeLists.txt                  // option(MAS_USE_EIGEN ON/OFF)
```

**数值一致性要求**：相同输入下，C++ 与 Python 的 `|Δ| / |ref| <= 1e-6`（float64）。
其中 FFT 与窗表是硬约束——用同一张表、同一套蝶形顺序。

**Eigen 策略**：`MAS_USE_EIGEN=ON` 时用 `Eigen::MatrixXcd` 解 MVDR 线性方程组；
`OFF` 时用自带的复 Cholesky/LDL^T 求解器。两条路径结果必须一致（容差 1e-9）。

## 8. GUI JSON schema（前端只依赖这一节）

```
GET  /api/arrays                       -> {presets:[...], current:GeometryJSON}
POST /api/array/design                 -> GeometryJSON
POST /api/beamform                     -> {freqs, beampattern, metrics, geometry}
POST /api/locate                       -> {map, shape, est_dirs, est_xyz, truth}
POST /api/imaging                      -> {dirty, deconv, shape, psf, method}
POST /api/enhance                      -> {snr_in, snr_out, stoi_proxy, audio_url?, curve}
POST /api/export/cpp                   -> {job_id, files[], compile_log, parity}
GET  /api/jobs/<id>                    -> {status, progress, result}
GET  /api/audio/<id>.wav               -> wav 二进制（标准库 wave 写出）
```

- 所有几何 JSON：

```json
{ "name":"spiral-arm", "mics":[[x,y,z],...], "center":[0,0,0],
  "meta":{"aperture":0.42,"min_spacing":0.011,"n_mics":16,
          "redundancy":{...},"kind":"spiral_arm","params":{...}} }
```

- 大数组（热力图 `map`）用**扁平数组 + shape** 传递，避免嵌套 JSON 开销。
- 所有 POST body 必须可序列化为纯 JSON（无 NaN/Inf，用 `null`）。

## 9. 术语与符号表

| 符号 | 含义 |
|---|---|
| `M` | 麦克风数量 |
| `F` | 单边谱频点数 = `n_fft//2+1` |
| `K` | STFT 帧数 |
| `G` | 方向网格点数 |
| `az, el` | 方位角、俯仰角（度） |
| `u` | 单位方向向量 |
| `a` | 导向矢量 `(M,)` |
| `w` | 波束成形权重 `(M,)` |
| `R` | 阵列协方差 `(M,M)` |
| `Gamma` | 噪声相干矩阵 `(M,M)` |
| `WNG` | 白噪增益 |
| `DI` | 指向性因子 |
| `PSF` | 点扩散函数 |
