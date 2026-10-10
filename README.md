# MicArray Studio

麦克风阵列仿真软件：**阵型设计 → 声场仿真 → 波束成形 / 定位 / 成像 / 语音增强 → 导出可编译验证的 C++17 算法**。

浏览器里拖阵型、看波束图与声场图，一键把算法导成自包含的 C++ 工程，并自动编译比对数值一致性。

---

## 快速开始

```bash
# 只需 numpy，不需要 pip 安装本项目（没有 setup.py，直接跑脚本）
python3 -c "import numpy; print(numpy.__version__)"   # 确认有 numpy

python3 selftest.py                 # 一键自检：四类算法 + C++ 导出验证（29 项）
python3 selftest.py --quick         # 跳过 C++ 编译（省时间）
python3 backend/server.py --port 8777
# 浏览器打开 http://127.0.0.1:8777
```

**依赖**

| 部分 | 依赖 |
|---|---|
| 引擎 + 后端 | Python 3.10+、numpy（**不需要 scipy / fastapi / uvicorn / matplotlib**） |
| GUI | 任意现代浏览器（前端零第三方库、零 CDN、零构建步骤） |
| C++ 导出验证 | `cmake` + 支持 C++17 的编译器；Eigen 可选（没装就走纯 C++ 实现） |

> 如果 `python3` 太旧或没装 numpy，用你环境里任意 3.10+ 且带 numpy 的解释器即可，
> 把上面的 `python3` 换成它的路径。所有脚本都从自身位置推导项目根目录，不依赖调用目录。

---

## 目录结构

```
engine/                     纯 numpy 仿真引擎（可独立于 GUI 使用）
  geometry.py               阵型构建：ULA/UCA/网格/螺旋臂/悬臂/Vogel/嵌套/多环
  acoustics.py              几何声学：时延、球面扩散、近场判据、无混叠上限
  steering.py               导向矢量（远场/近场）、扩散场相干矩阵
  dsp.py                    窗、STFT/ISTFT、独立 DFT（用于校验 FFT）
  tdoa.py                   GCC-PHAT（含抛物插值）、多源时延聚类
  signals.py                信号源、噪声、通道幅相/位置失配、评价指标
  sim.py                    声场仿真：镜像源法 RIR、房间、失配、噪声
  beamforming.py            DS / MVDR / 超指向（扩散场）/ 频不变 + WNG / DI
  localization.py           SRP-PHAT / MUSIC / 闭式+LM TDOA / 近场距离估计
  imaging.py                PSF / 脏图 / DAMAS / CLEAN-SC / 正交与函数波束成形
  enhance.py                MVDR+后置滤波 / 多通道维纳 / GSC / 谱减 / 去混响
  codegen.py                C++17 导出：生成 + 造向量 + 编译 + 比对
backend/
  server.py                 标准库 http.server + JSON API（无需 fastapi）
  params.py                 请求参数归一化（扁平/嵌套两种写法都接受）
web/
  index.html                单文件前端（原生 JS + Canvas，无任何第三方库）
  mock_server.py            前端自测用的假数据服务
docs/CONTRACT.md            接口契约 —— 坐标系、导向矢量、数据流、导出约定
tests/                      算法自检脚本
selftest.py                 一键自检入口
export/                     导出的 C++ 工程（运行时生成）
```

---

## 几个关键设计决定

### 1. 坐标系只有一个约定，写死在契约里

阵元位于 **xy 平面**，`boresight = +z`。俯仰角 `el` 从 xy 平面量向 +z：`el=+90` 是正对阵列，`el=0` 是阵面切线方向。

这一条曾经踩过坑：阵元在 xy 平面、而角度却按"el 从 xy 平面量起"理解时，`el=0` 会落到阵列的**盲区**（阵面内没有分辨率）。现在契约、实现、GUI 三处统一，`docs/CONTRACT.md` 第 2 节是唯一来源。

世界坐标（房间、镜像源）与阵列坐标严格分离：`p_world = R(boresight) @ p_array + cfg.center`。**所有算法输出都在阵列坐标系里**。

### 2. 导出即验证

导出 C++ 不是"生成一堆看起来对的代码"，而是：

```
设计权重 → 生成 C++ 工程 → 生成测试向量 → cmake 编译 → 运行 parity → 比对最大相对误差
```

实测结果：

| 项目 | 最大相对误差 |
|---|---|
| `beamform()` vs Python `beamforming.process()` | 7.8e-15 |
| `delay_and_sum()` 解析实现 vs Python | 8.3e-15 |
| `white_noise_gain()` vs Python `wng_db()` | 0 |
| `beampattern()` vs Python `evaluate_beampattern()` | 4.4e-16 |

编译不过或比对不过就不算导出成功。测试向量走纯文本，避免在实时路径上引入 JSON 解析依赖。

### 3. 数值一致性靠"同一个约定"，不靠容差

- FFT：C++ 用自研迭代 radix-2，Python 用 numpy，两者差异仅来自浮点求和顺序
- 窗表：量化到 12 位有效数字后**双方共用同一张表**
- 权重：实部/虚部分开存为 `constexpr` 数组，不依赖 `std::complex` 的内存布局
- 公式符号：`a_m(f,u) = exp(-j2πf·dot(p_m,u)/c)` 在契约里写死，两侧逐字对应

---

## 引擎能力与实测数字

### 阵型（同样 16 元，口径与无混叠上限的取舍）

| 阵型 | M | 口径 | min_d | 无混叠上限 | 冗余度 |
|---|---|---|---|---|---|
| 线阵 16 元 31mm | 16 | 0.465 m | 31.0 mm | 5532 Hz | 8.00 |
| 圆阵 16 元 Ø200mm | 16 | 0.200 m | 39.0 mm | 4395 Hz | 15.00 |
| 方阵 4×4 | 16 | 0.191 m | 45.0 mm | 3811 Hz | 13.33 |
| 螺旋臂 4×4（对数） | 16 | 0.500 m | 42.4 mm | 4042 Hz | 3.87 |
| **螺旋臂 4×4 宽带** | 16 | **1.000 m** | **14.1 mm** | **12127 Hz** | 3.87 |
| 悬臂 4 臂×4 麦 | 16 | 0.480 m | 31.5 mm | 5446 Hz | 4.62 |
| **Vogel 螺旋 16 元** | 16 | 0.274 m | 58.0 mm | 2958 Hz | **1.00** |
| 嵌套阵 4+4 | 8 | 0.245 m | 20.0 mm | 8575 Hz | 1.27 |

这张表本身就是这个软件的价值所在：**同样 16 元，对数径向密度能把"大口径 + 高无混叠上限"同时拿到**（1 m 口径 + 12.1 kHz），而等间距线阵做不到（0.465 m 口径就只有 5.5 kHz）。Vogel 螺旋则给出数学上最优的非冗余布点（冗余度 1.00）。

### 仿真

- 镜像源法 RIR，任意阶反射，Sabine RT60 估算
- 每个麦克风的 RIR 直达峰与理论时延误差 **< 0.5 采样点**
- `X == clean + noise` 恒等式精确成立（0.0e+00）
- 真值 SNR 精确等于目标值（25.000 dB）
- 分数延迟支持加窗 sinc（整数延迟精确到 3.3e-16）与线性插值两种模式

### 波束成形

- DS 的白噪增益恒等于 `10log10(M)` —— 在 5 种阵型上逐一验证
- 所有方法严格满足无失真约束 `w^H a = 1`（偏差 < 1e-13，含对角加载后）
- 端到端 SNR 提升 **12.04 dB**（16 元理论上限 12.04 dB，打满）
- MVDR 无加载置零深度 98 dB；加载增大则 WNG 升高、SINR 单调下降，权衡曲线干净
- 频不变设计把主瓣宽度跨频率一致性从 4.00 改善到 1.15（比值）

### 定位 / 成像 / 增强

| 功能 | 实测 |
|---|---|
| SRP-PHAT 单声源定位 | 误差 **0.63°**（37×72 网格） |
| 混响房间内定位 | 误差 < 1° |
| 近场 TDOA 位置估计 | 见 `tests/test_localization.py` |
| DAMAS 反卷积 | 主瓣显著变窄，31×61 网格 19 s |
| MVDR + 后置滤波 | 分段 SNR 提升 **+9.1 ~ +14.5 dB**，SI-SDR +6.4 dB |
| GSC | 分段 SNR 提升 **+8.4 dB**，SI-SDR +3.6 dB |

---

## 关于一个反直觉的物理结论

`DI = 13.03 dB` 竟然**超过**了各向同性上限 `10log10(16) = 12.04 dB`。一开始我以为是 bug，查清后确认**不是**：

三维扩散场相干矩阵 `Γ_ij = sinc(k·d_ij)` 在 `k·d` 较大时会出现**负相干**，此时 `w^HΓw` 可以小于 `w^Hw`，于是 DI 能超过上限。这不是算错，而是该模型在"阵元间距 >> 波长"时不再描述真实噪声场。

引擎提供 `beamforming.coherence_diagnostics()` 把 Γ 的最小特征值与条件数报出来，并在 `metrics['di_exceeds_bound']` 里标记这种情况。

## 关于近场损失

用平面波权重的延迟求和在近场会有损失，实测（0.5 m 口径）：

| 源距离 | 100–300 Hz | 1–2 kHz | 2–4 kHz | 4–7 kHz |
|---|---|---|---|---|
| 0.5 m | 0.996 | 0.816 | 0.494 | 0.562 |
| 1.2 m | 0.999 | 0.962 | 0.857 | 0.591 |
| 3.0 m | 1.000 | 0.994 | 0.975 | 0.901 |
| 20 m | 1.000 | 1.000 | 0.999 | 0.995 |

这是**真物理**：近场波前曲率（`r < 2D²/λ`）+ 高频空间混叠。验证方式是把近场导向矢量与单源最优解（协方差矩阵主奇异向量）对比，相干度 **0.9969–1.0000**，而远场权重只有 0.22–0.99。也就是说近场模型是正确的，损失是"用错模型"造成的。

`acoustics.is_near_field()` / `fresnel_distance()` 可以判断何时必须切近场模型。

---

## API

见 `docs/CONTRACT.md` 第 8 节。简要：

```
GET  /api/health                    模块可用性
GET  /api/arrays                    预设列表 + 角度约定说明
POST /api/array/design              阵型设计 -> 几何 + 指标（含冗余度直方图）
POST /api/beamform                  多方法波束图扫描 + WNG/DI/主瓣宽度
POST /api/locate                    SRP-PHAT / MUSIC / TDOA 定位
POST /api/imaging                   脏图 + DAMAS/CLEAN-SC 反卷积
POST /api/enhance                   MVDR+后置滤波 / MWF / GSC + 音频输出
POST /api/export/cpp                异步导出 C++（返回 job_id）
GET  /api/jobs/<id>                 任务状态与 parity 结果
GET  /api/audio/<name>.wav          试听音频
```

参数同时接受**扁平写法**（脚本/curl 友好）与**嵌套写法**（前端 `reqBody`），由 `backend/params.py` 归一化。

---

## 已知限制

- **导出的 C++ 只含运行时应用**（几何 + 固化权重 + FFT/STFT/卷积），
  不含自适应设计（如 MVDR 的协方差估计）。要在目标平台重设计需一并移植
  `beamforming.lcmv_weights`。
- 房间模型是**鞋盒 + 镜像源法**，不含空气吸收与散射；高阶反射呈离散回声而非扩散尾。
- 阵列假设**全向麦克风**，不含指向性麦克风模型。
- `gsc` 不接受 `geom=` 参数，走内部解析设计（其固定波束本就是解析延迟求和，
  影响可忽略）。
- **`dsp.istft` 的窗端归一化**：BH 窗端点 `wsum` 比稳态小 3e7 倍，因此动过频谱后
  必须用 `center=True`，否则首尾各约 `n_fft` 个样点被置零（引擎已把
  `wsum <= 0.05*max` 的位置置零，避免放大 1e4 倍的数值噪声；C++ 侧同阈值）。
  详见 `docs/CONTRACT.md` 第 5.1 节。
- DAMAS 耗时 ∝ `n_iter × G²`，默认 40 次迭代；更大网格请显式指定 `n_iter`。
- 语音增强的可懂度指标是**代理指标**（基于频带 SNR 加权），不是 STOI。
