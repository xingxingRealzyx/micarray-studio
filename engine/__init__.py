"""MicArray Studio —— 麦克风阵列仿真引擎。

纯 numpy 实现，无第三方依赖（不需要 scipy / matplotlib）。
对外接口约定见 docs/CONTRACT.md。

模块一览：

==================  ==================================================
geometry            阵型构建与指标（含螺旋臂 / 悬臂 / Vogel / 嵌套）
acoustics           几何声学：时延、球面扩散、近场判据、无混叠上限
steering            导向矢量（远场/近场）、扩散场相干矩阵
dsp                 窗、STFT/ISTFT、独立 DFT（用于校验 FFT）
tdoa                GCC-PHAT、多源时延聚类
signals             信号源、噪声、通道失配、评价指标
sim                 声场仿真：镜像源法 RIR、房间、失配、噪声
beamforming         DS / MVDR / 超指向 / 频不变 + WNG / DI
localization        SRP-PHAT / MUSIC / TDOA / 近场距离
imaging             PSF / 脏图 / DAMAS / CLEAN-SC
enhance             MVDR+后置滤波 / 多通道维纳 / GSC / 谱减
codegen             C++17 导出 + 编译 + 数值一致性比对
==================  ==================================================

注意：本包暴露的全部角度都在**阵列坐标系**中 —— 阵元位于 xy 平面、
boresight = +z、俯仰角从 xy 平面量向 +z。详见 CONTRACT 第 1、2 节。
"""

__version__ = "0.1.0"

from . import acoustics, dsp, geometry, steering, tdoa  # noqa: F401

# 以下几个模块体积较大，且依赖上面的基础模块；用惰性导入以免
# `import engine` 就付出全部导入开销（GUI 启动时会感知到）。
_LAZY = ("signals", "sim", "beamforming", "localization", "imaging",
         "enhance", "codegen")


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))


__all__ = ["acoustics", "dsp", "geometry", "steering", "tdoa", *_LAZY,
           "__version__"]
