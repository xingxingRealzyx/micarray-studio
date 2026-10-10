"""MicArray Studio 后端包：仿真引擎 + Web API。

引擎位于顶层 engine/ 包（纯 numpy，可独立于后端使用）。
本文件把它加入搜索路径，使 `import engine` 与 `from backend import ...` 都能工作。
"""
from __future__ import annotations

import pathlib as _pathlib
import sys as _sys

_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

__all__ = []
