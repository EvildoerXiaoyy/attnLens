"""
pytest 配置与共享 fixtures（L1 修复：集中 sys.path 管理）
"""
import sys
from pathlib import Path

# 将 src 和 tests/mocks 加入导入路径
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
MOCKS_DIR = Path(__file__).resolve().parent / "mocks"
for p in [SRC_DIR, MOCKS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
