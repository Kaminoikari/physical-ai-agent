#!/usr/bin/env bash
# 一鍵開啟 Meta-World 互動式 3D viewer。
# mjpython 需要 uv standalone Python 的 libpython dylib，用 DYLD 路徑指過去。
set -euo pipefail
cd "$(dirname "$0")"
export DYLD_FALLBACK_LIBRARY_PATH="/Users/charles/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib"
exec .venv/bin/mjpython week1_metaworld_viewer.py
