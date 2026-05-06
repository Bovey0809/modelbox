#!/bin/bash
#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end runner for the apple_silicon_yolo demo.
#
# Prereqs (one-shot):
#   bash scripts/macos/build-minimal-opencv.sh
#   python3 -m venv /tmp/yolovenv && \
#     /tmp/yolovenv/bin/pip install ultralytics coremltools
#
# Usage:
#   bash scripts/macos/run-apple-silicon-yolo.sh /path/to/video.mp4
#
# Exits non-zero if the DAG fails to produce the result mp4.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/mbinstall}"
OPENCV_PREFIX="${OPENCV_PREFIX:-/opt/homebrew/Cellar/opencv-mb/4.13.0}"
VENV_PYTHON="${VENV_PYTHON:-/tmp/yolovenv/bin/python}"

VIDEO="${1:-${VIDEO:-/tmp/test_video.mp4}}"
RESULT_MP4="${RESULT_MP4:-/tmp/apple_silicon_yolo_result.mp4}"

if [ ! -f "${VIDEO}" ]; then
    echo "[run] video not found: ${VIDEO}" >&2
    exit 2
fi

echo "[run] configuring modelbox against ${OPENCV_PREFIX}..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
# Drop stale CMake cache when OpenCV_DIR changes, so cv-dependent flowunits
# relink against the right opencv. Without this they pick up the previous
# build's link line and reference a different OpenCV install.
if [ -f CMakeCache.txt ] && \
   ! grep -q "OpenCV_DIR.*${OPENCV_PREFIX}" CMakeCache.txt 2>/dev/null; then
    echo "[run] OpenCV_DIR changed — wiping cmake cache"
    rm -f CMakeCache.txt
    rm -rf CMakeFiles
fi
cmake "${REPO_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DWITH_ALL_DEMO=ON \
    -DWITH_WEBUI=OFF \
    -DWITH_JAVA=OFF \
    -DPYTHONE_DISABLED=ON \
    -DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" \
    -DOpenCV_DIR="${OPENCV_PREFIX}/lib/cmake/opencv4" \
    -DPYTHON_EXECUTABLE="${VENV_PYTHON}" >/dev/null

echo "[run] building..."
cmake --build . -j"$(sysctl -n hw.ncpu)" >/dev/null

echo "[run] installing to ${INSTALL_PREFIX}..."
DESTDIR="${INSTALL_PREFIX}" cmake --install . >/dev/null

GRAPH="${INSTALL_PREFIX}/usr/local/share/modelbox/demo/apple_silicon_yolo/graph/apple_silicon_yolo.toml"

# Patch graph paths so it works without root install
python3 - <<PY
import re, sys
p = "${GRAPH}"
with open(p) as f: s = f.read()
s = re.sub(r'dir = \[\s*"[^"]*apple_silicon_yolo/flowunit"\s*\]',
           f'dir = [\\n    "${INSTALL_PREFIX}/usr/local/lib",\\n    "${INSTALL_PREFIX}/usr/local/share/modelbox/demo/apple_silicon_yolo/flowunit"\\n]\\nskip-default = true', s)
s = s.replace('/opt/modelbox/demo/video/yolo26n_test_video.mp4', "${VIDEO}")
s = s.replace('/tmp/yolo26n_test_video.mp4', "${VIDEO}")
s = s.replace('default_dest_url="/tmp/apple_silicon_yolo_result.mp4"',
              f'default_dest_url="${RESULT_MP4}"')
with open(p, 'w') as f: f.write(s)
PY

echo "[run] graph: ${GRAPH}"
echo "[run] video: ${VIDEO}"
echo "[run] result: ${RESULT_MP4}"

mkdir -p "${INSTALL_PREFIX}/var/run" "${INSTALL_PREFIX}/var/log/modelbox"
rm -f "${RESULT_MP4}" "/tmp/modelbox-driver-info" \
      "/tmp/modelbox-driver-scan-info" \
      "${INSTALL_PREFIX}/var/run/modelbox-driver-info"

echo "[run] launching modelbox-tool..."
DYLD_LIBRARY_PATH="${INSTALL_PREFIX}/usr/local/lib" \
    "${INSTALL_PREFIX}/usr/local/bin/modelbox-tool" flow -run "${GRAPH}"
EXIT=$?

if [ ! -f "${RESULT_MP4}" ] || [ $EXIT -ne 0 ]; then
    echo "[run] FAILED — see ${INSTALL_PREFIX}/var/log/modelbox/modelbox-tool.log"
    tail -20 "${INSTALL_PREFIX}/var/log/modelbox/modelbox-tool.log" 2>/dev/null
    exit 1
fi

size=$(stat -f%z "${RESULT_MP4}")
echo "[run] OK — wrote ${RESULT_MP4} (${size} bytes)"
ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -show_entries format=duration -of default "${RESULT_MP4}" 2>&1 | grep -v '^$' | head -10
