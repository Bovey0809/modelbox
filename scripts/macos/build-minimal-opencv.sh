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

# Build a minimal OpenCV that works with modelbox on macOS.
#
# Why a custom build:
#   * brew opencv 4.13 ships dylibs with unsubstituted @@HOMEBREW_PREFIX@@
#     placeholders in install_names on macOS 26 (Tahoe), breaking dlopen.
#   * brew's opencv pulls in libopencv_dnn, which links protobuf 34.x.
#     When modelbox-tool dlopens a cv-dependent flowunit at runtime,
#     opencv_dnn's protobuf static init crashes with a SIOF segfault in
#     EncodedDescriptorDatabase::Add.
#
# This script builds only the modules modelbox flowunits actually use
# (no dnn → no protobuf dep → no SIOF), with absolute install_names so
# install_name_tool isn't needed.
#
# Output: /opt/homebrew/Cellar/opencv-mb/<version>/{lib,include}.
# Set -DOpenCV_DIR=$(brew --prefix opencv-mb 2>/dev/null ||
#       echo /opt/homebrew/Cellar/opencv-mb/4.13.0)/lib/cmake/opencv4
# when configuring modelbox.

set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-4.13.0}"
PREFIX="${OPENCV_PREFIX:-/opt/homebrew/Cellar/opencv-mb/${OPENCV_VERSION}}"
SRC_DIR="${OPENCV_SRC:-/tmp/opencv-mb-src}"
BUILD_DIR="${OPENCV_BUILD:-/tmp/opencv-mb-build}"

# Modules modelbox flowunits transitively touch:
#   core      — cv::Mat, cv::Scalar
#   imgproc   — cv::resize, cv::rectangle, cv::putText, cv::cvtColor
#   imgcodecs — cv::imdecode (image_decoder flowunit)
#   videoio   — cv::VideoCapture (some demos)
#   highgui   — only the headers; no GUI windows used at runtime
#   calib3d, features2d, flann — pulled in transitively by imgproc
BUILD_LIST="core,imgproc,imgcodecs,videoio,highgui,calib3d,features2d,flann"

mkdir -p "${SRC_DIR}" "${BUILD_DIR}"

if [ ! -d "${SRC_DIR}/.git" ] && [ ! -f "${SRC_DIR}/CMakeLists.txt" ]; then
    echo "[opencv-mb] downloading OpenCV ${OPENCV_VERSION}..."
    curl -L -o "${SRC_DIR}.tar.gz" \
        "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz"
    rm -rf "${SRC_DIR}"
    mkdir -p "${SRC_DIR}"
    tar -xzf "${SRC_DIR}.tar.gz" -C "${SRC_DIR}" --strip-components=1
fi

cd "${BUILD_DIR}"

# Configure: minimal modules, no GUI, no contrib, no DNN/protobuf, absolute
# install names so dlopen finds libs without DYLD_LIBRARY_PATH.
cmake "${SRC_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_NAME_DIR="${PREFIX}/lib" \
    -DCMAKE_MACOSX_RPATH=ON \
    -DCMAKE_INSTALL_RPATH="${PREFIX}/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DBUILD_LIST="${BUILD_LIST}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JAVA=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENGL=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_PROTOBUF=OFF \
    -DWITH_TBB=OFF \
    -DWITH_OPENBLAS=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_QUIRC=OFF \
    -DWITH_ITT=OFF \
    -DWITH_OPENJPEG=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_OPENVINO=OFF \
    -DWITH_QT=OFF \
    -DWITH_V4L=OFF \
    -DPYTHON3_EXECUTABLE= \
    -DBUILD_opencv_python3=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build . -j"$(sysctl -n hw.ncpu)"
cmake --install .

# brew-style symlink so existing tooling that resolves $(brew --prefix opencv)
# can opt into this build by linking once.
mkdir -p /opt/homebrew/opt
ln -sfn "${PREFIX}" /opt/homebrew/opt/opencv-mb

echo
echo "[opencv-mb] built at ${PREFIX}"
echo "[opencv-mb] symlink: /opt/homebrew/opt/opencv-mb -> ${PREFIX}"
echo "[opencv-mb] modules: ${BUILD_LIST}"
echo
echo "Configure modelbox with:"
echo "  -DOpenCV_DIR=${PREFIX}/lib/cmake/opencv4"
