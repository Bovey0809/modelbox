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


cmake_minimum_required(VERSION 3.10)

# One graph per Ultralytics task. The base detection graph is
# apple_silicon_yolo.toml; the others reuse the same upstream pipeline
# but route through a task-specific yolo_*_post flowunit and (where the
# model differs) a task-specific yolo_*_detect inference flowunit.
foreach(GRAPH_BASE
        apple_silicon_yolo
        apple_silicon_yolo_obb
        apple_silicon_yolo_pose
        apple_silicon_yolo_seg
        apple_silicon_yolo_cls
        apple_silicon_yolo_track)
    set(GRAPH_NAME "${GRAPH_BASE}.toml")
    set(DEMO_GRAPH ${CMAKE_CURRENT_BINARY_DIR}/${GRAPH_NAME})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${GRAPH_NAME}.in ${DEMO_GRAPH})

    install(FILES
        ${DEMO_GRAPH} DESTINATION ${DEMO_APPLE_SILICON_YOLO_GRAPH_DIR}
        COMPONENT demo
    )
endforeach()
