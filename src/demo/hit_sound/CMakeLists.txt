#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
cmake_minimum_required(VERSION 3.10)

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

set(DEMO_HIT_SOUND_DIR "${MODELBOX_DEMO_INSTALL_DIR}/hit_sound" CACHE INTERNAL "")
set(DEMO_HIT_SOUND_FLOWUNIT_DIR "${DEMO_HIT_SOUND_DIR}/flowunit" CACHE INTERNAL "")
set(DEMO_HIT_SOUND_GRAPH_DIR "${DEMO_HIT_SOUND_DIR}/graph" CACHE INTERNAL "")

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()
