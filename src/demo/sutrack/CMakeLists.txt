#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

set(DEMO_SUTRACK_DIR "${MODELBOX_DEMO_INSTALL_DIR}/sutrack" CACHE INTERNAL "")
set(DEMO_SUTRACK_FLOWUNIT_DIR ${DEMO_SUTRACK_DIR}/flowunit)
set(DEMO_SUTRACK_GRAPH_DIR ${DEMO_SUTRACK_DIR}/graph)

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()
