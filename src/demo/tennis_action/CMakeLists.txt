#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

subdirlist(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} "CMakeLists.txt")

set(DEMO_TENNIS_ACTION_DIR "${MODELBOX_DEMO_INSTALL_DIR}/tennis_action" CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_FLOWUNIT_DIR "${DEMO_TENNIS_ACTION_DIR}/flowunit" CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_GRAPH_DIR "${DEMO_TENNIS_ACTION_DIR}/graph" CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_ASSETS_DIR "${DEMO_TENNIS_ACTION_DIR}/assets" CACHE INTERNAL "")
set(DEMO_TENNIS_ACTION_VIDEO "${DEMO_TENNIS_ACTION_DIR}/test/tennis_test.mp4" CACHE INTERNAL "")

# Install the test video alongside the demo so the graph TOML works post-install.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/tennis_test.mp4
    DESTINATION ${DEMO_TENNIS_ACTION_DIR}/test
    COMPONENT demo
)

foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()
