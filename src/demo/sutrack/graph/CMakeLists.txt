#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(GRAPH_NAME "sutrack.toml")
set(DEMO_GRAPH ${CMAKE_CURRENT_BINARY_DIR}/${GRAPH_NAME})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${GRAPH_NAME}.in ${DEMO_GRAPH} @ONLY)

install(FILES ${DEMO_GRAPH}
    DESTINATION ${DEMO_SUTRACK_GRAPH_DIR}
    COMPONENT demo
)
