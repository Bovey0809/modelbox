#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(GRAPH_NAMES
    "tennis_action_cuda.toml"
    "tennis_action_cpu_stub.toml"
)
foreach(name ${GRAPH_NAMES})
    set(DEMO_GRAPH ${CMAKE_CURRENT_BINARY_DIR}/${name})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${name}.in ${DEMO_GRAPH} @ONLY)
    install(FILES ${DEMO_GRAPH}
        DESTINATION ${DEMO_TENNIS_ACTION_GRAPH_DIR}
        COMPONENT demo
    )
endforeach()
