#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(FLOWUNIT_NAME "action_sink")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.py
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.py @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_TENNIS_ACTION_FLOWUNIT_DIR}
    COMPONENT demo
)
