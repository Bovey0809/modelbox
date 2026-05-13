#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

set(GRAPH_NAMES
    "hit_sound.toml"           # monolithic python flowunit (onnxruntime via Python)
    "hit_sound_openvino.toml"  # split: emitter + OpenVINO inference + postprocess
    "hit_sound_ort.toml"       # split: emitter + ORT inference driver + postprocess
)
if(APPLE)
    list(APPEND GRAPH_NAMES "hit_sound_coreml.toml")
    list(APPEND GRAPH_NAMES "hit_sound_coreml_native.toml")
endif()

foreach(name ${GRAPH_NAMES})
    set(DEMO_GRAPH ${CMAKE_CURRENT_BINARY_DIR}/${name})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${name}.in ${DEMO_GRAPH} @ONLY)
    install(FILES ${DEMO_GRAPH}
        DESTINATION ${DEMO_HIT_SOUND_GRAPH_DIR}
        COMPONENT demo
    )
endforeach()
