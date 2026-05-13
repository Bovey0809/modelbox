#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
cmake_minimum_required(VERSION 3.10)

# CoreML is Apple-only; skip on Linux builds.
if(NOT APPLE)
    return()
endif()

set(FLOWUNIT_NAME "hit_sound_infer_coreml")
set(FLOWUNIT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${FLOWUNIT_NAME})
file(MAKE_DIRECTORY ${FLOWUNIT_PATH})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FLOWUNIT_NAME}.toml
               ${FLOWUNIT_PATH}/${FLOWUNIT_NAME}.toml @ONLY)

set(HIT_SOUND_MLPACKAGE_SRC "${CMAKE_SOURCE_DIR}/assets/hit_sound/best.mlpackage")
set(HIT_SOUND_MLPACKAGE_DST "${FLOWUNIT_PATH}/best.mlpackage")
set(HIT_SOUND_MLPACKAGE_STAMP_SRC "${HIT_SOUND_MLPACKAGE_SRC}/Manifest.json")
set(HIT_SOUND_MLPACKAGE_STAMP_DST "${HIT_SOUND_MLPACKAGE_DST}/Manifest.json")

if(EXISTS ${HIT_SOUND_MLPACKAGE_STAMP_SRC})
    add_custom_command(
        OUTPUT  ${HIT_SOUND_MLPACKAGE_STAMP_DST}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${HIT_SOUND_MLPACKAGE_SRC} ${HIT_SOUND_MLPACKAGE_DST}
        DEPENDS ${HIT_SOUND_MLPACKAGE_STAMP_SRC}
        VERBATIM
    )
    add_custom_target(${FLOWUNIT_NAME}_model ALL DEPENDS ${HIT_SOUND_MLPACKAGE_STAMP_DST})
else()
    message(WARNING
        "hit_sound_infer_coreml: missing .mlpackage at ${HIT_SOUND_MLPACKAGE_SRC}. "
        "Run models/tennis_audio_seg/deploy/export_coreml.py before configuring.")
endif()

install(DIRECTORY ${FLOWUNIT_PATH}
    DESTINATION ${DEMO_HIT_SOUND_FLOWUNIT_DIR}
    COMPONENT demo USE_SOURCE_PERMISSIONS
)
