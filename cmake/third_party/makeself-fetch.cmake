# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(MAKESELF_NAME "makeself")
if(NOT DEFINED CANN_3RD_LIB_PATH OR CANN_3RD_LIB_PATH STREQUAL "")
    set(CANN_3RD_LIB_PATH "${CMAKE_SOURCE_DIR}/third_party")
endif()
set(MAKESELF_PATH "${CANN_3RD_LIB_PATH}/makeself")

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
    message(STATUS "Downloading ${MAKESELF_NAME} from ${MAKESELF_URL}")

    set(_makeself_download_args)
    if(DEFINED CANN_3RD_PKG_PATH AND NOT CANN_3RD_PKG_PATH STREQUAL "")
        list(APPEND _makeself_download_args DOWNLOAD_DIR "${CANN_3RD_PKG_PATH}")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${MAKESELF_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        ${_makeself_download_args}
        SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})

    if(UNIX)
        if(EXISTS "${MAKESELF_PATH}/makeself.sh")
            execute_process(
                COMMAND chmod 700 "${MAKESELF_PATH}/makeself.sh"
                RESULT_VARIABLE CHMOD_RESULT
                ERROR_VARIABLE CHMOD_ERROR
            )
            if(NOT CHMOD_RESULT EQUAL 0)
                message(FATAL_ERROR "chmod failed for ${MAKESELF_PATH}/makeself.sh: ${CHMOD_ERROR}")
            endif()
        endif()

        if(EXISTS "${MAKESELF_PATH}/makeself-header.sh")
            execute_process(
                COMMAND chmod 700 "${MAKESELF_PATH}/makeself-header.sh"
                RESULT_VARIABLE CHMOD_RESULT
                ERROR_VARIABLE CHMOD_ERROR
            )
            if(NOT CHMOD_RESULT EQUAL 0)
                message(FATAL_ERROR "chmod failed for ${MAKESELF_PATH}/makeself-header.sh: ${CHMOD_ERROR}")
            endif()
        endif()
    endif()
endif()
