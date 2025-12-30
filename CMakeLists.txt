# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/version.info" _pto_isa_version_line REGEX "^Version=")
if(_pto_isa_version_line)
  string(REPLACE "Version=" "" PTO_ISA_VERSION "${_pto_isa_version_line}")
else()
  set(PTO_ISA_VERSION "0.0.0")
endif()

project(pto_isa VERSION "${PTO_ISA_VERSION}" LANGUAGES NONE)

set(CANN_3RD_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/third_party"
    CACHE PATH "Third-party dependency directory (e.g. makeself).")

include(cmake/package.cmake)
pack_built_in()
