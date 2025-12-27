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
project(pto_docs LANGUAGES NONE)

option(PTO_DOCS_BUILD_ON_ALL "Build the MkDocs site as part of the default build" OFF)
option(PTO_DOCS_BOOTSTRAP_VENV "Automatically create a MkDocs virtualenv under the build directory" ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PTO_MKDOCS_CONFIG "${CMAKE_CURRENT_LIST_DIR}/mkdocs/mkdocs.yml")
set(PTO_DOCS_SITE_DIR "${CMAKE_BINARY_DIR}/site")
set(PTO_MKDOCS_CHECK "${CMAKE_CURRENT_LIST_DIR}/mkdocs/check_mkdocs.py")
set(PTO_MKDOCS_REQUIREMENTS "${CMAKE_CURRENT_LIST_DIR}/mkdocs/requirements.txt")

set(PTO_DOCS_PYTHON "${Python3_EXECUTABLE}")
set(PTO_DOCS_CHECK_CMD "${Python3_EXECUTABLE}" "${PTO_MKDOCS_CHECK}")

if(PTO_DOCS_BOOTSTRAP_VENV)
  set(PTO_DOCS_VENV_DIR "${CMAKE_BINARY_DIR}/mkdocs-venv")
  if(WIN32)
    set(PTO_DOCS_PYTHON "${PTO_DOCS_VENV_DIR}/Scripts/python.exe")
  else()
    set(PTO_DOCS_PYTHON "${PTO_DOCS_VENV_DIR}/bin/python")
  endif()
  set(PTO_DOCS_CHECK_CMD "${Python3_EXECUTABLE}" "${PTO_MKDOCS_CHECK}" --ensure-venv "${PTO_DOCS_VENV_DIR}")
endif()

set(_pto_docs_all)
if(PTO_DOCS_BUILD_ON_ALL)
  set(_pto_docs_all ALL)
endif()

add_custom_target(pto_docs_prepare
  COMMAND ${PTO_DOCS_CHECK_CMD}
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  COMMENT "Checking/bootstrapping MkDocs environment (requirements: ${PTO_MKDOCS_REQUIREMENTS})"
  VERBATIM
)

add_custom_target(pto_docs ${_pto_docs_all}
  DEPENDS pto_docs_prepare
  COMMAND "${PTO_DOCS_PYTHON}" -m mkdocs build
          -f "${PTO_MKDOCS_CONFIG}"
          -d "${PTO_DOCS_SITE_DIR}"
          --clean
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  COMMENT "Building MkDocs static site into ${PTO_DOCS_SITE_DIR}"
  VERBATIM
)

# Convenience target for local preview (non-blocking builds usually prefer running this manually).
add_custom_target(pto_docs_serve
  DEPENDS pto_docs_prepare
  COMMAND "${PTO_DOCS_PYTHON}" -m mkdocs serve
          -f "${PTO_MKDOCS_CONFIG}"
          -a "127.0.0.1:8000"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  COMMENT "Serving MkDocs site at http://127.0.0.1:8000/"
  VERBATIM
)
