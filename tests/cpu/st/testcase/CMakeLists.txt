# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_cpu_sim_st NAME)
    set(FILES_LIST main.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_kernel.cpp)
        list(APPEND FILES_LIST ${NAME}_kernel.cpp)
    endif()
    add_executable(${NAME} ${FILES_LIST})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -std=c++23)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../include/
        ${PROJECT_SOURCE_DIR}/../../../include/common
        ${PROJECT_SOURCE_DIR}/../stubs
        ${PROJECT_SOURCE_DIR}/../../common
    )

    target_link_libraries(${NAME} PRIVATE
        stdc++ m dl GTest::gtest GTest::gtest_main pthread
    )
endfunction()

set(ALL_TESTCASES
tmatmul
tabs
tload
tstore
textract
texpands
tmov
trowsum
tadd
texp
trowmax
tsqrt
tfillpad
ttrans
tsels
trem
tshl
tshr
tand
tor
txor
tmin
tcmp
tlog
tneg
tnot
trecip
trelu
tprelu
taddc
tsubc
tsubs
trems
tmaxs
tands
tors
txors
tlrelu
taddsc
tsubsc
)

foreach(TESTCASE ${ALL_TESTCASES})
if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE))
    add_subdirectory(${TESTCASE})
endif()
endforeach()
