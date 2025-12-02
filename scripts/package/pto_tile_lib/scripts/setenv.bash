#!/bin/bash
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

param_mult_ver=$1
REAL_SHELL_PATH=`realpath ${BASH_SOURCE[0]}`
CANN_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../ && pwd)
if [ -d "${CANN_PATH}/include/pto" ] && [ -d "${CANN_PATH}/../cann" ]; then
    INSATLL_PATH=$(cd $(dirname ${REAL_SHELL_PATH})/../../../ && pwd)
    if [ -L "${INSATLL_PATH}/cann/include/pto" ]; then
        _ASCEND_PTO_TILE_LIB_PATH=`cd ${CANN_PATH}/include/pto && pwd`
        if [ "$param_mult_ver" = "multi_version" ]; then
            _ASCEND_PTO_TILE_LIB_PATH=`cd ${INSATLL_PATH}/cann/include/pto && pwd`
        fi
    fi
elif [ -d "${CANN_PATH}/include/pto" ]; then
    _ASCEND_PTO_TILE_LIB_PATH=`cd ${CANN_PATH}/include/pto && pwd`
fi  

export ASCEND_PTO_TILE_LIB_PATH=${_ASCEND_PTO_TILE_LIB_PATH}

