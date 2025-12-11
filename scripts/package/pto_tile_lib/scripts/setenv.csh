#!/bin/csh
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set param_mult_ver = $argv[1]
set REAL_SHELL_PATH = `realpath $0`
set CANN_PATH = `cd $(dirname $REAL_SHELL_PATH)/../../ && pwd`
if (-d "$CANN_PATH/include/pto" && -d "$CANN_PATH/../cann") then
    set INSATLL_PATH = `cd $(dirname $REAL_SHELL_PATH)/../../../ && pwd`
elseif (-d "$CANN_PATH/include/pto") then
    set _ASCEND_PTO_TILE_LIB_PATH = `cd $CANN_PATH/include/pto && pwd`
endif

setenv ASCEND_PTO_TILE_LIB_PATH ${_ASCEND_PTO_TILE_LIB_PATH}

