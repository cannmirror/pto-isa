#!/usr/bin/fish
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set param_mult_ver $argv[1]
set REAL_SHELL_PATH (realpath (command -v $argv[0]))
set CANN_PATH (cd (dirname $REAL_SHELL_PATH)/../../ && pwd)
if test -d "$CANN_PATH/include/pto" -a test -d "$CANN_PATH/../cann"
    set INSATLL_PATH (cd (dirname $REAL_SHELL_PATH)/../../../ && pwd)
elseif test -d "$CANN_PATH/include/pto"
    set _ASCEND_PTO_TILE_LIB_PATH (cd $CANN_PATH/include/pto && pwd)
end

set -x ASCEND_PTO_TILE_LIB_PATH $_ASCEND_PTO_TILE_LIB_PATH

