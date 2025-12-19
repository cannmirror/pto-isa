#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(19)


def gen_case():
    H, W = 64, 64
    x = np.random.uniform(-2.0, 2.0, size=(H, W)).astype(np.float32)
    golden = np.sum(x, axis=0).astype(np.float32)
    x.tofile("input.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "testcases"), exist_ok=True)

    case_name = "TCOLSUM_Test.case_float_64x64"
    os.makedirs(case_name, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(case_name)
    gen_case()
    os.chdir(cwd)
