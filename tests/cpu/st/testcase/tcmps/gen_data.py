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


def gen_case(case_name: str):
    H, W = 64, 64
    scalar = np.float32(5.0)
    x = np.random.uniform(-2.0, 8.0, size=(H, W)).astype(np.float32)
    golden = (x > scalar).astype(np.float32)
    x.tofile("input1.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")
    os.makedirs(testcases_dir, exist_ok=True)

    case_name = "TCMPS_Test.case_float_64x64_scalar5_gt"
    os.makedirs(case_name, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(case_name)
    gen_case(case_name)
    os.chdir(cwd)

