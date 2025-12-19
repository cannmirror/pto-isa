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
    H, W = 2, 32
    mask_cols = (W + 7) // 8
    src0 = np.random.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
    src1 = np.random.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
    mask = np.random.randint(0, 256, size=(H, mask_cols), dtype=np.uint8)

    golden = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            byte = c // 8
            bit = 1 << (c % 8)
            pick0 = (int(mask[r, byte]) & bit) != 0
            golden[r, c] = src0[r, c] if pick0 else src1[r, c]

    mask.tofile("input1.bin")
    src0.tofile("input2.bin")
    src1.tofile("input3.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "testcases"), exist_ok=True)

    case_name = "TSEL_Test.case_float_2x32"
    os.makedirs(case_name, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(case_name)
    gen_case()
    os.chdir(cwd)

