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


def gen_case(case_dir: str, rows: int, cols: int):
    os.makedirs(case_dir, exist_ok=True)
    os.chdir(case_dir)

    src = np.random.uniform(low=-4, high=4, size=[rows, cols]).astype(np.float32)
    idx = np.random.randint(0, rows, size=[rows, cols]).astype(np.uint16)

    dst = np.zeros([rows, cols], dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            dst[idx[i, j], j] = src[i, j]

    src.tofile("input1.bin")
    idx.tofile("input2.bin")
    dst.tofile("golden.bin")
    os.chdir("..")


if __name__ == "__main__":
    gen_case("TSCATTERTest.case_float_16x16_16x16_16x16", 16, 16)

