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
import struct
import ctypes
import numpy as np
np.random.seed(2025)


def gen_golden_data(case_name, param):
    dtype = param.datatype

    vr, vc = param.valid_row, param.valid_col

    input1 = np.random.random(vr * vc).astype(dtype)
    input2 = np.random.random(vr).astype(dtype)
    golden = np.zeros(vr * vc).astype(dtype)

    for i in range(vr):
        for j in range(vc):
            golden[i * vc + j] = input1[i * vc + j] - input2[i]

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TRowExpandSub:
    def __init__(self, datatype, valid_row, valid_col, row, col):
        self.datatype = datatype
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.row = row
        self.col = col


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TROWEXPANDSUBTest.case1",
        "TROWEXPANDSUBTest.case2",
        "TROWEXPANDSUBTest.case3",
        "TROWEXPANDSUBTest.case4",
        "TROWEXPANDSUBTest.case5",
        "TROWEXPANDSUBTest.case6",
    ]

    case_params_list = [
        TRowExpandSub(np.float32, 16, 16, 16, 16),
        TRowExpandSub(np.float32, 16, 16, 32, 32),
        TRowExpandSub(np.float16, 16, 16, 16, 16),
        TRowExpandSub(np.float16, 16, 16, 32, 32),
        TRowExpandSub(np.float32, 1, 16384, 1, 16384),
        TRowExpandSub(np.float32, 2048, 1, 2048, 8),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
