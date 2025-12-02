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
    datatype = param.datatype

    m, k, n = param.row, param.src_col, param.dst_col

    input = np.random.rand(m, k) * 10
    golden = np.repeat(input[:, 0], n, axis=0)

    input = input.astype(datatype)


    golden = golden.astype(datatype)

    input.tofile("./input.bin")
    golden.tofile("./golden.bin")


class TRowExpand:
    def __init__(self, datatype, row, src_col, dst_col):
        self.datatype = datatype
        self.row = row
        self.src_col = src_col
        self.dst_col = dst_col


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TROWEXPANDTest.case0",
        "TROWEXPANDTest.case1",
        "TROWEXPANDTest.case2",
        "TROWEXPANDTest.case3",
        "TROWEXPANDTest.case4",
    ]

    case_params_list = [
        TRowExpand(np.uint16, 16 , 16, 512),
        TRowExpand(np.uint8,    16 , 32, 256),
        TRowExpand(np.uint32,   16 , 8,  128),
        TRowExpand(np.float32, 16 , 32, 512),
        TRowExpand(np.uint16,    16, 1, 255),  # valid_cols = 255
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
