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
    data_type = param.data_type
    m, k, n, p, q = param.row, param.src_col, param.src_validcol, param.dst_col, param.dst_validcol
    input_arr = np.random.rand(m, k) * 10
    input_arr = input_arr.astype(data_type)
    golden = np.zeros((m, p))
    for i in range(m):
        for j in range(q):
            golden[i][j] = input_arr[i][0]
    golden = golden.astype(data_type)
    input_arr.tofile("./input.bin")
    golden.tofile("./golden.bin")


class TRowExpand:
    def __init__(self, data_type, row, src_col, src_validcol, dst_col, dst_validcol):
        self.data_type = data_type
        self.row = row
        self.src_col = src_col
        self.src_validcol = src_validcol
        self.dst_col = dst_col
        self.dst_validcol = dst_validcol


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TROWEXPANDTest.case0",
        "TROWEXPANDTest.case1",
        "TROWEXPANDTest.case2",
        "TROWEXPANDTest.case3",
        "TROWEXPANDTest.case4",
        "TROWEXPANDTest.case5",
        "TROWEXPANDTest.case6",
        "TROWEXPANDTest.case7",
    ]

    case_params_list = [
        TRowExpand(np.uint16, 16, 16, 16, 512, 512),
        TRowExpand(np.uint8, 16, 32, 32, 256, 256),
        TRowExpand(np.uint32, 16, 8, 8, 128, 128),
        TRowExpand(np.float32, 16, 32, 32, 512, 512),
        TRowExpand(np.uint16, 16, 16, 1, 256, 255),
        TRowExpand(np.uint8, 16, 32, 1, 512, 511),
        TRowExpand(np.uint32, 16, 8, 1, 128, 127),
        TRowExpand(np.uint16, 16, 8, 1, 128, 127),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)