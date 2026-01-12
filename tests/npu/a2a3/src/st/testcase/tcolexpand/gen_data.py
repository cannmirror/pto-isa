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

np.random.seed(23)


def gen_golden_data(case_name, param):
    data_type = param.data_type
    dst_row = param.dst_row
    dst_col = param.dst_col
    dst_valid_row = param.dst_valid_row
    dst_valid_col = param.dst_valid_col

    input_arr = np.random.rand(1, dst_col) * 10
    input_arr = input_arr.astype(data_type)
    golden = np.zeros((dst_row, dst_col), dtype=data_type)
    for i in range(dst_valid_row):
        for j in range(dst_valid_col):
            golden[i, j] = input_arr[0, j]
    input_arr.tofile('input.bin')
    golden.tofile('golden.bin')


class TColExpand:
    def __init__(self, data_type, dst_row, dst_col, dst_valid_row, dst_valid_col):
        self.data_type = data_type
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.dst_valid_row = dst_valid_row
        self.dst_valid_col = dst_valid_col

if __name__ == "__main__":
    
    case_name_list = [
        "TCOLEXPANDTest.case1",
        "TCOLEXPANDTest.case2",
        "TCOLEXPANDTest.case3",
        "TCOLEXPANDTest.case4",
        "TCOLEXPANDTest.case5",
        "TCOLEXPANDTest.case6",
        "TCOLEXPANDTest.case7",
        "TCOLEXPANDTest.case8",
        "TCOLEXPANDTest.case9",
        "TCOLEXPANDTest.case10",
        "TCOLEXPANDTest.case11",
        "TCOLEXPANDTest.case12",
    ]
    
    case_params_list = [
        TColExpand(np.int16, 32, 32, 16, 8),
        TColExpand(np.int32, 24, 16, 16, 8),
        TColExpand(np.float32, 24, 16, 16, 8),
        TColExpand(np.int16, 16, 128, 8, 127),
        TColExpand(np.int32, 16, 64, 15, 63),
        TColExpand(np.float32, 16, 64, 15, 63),
        TColExpand(np.int16, 12, 256, 6, 256),
        TColExpand(np.int32, 16, 256, 15, 256),
        TColExpand(np.float32, 16, 64, 15, 64),
        TColExpand(np.int16, 16, 256, 7, 255),
        TColExpand(np.int32, 32, 256, 31, 255),
        TColExpand(np.float32, 1, 64, 1, 63),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)