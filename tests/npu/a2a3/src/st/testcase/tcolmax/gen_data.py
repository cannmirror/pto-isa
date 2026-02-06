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

np.random.seed(23)


def gen_golden_data(case_name, param):
    data_type = param.data_type
    cols = param.col
    src_row = param.src_row
    src_valid_row = param.src_valid_row

    input_arr = np.random.rand(src_valid_row, cols) * 10
    input_arr = input_arr.astype(data_type)
    golden = np.zeros((1, cols), dtype=data_type)
    golden[0] = np.max(input_arr, axis=0)
    input_arr.tofile('input.bin')
    golden.tofile('golden.bin')


class TColMax:
    def __init__(self, data_type, col, src_row, src_valid_row):
        self.data_type = data_type
        self.col = col
        self.src_row = src_row
        self.src_valid_row = src_valid_row

if __name__ == "__main__":
    
    case_name_list = [
        "TCOLMAXTest.case1",
        "TCOLMAXTest.case2",
        "TCOLMAXTest.case3",
        "TCOLMAXTest.case4",
        "TCOLMAXTest.case5",
        "TCOLMAXTest.case6",
        "TCOLMAXTest.case7",
        "TCOLMAXTest.case8",
        "TCOLMAXTest.case9",
        "TCOLMAXTest.case10",
        "TCOLMAXTest.case11",
        "TCOLMAXTest.case12",
        "TCOLMAXTest.case13",
    ]
    
    case_params_list = [
        TColMax(np.int16, 16, 16, 8),
        TColMax(np.int32, 16, 16, 8),
        TColMax(np.float32, 16, 16, 8),
        TColMax(np.int16, 128, 16, 8),
        TColMax(np.int32, 64, 16, 8),
        TColMax(np.float32, 64, 16, 8),
        TColMax(np.int16, 512, 16, 8),
        TColMax(np.int32, 256, 16, 8),
        TColMax(np.float32, 256, 16, 8),
        TColMax(np.int16, 512, 16, 7),
        TColMax(np.int32, 256, 32, 31),
        TColMax(np.float32, 256, 32, 31),
        TColMax(np.float32, 256, 16, 1),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)