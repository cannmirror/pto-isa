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


def gen_golden_data_tsel(param):
    dtype = param.dtype

    row, col = [param.validRows, param.validCols]
    maskCol = (col+7)//8

    output = np.zeros(row*col).astype(dtype)

    input0 = np.random.rand(row*col).astype(dtype)
    input1 = np.random.rand(row*col).astype(dtype)
    mask_size = row*maskCol
    mask = np.random.randint(0, 255, size = mask_size, dtype=np.uint8)
    golden = np.zeros(row*col).astype(dtype)

    for i in range(0, row):
        for j in range(0, maskCol):
            byte = mask[i*maskCol + j]
            for k in range(8):
                bit = (byte >> k) & 1
                idx = i * col + j * 8 + k
                if j * 8 + k < col:
                    if bit == 1:
                        golden[idx] = input0[idx]
                    else:
                        golden[idx] = input1[idx]
    
    input0.tofile("input0.bin")
    input1.tofile("input1.bin")
    mask.tofile("mask.bin")
    golden.tofile("golden.bin")

    return output, input0, input1, golden

class tselParams:
    def __init__(self, name, dtype, rows, cols, validRows, validCols):
        self.name = name
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.validRows = validRows
        self.validCols = validCols

if __name__ == "__main__":
    # Get the absolute path of the script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    # if not os.path.exists(testcases_dir):
    #     os.makedirs(testcases_dir)

    case_params_list = [
        tselParams("TSELTest.case1", np.float32, 2, 128, 2, 128),
        tselParams("TSELTest.case2", np.float32, 2, 32, 2, 32),
        tselParams("TSELTest.case3", np.float32, 2, 160, 2, 160),
        tselParams("TSELTest.case4", np.float16, 2, 128, 2, 128),
        tselParams("TSELTest.case5", np.float16, 2, 32, 2, 32),
        tselParams("TSELTest.case6", np.float16, 2, 160, 2, 160),
        tselParams("TSELTest.case7", np.float32, 10, 64, 10, 54),
        tselParams("TSELTest.case8", np.float32, 2, 4096, 2, 4096),
        tselParams("TSELTest.case9", np.float32, 1024, 8, 1024, 8),
        tselParams("TSELTest.case10", np.int32, 2, 128, 2, 128),
        tselParams("TSELTest.case11", np.int16, 2, 128, 2, 128),
        tselParams("TSELTest.case12", np.float32, 2, 8, 2, 8),
        tselParams("TSELTest.case13", np.float16, 2, 16, 2, 16),
    ]

    for i, param in enumerate(case_params_list):
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsel(param)
        os.chdir(original_dir)