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

PAD_VALUE_NULL = "PAD_VALUE_NULL"
PAD_VALUE_MAX = "PAD_VALUE_MAX"
PAD_VALUE_MIN = "PAD_VALUE_MIN"

def gen_golden_data(case_name, param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = [param.dst_tile_row, param.dst_tile_col]
    src0_tile_row, src0_tile_col = [param.src0_tile_row, param.src0_tile_col]
    src1_tile_row, src1_tile_col = [param.src1_tile_row, param.src1_tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1, input2 = [], []
    if dtype == np.int16:
        input1 = np.random.randint(-30_000, 30_000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
        input2 = np.random.randint(-30_000, 30_000, size=[src1_tile_row, src1_tile_col]).astype(dtype)
    elif dtype == np.int32:
        input1 = np.random.randint(-2_000_000_000, 2_000_000_000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
        input2 = np.random.randint(-2_000_000_000, 2_000_000_000, size=[src1_tile_row, src1_tile_col]).astype(dtype)
    elif dtype == np.float16:
        input1 = np.random.uniform(-30_000, 30_000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
        input2 = np.random.uniform(-30_000, 30_000, size=[src1_tile_row, src1_tile_col]).astype(dtype)
    elif dtype == np.float32:
        input1 = np.random.uniform(-2_000_000_000, 2_000_000_000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
        input2 = np.random.uniform(-2_000_000_000, 2_000_000_000, size=[src1_tile_row, src1_tile_col]).astype(dtype)

    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    for h in range(h_valid):
        for w in range(w_valid):
            golden[h][w] = max(input1[h][w], input2[h][w])
    golden = np.array(golden).astype(dtype)
    output = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)

    for h in range(dst_tile_row):
        for w in range(dst_tile_col):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden

class testParams:
    def __init__(self, dtype, dst_tile_row, dst_tile_col, src0_tile_row, src0_tile_col, src1_tile_row, src1_tile_col, 
                 valid_row, valid_col, pad_value_type=PAD_VALUE_NULL):
        self.dtype = dtype
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.src0_tile_row = src0_tile_row
        self.src0_tile_col = src0_tile_col
        self.src1_tile_row = src1_tile_row
        self.src1_tile_col = src1_tile_col 
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.pad_value_type = pad_value_type


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TMAXPLUSTest.case_{dtype_str}_{param.valid_row}x{param.valid_col}_{param.pad_value_type}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        testParams(np.float32, 64, 64, 64, 64, 64, 64, 64, 64),
        testParams(np.int32, 64, 64, 64, 64, 64, 64, 64, 64),
        testParams(np.int16, 64, 64, 64, 64, 64, 64, 64, 64),
        testParams(np.float16, 64, 64, 64, 64, 64, 64, 64, 64),
        testParams(np.float32, 60, 128, 64, 64, 60, 128, 60, 60, PAD_VALUE_MAX),
        testParams(np.int32, 60, 128, 64, 64, 60, 128, 60, 60, PAD_VALUE_MAX),
        testParams(np.float16, 1, 3600, 2, 4096, 1, 3600, 1, 3600, PAD_VALUE_MAX),
        testParams(np.int16, 16, 256, 20, 512, 16, 256, 16, 200, PAD_VALUE_MAX),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)