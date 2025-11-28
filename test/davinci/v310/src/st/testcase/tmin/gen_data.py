#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

PAD_VALUE_NULL = "PAD_VALUE_NULL"
PAD_VALUE_MAX = "PAD_VALUE_MAX"
PAD_VALUE_MIN = "PAD_VALUE_MIN"

def gen_golden_data(case_name, param):
    dtype = param.dtype

    H, W = [param.global_row, param.global_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1, input2 = [], []
    if dtype == np.int16:
        input1 = np.random.randint(-30_000, 30_000, size=[H, W]).astype(dtype)
        input2 = np.random.randint(-30_000, 30_000, size=[H, W]).astype(dtype)
    elif dtype == np.int32:
        input1 = np.random.randint(-2_000_000_000, 2_000_000_000, size=[H, W]).astype(dtype)
        input2 = np.random.randint(-2_000_000_000, 2_000_000_000, size=[H, W]).astype(dtype)
    elif dtype == np.float16:
        input1 = np.random.uniform(-30_000, 30_000, size=[H, W]).astype(dtype)
        input2 = np.random.uniform(-30_000, 30_000, size=[H, W]).astype(dtype)
    elif dtype == np.float32:
        input1 = np.random.uniform(-2_000_000_000, 2_000_000_000, size=[H, W]).astype(dtype)
        input2 = np.random.uniform(-2_000_000_000, 2_000_000_000, size=[H, W]).astype(dtype)

    golden = np.minimum(input1, input2)

    output = np.zeros([H, W]).astype(dtype)
    for h in range(H):
        for w in range(W):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]
                input1[h][w] = output[h][w]
                input2[h][w] = output[h][w]
    
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")
    
    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden

class testParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col, pad_value_type=PAD_VALUE_NULL):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
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
    return f"TMINTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}_{param.pad_value_type}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        testParams(np.float32, 64, 64, 64, 64, 64, 64),
        testParams(np.int32, 64, 64, 64, 64, 64, 64),
        testParams(np.int16, 64, 64, 64, 64, 64, 64),
        testParams(np.float16, 64, 64, 64, 64, 64, 64),

        testParams(np.float32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MIN),
        testParams(np.int32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MIN),

        testParams(np.float16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MIN),
        testParams(np.int16, 16, 200, 20, 512, 16, 200, PAD_VALUE_MIN),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)