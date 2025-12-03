#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

def gen_golden_data_trowmax(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [min(H, param.valid_row), min(W, param.valid_col)]

    # Generate random input array
    input1 = np.random.uniform(low=-16, high=16, size=[H, W]).astype(dtype)

    # Apply valid region constraints
    golden = np.full((h_valid), np.finfo(dtype).min, dtype=dtype)
    for i in range(h_valid):
        golden[i] = np.max(input1[i][:w_valid])

    golden =golden.astype(dtype)
    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    golden.tofile("golden.bin")

    return input1, golden

class trowmaxParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TROWMAXTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        trowmaxParams(np.float32, 64, 64, 64, 64, 64, 64),
        trowmaxParams(np.float16, 64, 64, 64, 64, 64, 64),
        trowmaxParams(np.float16, 161, 161, 32, 32, 161, 161),
        trowmaxParams(np.float32, 77, 81, 32, 16, 77, 81),
        trowmaxParams(np.float32, 32 ,32, 32, 16, 32, 32),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trowmax(case_name, param)
        os.chdir(original_dir)