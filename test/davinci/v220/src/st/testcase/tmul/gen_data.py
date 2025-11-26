#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

def gen_golden_data_tmul(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[H, W]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[H, W]).astype(dtype)

    golden = np.multiply(input1, input2)

    # Apply valid region constraints
    output = np.zeros([H, W]).astype(dtype)
    for h in range(H):
        for w in range(W):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden

class tmulParams:
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
    return f"TMULTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tmulParams(np.float32, 64, 64, 64, 64, 64, 64),
        tmulParams(np.int32, 64, 64, 64, 64, 64, 64),
        tmulParams(np.float16, 64, 64, 64, 64, 64, 64),
        tmulParams(np.int16, 64, 64, 64, 64, 64, 64),
        tmulParams(np.float16, 161, 161, 32, 32, 161, 161),
        tmulParams(np.int32, 77, 81, 32, 16, 77, 81),
        tmulParams(np.int32, 32, 32, 32, 16, 32, 32),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tmul(case_name, param)
        os.chdir(original_dir)