#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import math
import struct
import ctypes

def gen_golden_data(case_name, param):
    test_type = param.test_type
    rows = param.rows
    cols = param.cols
    input_arr = np.random.uniform(low=-1, high = 1, size=(rows, cols)).astype(test_type)
    input_arr.tofile("input_arr.bin")
    idx = np.arange(rows * cols).astype(np.uint32)
    idx.tofile("input_idx.bin")
    tmp = np.zeros((1, cols)).astype(test_type)
    tmp.tofile("input_tmp.bin")

    reshape_val = 32
    if cols < 32:
        reshape_val = cols
    input_reshaped = input_arr.reshape(-1, reshape_val)
    idx_reshaped = idx.reshape(-1, 32)
    if cols < 32:
        pad_value = np.finfo(test_type).min
        input_reshaped_padded = np.pad(
            input_reshaped,
            pad_width = ((0, 0), (0, 32 - cols)),
            mode='constant',
            constant_values=pad_value
        )
        input_reshaped = input_reshaped_padded

    if cols > 32:
        reshape_val = cols
    # sort each group of 32 elements based on input values in descending order
    sorted_indices = np.argsort(-input_reshaped, axis=1)
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)
    sorted_input = sorted_input.reshape(rows, reshape_val)
    sorted_idx = sorted_idx.reshape(rows, reshape_val)
    flat_input = sorted_input.flatten().astype(test_type)
    flat_idx   = sorted_idx.flatten()
    # create pairs of (value, index)
    sorted_pairs = zip(flat_input, flat_idx)

    with open("golden_output.bin", 'wb') as f:
        for value, index in sorted_pairs:
            if test_type == np.float32:
                # pack the float32 value and the index as a 32-bit unsigned integer
                packed_data = struct.pack('fI', float(value), ctypes.c_uint32(index).value)
                f.write(packed_data)
            elif test_type == np.float16:
                packed_data = struct.pack('e xxI', value, ctypes.c_uint32(index).value)
                f.write(packed_data)


class tsort32Params:
    def __init__(self, test_type, rows, cols):
        self.test_type = test_type
        self.rows = rows
        self.cols = cols

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TSort32Test.case1",
        "TSort32Test.case2",
    ]

    case_params_list = [
        tsort32Params(np.float32, 2, 32),
        tsort32Params(np.float16, 4, 64)
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)