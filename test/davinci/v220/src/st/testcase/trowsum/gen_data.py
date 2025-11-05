#!/usr/bin/python3
# coding=utf-8

import os
import struct
import ctypes
import numpy as np
np.random.seed(23)

def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col

    input_arr = np.random.uniform(low=-8, high=8, size=(rows, cols)).astype(data_type)
    output_arr = np.zeros((rows,1), dtype=data_type)
    for i in range(rows):
        for j in range(cols):
            output_arr[i,0] += input_arr[i,j]
    input_arr.tofile('input.bin')
    output_arr.tofile('golden.bin')

class trowsumParams:
    def __init__(self, name, data_type, row, col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.col = col

if __name__ == "__main__":
    case_params_list = [
        trowsumParams("TROWSUMTest.case1", np.float32, 127, 64 - 1),
        trowsumParams("TROWSUMTest.case2", np.float32, 63, 64),
        trowsumParams("TROWSUMTest.case3", np.float32, 31, 64 * 2 - 1),
        trowsumParams("TROWSUMTest.case4", np.float32, 15, 64 * 3),
        trowsumParams("TROWSUMTest.case5", np.float32, 7, 64 * 7 + 1),
        trowsumParams("TROWSUMTest.case6", np.float16, 255 + 1, 15)
        # todo
    ]

    for i, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)