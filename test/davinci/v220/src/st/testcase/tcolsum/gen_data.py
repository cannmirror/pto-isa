#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(23)

def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col

    input_arr = np.random.uniform(low=0, high=16, size=(row, col)).astype(data_type)
    output_arr = np.zeros((row))
    for i in range(valid_row):
        for j in range(valid_col):
            output_arr[i] += input_arr[i, j]

    # 先计算, 再强转类型, 保证结果精度不裂化
    output_arr = output_arr.astype(data_type)
    input_arr.tofile('input.bin')
    output_arr.tofile('golden.bin')

class trowsumParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col

if __name__ == "__main__":
    case_params_list = [
        trowsumParams("TROWSUMTest.case1", np.float32, 127, 127, 64, 64 - 1),
        trowsumParams("TROWSUMTest.case2", np.float32, 63, 63, 64, 64),
        trowsumParams("TROWSUMTest.case3", np.float32, 31, 31, 64 * 2, 64 * 2 - 1),
        trowsumParams("TROWSUMTest.case4", np.float32, 15, 15, 64 * 3, 64 * 3),
        trowsumParams("TROWSUMTest.case5", np.float32, 7, 7, 64 * 7, 64 * 7 - 1),
        trowsumParams("TROWSUMTest.case6", np.float16, 256, 256, 16, 16 - 1)
    ]

    for i, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)