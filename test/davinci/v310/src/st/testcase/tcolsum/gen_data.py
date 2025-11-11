#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
np.random.seed(19)

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
            output_arr[j] += input_arr[i, j]

    # 先计算, 再强转类型, 保证结果精度不裂化
    output_arr = output_arr.astype(data_type)
    input_arr.tofile('input.bin')
    output_arr.tofile('golden.bin')

class tcolsumParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col

if __name__ == "__main__":
    case_params_list = [
        tcolsumParams("TCOLSUMTest.case1", np.float32, 64, 64, 64, 64),
    ]

    for i, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)