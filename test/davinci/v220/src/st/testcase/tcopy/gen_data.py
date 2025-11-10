#!/usr/bin/python3
# coding=utf-8

import os

import numpy as np
np.random.seed(19)


def gen_golden_data(case_name, param):
    data_type = param.data_type
    rows = param.src_valid_row
    cols = param.src_valid_col
    input_arr = np.random.uniform(low = -1, high = 1, size=(rows, cols)).astype(param.data_type)
    input_arr.tofile("input_arr.bin")
    golden = input_arr.copy()
    golden.tofile("./golden.bin")


class tcopyParams:
    def __init__(self, data_type, src_valid_row, src_valid_col, dst_valid_row, dst_valid_col):
        self.data_type = data_type
        self.src_valid_col = src_valid_col
        self.src_valid_row = src_valid_row
        self.dst_valid_row = dst_valid_row
        self.dst_valid_col = dst_valid_col


if __name__ == "__main__":
    case_name_list = [
        "TCOPYTest.case1"
    ]

    case_params_list = [
        tcopyParams(np.float32, 128 , 128, 128, 128)
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)