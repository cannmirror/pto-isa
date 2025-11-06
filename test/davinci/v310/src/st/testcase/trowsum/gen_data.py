#!/usr/bin/python3
# coding=utf-8

import os
import struct
import ctypes
import numpy as np
np.random.seed(19)


def gen_golden_data(case_name, param):
    src_type = param.datatype
    dst_type = param.datatype
    rows = param.row
    cols = param.col

    input_arr = np.random.uniform(low=-8, high=8, size=(rows,cols)).astype(src_type)
    result_arr = input_arr.sum(axis=1, keepdims=True)
    output_arr = np.zeros((rows,cols), dtype=dst_type)
    for i in range(cols):
        output_arr[i,0]=result_arr[i,0]
    input_arr.tofile('input0.bin')
    output_arr.tofile('golden.bin')

class trowsumParams:
    def __init__(self, datatype, row, col):
        self.datatype = datatype
        self.row = row
        self.col = col

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TROWSUMTest.test1",
        "TROWSUMTest.test2",
        "TROWSUMTest.test3"
    ]
    
    case_params_list = [
        trowsumParams(np.float32, 16, 16),
        trowsumParams(np.float16, 16, 16),
        trowsumParams(np.float32, 666, 666)
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)