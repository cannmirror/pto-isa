#!/user/bin/python3
# coding=utf-8

import os

import numpy as np
# from m1_dtypes import bfloat16, float8_e4m3fn, float8_e5m2fnuz
np.random.seed(0)


def Gather(src, indices):
    output = np.zeros_like(indices, dtype=src.dtype)
    for i in range(indices.shape[0]):
        output[i] = src[indices[i]]
    return output


def gen_golden_data(case_name, param):
    src0_type = param.src0type
    src1_type = param.src1type
    src0_row = param.src0_row
    src0_col = param.src0_col
    src1_row = param.src1_row
    src1_col = param.src1_col

    src_data = np.random.randint(-20, 20, (src0_row*src0_col).astype(src0_type))
    indices = np.random.randint(0, src0_row*src0_col, src1_row*src1_col).astype(src1_type)
    golden = Gather(src_data, indices)

    src_data.tofile("./src0.bin")
    indices.tofile("./src1.bin")
    golden.tofile("./golden.bin")
    os.chdir(original_dir)

class tgatherParams:
    def __init__(self, src0type, src1type, src0_row, src0_col, src1_row, src1_col):
        self.src0type = src0type
        self.src1type = src1type
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col

if __name__ == '__main__':
    case_name_list = [
        "TGATHERTest.case1_float",
        "TGATHERTest.case2_int32",
        "TGATHERTest.case3_half",
        "TGATHERTest.case4_int16",
    ]

    case_params_list = [
        tgatherParams(np.float32, np.int32, 32, 1024, 16, 64),
        tgatherParams(np.int32, np.int32, 32, 512, 16, 256),
        tgatherParams(np.half, np.int16, 16, 1024, 16, 128),
        tgatherParams(np.int16, np.int16, 32, 256, 32, 64),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.mkdir(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)