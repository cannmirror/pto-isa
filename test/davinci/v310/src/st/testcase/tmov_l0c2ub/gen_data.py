#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    dst_type = param.ctype

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, False, False, False

    x1_gm = np.random.randint(1, 5, [m, k]).astype(a_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(b_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(dst_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if is_bias:
        golden = np.matmul(x1_gm, x2_gm).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm, x2_gm).astype(dst_type)

    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmovParams:
    def __init__(self, atype, btype, ctype, m, k, n):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMOVTest.case_nd_1",
        "TMOVTest.case_nd_2",
        "TMOVTest.case_nd_3",
        "TMOVTest.case_nd_4",
        "TMOVTest.case_nd_5",
        "TMOVTest.case_nd_6",
    ]

    case_params_list = [
        tmovParams(np.float16, np.float16, np.float32, 64, 128, 128),   # f32 -> f32
        tmovParams(np.float16, np.float16, np.float16, 128, 128, 64),    # f32 -> f16
        tmovParams(np.float16, np.float16, np.float32, 64, 64, 64),     # (64, 64) -> (64, 128)
        tmovParams(np.float16, np.float16, np.float32, 31, 24, 24),     # (31, 24)
        tmovParams(np.float16, np.float16, np.float32, 32, 32, 64),     # sub block id = 1
        tmovParams(np.float16, np.float16, bfloat16, 128, 64, 128),     # f32 -> bf16
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)