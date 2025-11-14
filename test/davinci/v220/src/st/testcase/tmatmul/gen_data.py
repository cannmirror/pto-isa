#!/usr/bin/python3
# coding=utf-8

import os

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False

    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(bias_type)

    if is_bias:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.is_bias = is_bias
        if (bias_type):
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case1", # 此名称要和TEST_F(TMATMULTest, case1)定义的名称一致
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULBIASTest.case1",
        "TMATMULBIASTest.case2",
        "TMATMULBIASTest.case3",
        "TMATMULBIASTest.case4",
        "TMATMULBIASTest.case5",
        "TMATMULBIASTest.case6",
        "TMATMULBIASTest.case7",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, False),
        tmatmulParams(np.int8, np.int8, np.int32, 128, 128, 64, False),
        tmatmulParams(np.float16, np.float16, np.float32, 127, 128, 63, False),
        # bias test
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, True, np.float32),
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 63, True, np.float32),
        tmatmulParams(np.float32, np.float32, np.float32, 15, 16, 15, True, np.float32),
        tmatmulParams(np.int8, np.int8, np.int32, 127, 127, 63, True),
        tmatmulParams(bfloat16, bfloat16, np.float32, 128, 128, 64, True, np.float32),
        tmatmulParams(np.int8, np.int8, np.int32, 128, 128, 64, True),
        # bias + acc test
        tmatmulParams(np.int8, np.int8, np.int32, 128, 128, 64, True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)