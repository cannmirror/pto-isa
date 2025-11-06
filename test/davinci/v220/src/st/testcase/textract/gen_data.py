#!/usr/bin/python3
# coding=utf-8

import os

import numpy as np
np.random.seed(19)

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, start_m, start_k, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.start_m, param.start_k,False, param.is_atrans, param.is_btrans

    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(dst_type)

    # 获取切片
    x1_slice = x1_gm[start_m:, start_k:]  # 从(rowIdx1, colIdx1)开始到结束
    x2_slice = x2_gm[start_k:, :]  # 从(rowIdx2, colIdx2)开始到结束

    # A:[m-start_m, k-start_k]
    # B:[k-start_k, n]
    # C:[m-start_m, n]
    if is_bias:
        golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type) + bias_gm
    else:
        golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()#[N,K]

    c0_size = 16
    if src_type == np.float32:
        c0_size = 8
    elif src_type == np.int8:
        c0_size = 32

    #转成NZ格式的输入
    x1_gm = x1_gm.reshape((int(x1_gm.shape[0] / 16), 16, int(x1_gm.shape[1] / c0_size), c0_size)).transpose(2, 0, 1, 3)
    x1_gm = x1_gm.reshape(x1_gm.shape[0] * x1_gm.shape[1], x1_gm.shape[2] * x1_gm.shape[3])

    x2_gm = x2_gm.reshape((int(x2_gm.shape[0] / 16), 16, int(x2_gm.shape[1] / c0_size), c0_size)).transpose(2, 0, 1, 3)
    x2_gm = x2_gm.reshape(x2_gm.shape[0] * x2_gm.shape[1], x2_gm.shape[2] * x2_gm.shape[3])

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")

    os.chdir(original_dir)

class textractParams:
    def __init__(self, atype, btype, ctype, m, n, k, start_m, start_k, is_atrans=0, is_btrans=0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.n = n
        self.k = k
        self.start_m = start_m
        self.start_k = start_k
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TEXTRACTTest.case1_half_0_1_param", # 此名称需要和 TEST_F(TMATMULTest, case1)定义的名称一致
        "TEXTRACTTest.case2_int8_0_1_param",
        "TEXTRACTTest.case3_float_0_1_param",
        "TEXTRACTTest.case11_half_0_1_16_32_param",
        "TEXTRACTTest.case12_int8_0_1_48_64_param",
        "TEXTRACTTest.case13_float_0_1_32_48_param",
        "TEXTRACTTest.case21_half_1_1_param",
        "TEXTRACTTest.case22_int8_1_1_param",
        "TEXTRACTTest.case23_float_1_1_param",
        "TEXTRACTTest.case31_half_1_1_96_64_param",
        "TEXTRACTTest.case32_int8_1_1_32_32_param",
        "TEXTRACTTest.case33_float_1_1_32_16_param",
        "TEXTRACTTest.case41_dynamic_half_0_1_16_32_param",
        "TEXTRACTTest.case42_dynamic_int8_1_1_32_32_param",
        "TEXTRACTTest.case43_dynamic_int8_0_1_param",
        "TEXTRACTTest.case44_dynamic_half_1_1_param",
    ]

    case_params_list = [
        textractParams(np.float16, np.float16, np.float32, 64, 32, 80, 0, 0, 0, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 0, 0, 0, 1),
        textractParams(np.float32, np.float32,  np.float32, 128, 48, 64, 0, 0, 0, 1),
        textractParams(np.float16, np.float16, np.float32, 64, 32, 80, 16, 32, 0, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 48, 64, 0, 1),
        textractParams(np.float32, np.float32,  np.float32, 96, 48, 64, 32, 48, 0, 1),
        textractParams(np.float16, np.float16, np.float32, 128, 64, 128, 0, 0, 1, 1),
        textractParams(np.int8, np.int8, np.int32, 64, 64, 128, 0, 0, 1, 1),
        textractParams(np.float32, np.float32,  np.float32, 64, 32, 96, 0, 0, 1, 1),
        textractParams(np.float16, np.float16, np.float32, 128, 64, 128, 96, 64, 1, 1),
        textractParams(np.int8, np.int8, np.int32, 64, 64, 128, 32, 32, 1, 1),
        textractParams(np.float32, np.float32,  np.float32, 64, 32, 96, 32, 16, 1, 1),
        textractParams(np.float16, np.float16, np.float32, 64, 32, 80, 16, 32, 0, 1),
        textractParams(np.int8, np.int8, np.int32, 64, 64, 128, 32, 32, 1, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 0, 0, 0, 1),
        textractParams(np.float16, np.float16, np.float32, 128, 64, 128, 0, 0, 1, 1),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)