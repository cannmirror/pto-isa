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
    # x1_gm = x1_gm.reshape((int(x1_gm.shape[0] / 16), 16, int(x1_gm.shape[1] / c0_size), c0_size)).transpose(2, 0, 1, 3)
    # x1_gm = x1_gm.reshape(x1_gm.shape[0] * x1_gm.shape[1], x1_gm.shape[2] * x1_gm.shape[3])

    # x2_gm = x2_gm.reshape((int(x2_gm.shape[0] / 16), 16, int(x2_gm.shape[1] / c0_size), c0_size)).transpose(2, 0, 1, 3)
    # x2_gm = x2_gm.reshape(x2_gm.shape[0] * x2_gm.shape[1], x2_gm.shape[2] * x2_gm.shape[3])


    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class textractParams:
    def __init__(self, atype, btype, ctype, m, k, n, start_m, start_k, is_atrans=0, is_btrans=0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.start_m = start_m
        self.start_k = start_k
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TEXTRACTTest.case1",
        "TEXTRACTTest.case2",
        "TEXTRACTTest.case3",
        "TEXTRACTTest.case4",
        "TEXTRACTTest.case5",
        "TEXTRACTTest.case6",
        "TEXTRACTTest.case7",
        "TEXTRACTTest.case8",
        "TEXTRACTTest.case9",
    ]

    case_params_list = [
        textractParams(np.float16, np.float16, np.float32, 32, 96, 64, 0, 0, 0, 0),
        textractParams(np.float32, np.float32, np.float32, 128, 48, 64, 0, 0, 0, 0),
        textractParams(np.int8, np.int8, np.int32, 128, 128, 64, 0, 0, 0, 0),
        textractParams(np.float16, np.float16, np.float32, 64, 96, 64, 32, 0, 0, 0),
        textractParams(np.float32, np.float32, np.float32, 128, 128, 64, 32, 0, 0, 0),
        textractParams(np.int8, np.int8, np.int32, 128, 128, 64, 32, 0, 0, 0),
        textractParams(np.float16, np.float16, np.float32, 128, 128, 64, 0, 0, 1, 0),
        textractParams(np.float32, np.float32, np.float32, 128, 128, 64, 0, 0, 1, 0),
        textractParams(np.int8, np.int8, np.int32, 128, 128, 64, 0, 0, 1, 0),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)


