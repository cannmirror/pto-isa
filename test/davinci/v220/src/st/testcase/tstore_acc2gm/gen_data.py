#!/usr/bin/python3
# coding=utf-8

import os

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

def gen_golden_data(case_name, g_info):
    src_data_type = g_info.src_data_type
    dst_data_type = g_info.dst_data_type
    gShape0 = g_info.gShape0
    gShape1 = g_info.gShape1
    gShape2 = g_info.gShape2
    gShape3 = g_info.gShape3
    gShape4 = g_info.gShape4
    gWholeShape0 = g_info.gWholeShape0
    gWholeShape1 = g_info.gWholeShape1
    gWholeShape2 = g_info.gWholeShape2
    gWholeShape3 = g_info.gWholeShape3
    gWholeShape4 = g_info.gWholeShape4
    m = g_info.m
    n = g_info.n
    k = g_info.k
    format = g_info.format
    x1_gm = np.random.randint(-5, 5, [m, k]).astype(src_data_type)
    x2_gm = np.random.randint(-5, 5, [k, n]).astype(src_data_type)
    golden = np.matmul(x1_gm.astype(dst_data_type), x2_gm.astype(dst_data_type)).astype(dst_data_type)
    c0_size = 16
    if format == 2:
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    elif format == 3:
        c0_size = 8
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")

class TStoreAcc2gmParams:
    def __init__(self, dst_data_type, src_data_type, format, gShape0, gShape1, gShape2, gShape3, gShape4,
                gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4, m, n, k):
        self.src_data_type = src_data_type
        self.dst_data_type = dst_data_type
        self.format = format
        self.gShape0 = gShape0
        self.gShape1 = gShape1
        self.gShape2 = gShape2
        self.gShape3 = gShape3
        self.gShape4 = gShape4
        self.gWholeShape0 = gWholeShape0
        self.gWholeShape1 = gWholeShape1
        self.gWholeShape2 = gWholeShape2
        self.gWholeShape3 = gWholeShape3
        self.gWholeShape4 = gWholeShape4
        self.m = m
        self.n = n
        self.k = k


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TStoreAcc2gmTest.case1",
        "TStoreAcc2gmTest.case2",
        "TStoreAcc2gmTest.case3",
        "TStoreAcc2gmTest.case4",
        "TStoreAcc2gmTest.case5",
        "TStoreAcc2gmTest.case6",
        "TStoreAcc2gmTest.case7",
        "TStoreAcc2gmTest.case8",
        "TStoreAcc2gmTest.case9",
        "TStoreAcc2gmTest.case10",
        "TStoreAcc2gmTest.case11",
        "TStoreAcc2gmTest.case12",
        "TStoreAcc2gmTest.case13",
        "TStoreAcc2gmTest.case14",
        "TStoreAcc2gmTest.case15",
    ]

    case_params_list = [
        TStoreAcc2gmParams(np.float32, np.float32, 1, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, 61),
        TStoreAcc2gmParams(np.float32, np.float32, 1, 1, 1, 1, 31, 32, 1, 2, 3, 31, 32, 31, 32, 126),
        TStoreAcc2gmParams(np.float32, np.float16, 1, 1, 1, 1, 65, 128, 1, 2, 3, 65, 128, 65, 128, 96),
        TStoreAcc2gmParams(np.float16, np.float16, 1, 1, 1, 1, 73, 64, 2, 2, 3, 73, 64, 73, 64, 32),
        TStoreAcc2gmParams(np.float32, bfloat16, 1, 1, 1, 1, 13, 32, 2, 3, 7, 13, 32, 13, 32, 25),
        TStoreAcc2gmParams(bfloat16, bfloat16, 1, 1, 1, 1, 100, 222, 5, 7, 7, 100, 222, 100, 222, 60),

        TStoreAcc2gmParams(np.float32, np.float32, 2, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 25),
        TStoreAcc2gmParams(np.float32, np.float32, 2, 1, 2, 3, 16, 16, 1, 2, 3, 16, 16, 48, 32, 45),
        TStoreAcc2gmParams(np.float32, np.float16, 2, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 24),
        TStoreAcc2gmParams(np.float16, np.float16, 2, 2, 3, 6, 16, 16, 2, 3, 6, 16, 16, 96, 96, 23),
        TStoreAcc2gmParams(np.float32, bfloat16, 2, 2, 3, 3, 16, 16, 2, 3, 3, 16, 16, 48, 96, 22),
        TStoreAcc2gmParams(bfloat16, bfloat16, 2, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32),

        TStoreAcc2gmParams(np.int32, np.int8, 1, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27),
        TStoreAcc2gmParams(np.int32, np.int8, 2, 2, 3, 4, 16, 16, 2, 3, 4, 16, 16, 64, 96, 30),

        TStoreAcc2gmParams(np.float32, np.float32, 3, 3, 8, 4, 16, 8, 3, 8, 4, 16, 8, 64, 192, 43)
    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)