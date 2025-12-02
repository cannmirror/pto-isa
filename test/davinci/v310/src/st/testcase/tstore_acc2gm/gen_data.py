#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)

def gen_golden_data(case_name, g_info):
    src_data_type = g_info.src_data_type
    dst_data_type = g_info.dst_data_type
    g_shape0 = g_info.g_shape0
    g_shape1 = g_info.g_shape1
    g_shape2 = g_info.g_shape2
    g_shape3 = g_info.g_shape3
    g_shape4 = g_info.g_shape4
    g_whole_shape0 = g_info.g_whole_shape0
    g_whole_shape1 = g_info.g_whole_shape1
    g_whole_shape2 = g_info.g_whole_shape2
    g_whole_shape3 = g_info.g_whole_shape3
    g_whole_shape4 = g_info.g_whole_shape4
    m = g_info.m
    n = g_info.n
    k = g_info.k
    format = g_info.format
    
    # 1: scalar quant mode, 2: vector quant mode
    # one case int8 * int8 = int32 -> half
    if g_info.quant_mode == 2:
        quant_vector = np.random.uniform(0.1, 2.0, [1, g_info.n]).astype(np.float32)
        quant_vector_gm = np.frombuffer(quant_vector, np.int32)
        quant_vector_gm = quant_vector_gm.astype(np.uint64)
    
    x1_gm = np.random.randint(1, 10, [m, k]).astype(src_data_type)
    x2_gm = np.random.randint(1, 10, [k, n]).astype(src_data_type)
    golden = np.matmul(x1_gm.astype(dst_data_type), x2_gm.astype(dst_data_type)).astype(dst_data_type)
    
    if g_info.quant_mode == 1:
        golden = golden * 1
    elif g_info.quant_mode == 2:
        quant_vector = quant_vector.view("uint32")
        for index, data in enumerate(quant_vector):
            # 1 sign bit, 8 exponent bits and 10 mantissa bits
            quant_vector[index] = np.bitwise_and(data, 0xFFFFE000)
        quant_vector = quant_vector.view("float32")
        for i in range(m):
            golden[i, :] = golden[i, :] * quant_vector

    c0_size = 16
    if format == 2:
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    elif format == 3:
        c0_size = 8
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")
    if g_info.quant_mode == 2:
        quant_vector_gm.tofile("./quant_vector_gm.bin")

class TStoreAcc2gmParams:
    def __init__(self, dst_data_type, src_data_type, format, g_shape0, g_shape1, g_shape2, g_shape3, g_shape4,
                g_whole_shape0, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4, m, n, k, quant_mode):
        self.src_data_type = src_data_type
        self.dst_data_type = dst_data_type
        # 1: NZ2ND 2: NZ2NZ 3: channelSplit
        self.format = format
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4
        self.m = m
        self.n = n
        self.k = k
        self.quant_mode = quant_mode


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
        "TStoreAcc2gmTest.case16",
        "TStoreAcc2gmTest.case17"
    ]

    case_params_list = [
        TStoreAcc2gmParams(np.float32, np.float32, 1, 1, 1, 1, 128, 128, 1, 2, 3, 256, 128, 128, 128, 16, 0),
        TStoreAcc2gmParams(np.float32, np.float32, 1, 1, 1, 1, 31, 32, 1, 2, 3, 31, 32, 31, 32, 15, 0),
        TStoreAcc2gmParams(np.float32, np.float16, 1, 1, 1, 1, 65, 128, 1, 2, 3, 65, 128, 65, 128, 96, 0),
        TStoreAcc2gmParams(np.float16, np.float16, 1, 1, 1, 1, 73, 64, 2, 2, 3, 73, 64, 73, 64, 32, 0),
        TStoreAcc2gmParams(np.float32, bfloat16, 1, 1, 1, 1, 13, 32, 2, 3, 7, 13, 32, 13, 32, 25, 0),
        TStoreAcc2gmParams(bfloat16, bfloat16, 1, 1, 1, 1, 100, 222, 5, 7, 7, 100, 222, 100, 222, 60, 0),

        TStoreAcc2gmParams(np.float32, np.float32, 2, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 25, 0),
        TStoreAcc2gmParams(np.float32, np.float32, 2, 1, 2, 3, 16, 16, 1, 2, 3, 16, 16, 48, 32, 45, 0),
        TStoreAcc2gmParams(np.float32, np.float16, 2, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 24, 0),
        TStoreAcc2gmParams(np.float16, np.float16, 2, 2, 3, 6, 16, 16, 2, 3, 6, 16, 16, 96, 96, 23, 0),
        TStoreAcc2gmParams(np.float32, bfloat16, 2, 2, 3, 3, 16, 16, 2, 3, 3, 16, 16, 48, 96, 22, 0),
        TStoreAcc2gmParams(bfloat16, bfloat16, 2, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32, 0),

        TStoreAcc2gmParams(np.int32, np.int8, 1, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27, 0),
        TStoreAcc2gmParams(np.int32, np.int8, 2, 2, 3, 4, 16, 16, 2, 3, 4, 16, 16, 64, 96, 30, 0),

        TStoreAcc2gmParams(np.float32, np.float32, 3, 3, 8, 4, 16, 8, 3, 8, 4, 16, 8, 64, 192, 43, 0),
        TStoreAcc2gmParams(np.float16, np.int8, 1, 1, 1, 1, 32, 32, 1, 2, 3, 32, 32, 32, 32, 32, 1),
        TStoreAcc2gmParams(np.float16, np.int8, 1, 1, 1, 1, 32, 32, 1, 2, 3, 32, 32, 32, 32, 32, 2),
    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
