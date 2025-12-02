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
def create_padded_tensors(
    x1_gm, x2_gm, m, n, k, target_m, target_n, target_k, src_type=np.int8,
    rand_range_right=(1,5),
    rand_range_down=(1,5),
    rand_range_corner=(1,5)):
    assert target_m >= m, f"target_m ({target_m}) mast be >= m ({m})"
    assert target_n >= n, f"target_n ({target_n}) mast be >= n ({n})"
    assert target_k >= k, f"target_k ({target_k}) mast be >= k ({k})"
    #构造x1_gm_padded：target_m, target_k
    x1_gm_padded = np.zeros((target_m, target_k), dtype=np.int32).astype(src_type)
    #原始数据
    x1_gm_padded[:m, :k] = x1_gm
    #右侧补随机值（k方向扩展）
    right_fill = np.random.randint(rand_range_right[0], rand_range_right[1],
                                    size=(m, target_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[:m, k:target_k] = right_fill
    #下方补0（m方向扩展）
    x1_gm_padded[m:target_m, :k] = 0

    #右下角补随机值
    corner_fill = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                    size=(target_m - m, target_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[m:target_m, k:target_k] = corner_fill
    #构造x2_gm_padded：target_k, target_n
    x2_gm_padded = np.zeros((target_k, target_n), dtype=np.int32).astype(src_type)
    x2_gm_padded[:k, :n] = x2_gm
    down_fill = np.random.randint(rand_range_down[0], rand_range_down[1],
                                    size=(target_k - k, n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:target_k, :n] = down_fill
    x2_gm_padded[:k, n:target_n] = 0
    corner_fill2 = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                     size=(target_k - k, target_n - n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:target_k, n:target_n] = corner_fill2
    return x1_gm_padded, x2_gm_padded

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, n, k, start_m, start_n, start_k, is_atrans, is_btrans, target_m, target_n, target_k = \
    param.m, param.n, param.k, param.start_m, param.start_n, param.start_k, \
    param.is_atrans, param.is_btrans, param.target_m, param.target_n, param.target_k
    
    target_m = target_m if target_m > 0 else m
    target_n = target_n if target_n > 0 else n
    target_k = target_k if target_k > 0 else k
    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    x1_slice = x1_gm[start_m:, start_k:]  # 从(rowIdx1, colIdx1)开始到结束
    x2_slice = x2_gm[start_k:, start_n:]  # 从(rowIdx2, colIdx2)开始到结束
    #计算真值
    golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)
    #填充、转置处理
    x1_gm, x2_gm = create_padded_tensors(x1_gm, x2_gm, m, n, k, target_m, target_n, target_k, src_type, rand_range_right=(1,5), rand_range_down=(1,5), rand_range_corner=(1,5))
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
    golden.tofile("./golden.bin")

    os.chdir(original_dir)

class textractParams:
    def __init__(self, atype, btype, ctype, m, n, k, start_m, start_n, start_k, is_atrans=0, is_btrans=0, target_m = 0, target_n = 0, target_k = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.n = n
        self.k = k
        self.start_m = start_m
        self.start_n = start_n
        self.start_k = start_k
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans
        self.target_m = target_m
        self.target_n = target_n
        self.target_k = target_k

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TEXTRACTTest.case1_half_0_1_param", # 此名称需要和 TEST_F(TMATMULTest, case1)定义的名称一致
        "TEXTRACTTest.case2_int8_0_1_param",
        "TEXTRACTTest.case3_float_0_1_param",
        "TEXTRACTTest.case4_bfloat16_0_1_param",

        "TEXTRACTTest.case11_half_0_1_16_16_32_param",
        "TEXTRACTTest.case12_int8_0_1_48_32_64_param",
        "TEXTRACTTest.case13_float_0_1_32_16_48_param",
        "TEXTRACTTest.case14_bfloat16_0_1_32_32_16_param",

        "TEXTRACTTest.case21_half_1_0_param",
        "TEXTRACTTest.case22_int8_1_0_param",
        "TEXTRACTTest.case23_float_1_0_param",
        "TEXTRACTTest.case24_bfloat16_1_0_param",

        "TEXTRACTTest.case31_half_1_0_96_0_64_param",
        "TEXTRACTTest.case32_int8_1_0_32_0_32_param",
        "TEXTRACTTest.case33_float_1_0_32_0_16_param",
        "TEXTRACTTest.case34_bfloat16_1_0_32_0_48_param",

        "TEXTRACTTest.case41_float_1_0_65_66_40_param",
        "TEXTRACTTest.case42_int8_1_0_65_66_40_param",
        "TEXTRACTTest.case43_half_1_0_65_66_40_param",
        "TEXTRACTTest.case44_bfloat16_1_0_65_66_40_param",

        "TEXTRACTTest.case51_dynamic_half_0_1_16_0_32_param",
        "TEXTRACTTest.case52_dynamic_int8_1_1_32_0_32_param",
        "TEXTRACTTest.case53_dynamic_int8_0_1_param",
        "TEXTRACTTest.case54_dynamic_half_1_1_param",
    ]

    case_params_list = [
        ## A MK输入，B NK输入， 均需转置
        textractParams(np.float16, np.float16, np.float32, 64, 32, 80, 0, 0, 0, 0, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 0, 0, 0, 0, 1),
        textractParams(np.float32, np.float32,  np.float32, 128, 48, 64, 0, 0, 0, 0, 1),
        textractParams(bfloat16, bfloat16, np.float32, 64, 48, 96, 0, 0, 0, 0, 1),

        textractParams(np.float16, np.float16, np.float32, 64, 32, 80, 16, 16, 32, 0, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 48, 32, 64, 0, 1),
        textractParams(np.float32, np.float32,  np.float32, 96, 48, 64, 32, 16, 48, 0, 1),
        textractParams(bfloat16, bfloat16, np.float32, 64, 48, 96, 32, 32, 16, 0, 1),
        ## A KM输入 B KN输入， 均不需转置
        textractParams(np.float16, np.float16, np.float32, 128, 64, 128, 0, 0, 0, 1, 0),
        textractParams(np.int8, np.int8, np.int32, 64, 64, 128, 0, 0, 0, 1, 0),
        textractParams(np.float32, np.float32,  np.float32, 64, 32, 96, 0, 0, 0, 1, 0),
        textractParams(bfloat16, bfloat16, np.float32, 96, 80, 96, 0, 0, 0, 1, 0),

        textractParams(np.float16, np.float16, np.float32, 128, 64, 128, 96, 32, 64, 1, 0),
        textractParams(np.int8, np.int8, np.int32, 64, 64, 128, 32, 32, 32, 1, 0),
        textractParams(np.float32, np.float32,  np.float32, 64, 32, 96, 32, 16, 16, 1, 0),
        textractParams(bfloat16, bfloat16, np.float32, 96, 80, 96, 32, 64, 48, 1, 0),        
        ## 非对齐场景 A KM输入， B KN输入， 均需转置
        textractParams(np.float32, np.float32, np.float32, 65, 66, 40, 0, 0, 0, 1, 0, 80, 80, 48),
        textractParams(np.int8, np.int8, np.int32, 65, 66, 40, 0, 0, 0, 1, 0, 96, 96, 64),
        textractParams(np.float16, np.float16, np.float32, 65, 66, 40, 0, 0, 0, 1, 0, 80, 80, 48),
        textractParams(bfloat16, bfloat16, np.float32, 65, 66, 40, 0, 0, 0, 1, 0, 80, 80, 48),
        ##动态输入
        textractParams(np.float16, np.float16, np.float32, 64, 32, 80, 16, 0, 32, 0, 1),
        textractParams(np.int8, np.int8, np.int32, 64, 64, 128, 32, 0, 32, 1, 1),
        textractParams(np.int8, np.int8, np.int32, 128, 64, 128, 0, 0, 0, 0, 1),
        textractParams(np.float16, np.float16, np.float32, 128, 64, 128, 0, 0, 0, 1, 1),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)