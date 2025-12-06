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
import copy
import struct
np.random.seed(19)

bfloat16 = ml_dtypes.bfloat16

def extract_quant_params(quant_gm):
    """
    从uint64类型的quant_gm中提取M1、offset、sign参数
    Args:
        quant_g: uint64类型的整数
    Return:
        M1: 自定义格式(1,8,10)的浮点数
        offset: 9位整数
        sign: 1位布尔值(0或1)
    """
    quant_gm = int(quant_gm)
    M1_bits = (quant_gm >> 13) & 0x7FFFF  # 提取M1的19位[31:13]，0x7FFFF是19位掩码
    offset = (quant_gm >> 37) & 0x1FF     # 提取offset的9位[45:37]，0x1FF是9位掩码
    sign = (quant_gm >> 46) & 0x1         # 提取sign的一位[46]，0x1是1位掩码

    # 解析M1为(1,8,10)格式的浮点数
    sign_bit = (M1_bits >> 18) & 0x1
    exponent = (M1_bits >> 10) & 0xFF
    mantissa = M1_bits & 0x3FF
    exponent_bias = 127  # 假设指数偏倚量为127，与float32一致
    M1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return M1, offset, sign

def saturation(value, min_val, max_val, target_type):
    """
    将输入的浮点数进行饱和处理，并转换为目标类型
    """
    x_clamped = np.clip(value, min_val, max_val)  # 饱和处理
    return np.round(x_clamped).astype(target_type)  # 四舍五入并转换为目标类型

def qf2b8_pre(data, quant_gm):
    """
    float32 -> int8
    int32 ->int8
    """
    m1, offset, sign = extract_quant_params(quant_gm)
    tmp1 = saturation(data.astype(np.float32) * m1, -256, 255, np.int16) + offset
    if sign:
        return saturation(tmp1, -128, 127, np.int8)
    else:
        return saturation(tmp1, 0, 255, np.uint8)

def qf2f16_pre(data, quant_gm):
    """
    float32 -> float16
    """
    m1, offset, sign = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16)

def qf2bf16_pre(data, quant_gm):
    """
    float32 -> bfloat16
    """
    m1, offset, sign = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, 0x0080, 0x7F80, bfloat16)

def gen_golden_data(case_name, param):
    src_type = param.atype
    l0c_type = param.ctype
    bias_type = param.bias_type
    dst_type = param.dst_type
    quant_type = param.quant_type

    m, k, n, is_bias, is_quant = param.m, param.k, param.n, param.is_bias, param.is_quant

    x1_gm = np.random.randint(-1, 10, [m, k]).astype(src_type)
    x2_gm = np.random.randint(-1, 10, [k, n]).astype(src_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(bias_type)

    if is_bias:
        golden = np.matmul(x1_gm.astype(l0c_type), x2_gm.astype(l0c_type)).astype(l0c_type) + bias_gm.astype(l0c_type)
    else:
        golden = np.matmul(x1_gm.astype(l0c_type), x2_gm.astype(l0c_type)).astype(l0c_type)

    # fixpipe
    temp_quant_tensor = np.random.randint(1, 5, n).astype(np.float32)
    temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
    for i, _ in enumerate(temp_quant_tensor_api):
        temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor_api[i]))[0]
        temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
    quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
    quant_tensor = quant_tensor.astype(quant_type)
    quant_golden = np.zeros((m, n), dtype = dst_type)
    if is_quant:
        for i in range(m):
            for j in range(n):
                if dst_type == np.int8:
                    # int32 -> int8
                    # float32 -> int8
                    quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
                elif dst_type == np.float16:
                    # float32 -> float16
                    quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
                elif dst_type == bfloat16:
                    # float32 -> bfloat16
                    quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
                else:
                    quant_golden[i, j] = golden[i, j] * quant_tensor[j]

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    quant_tensor.tofile("./quant_gm.bin")
    golden.tofile("./golden.bin")
    quant_golden.tofile("./quant_golden.bin")
    os.chdir(original_dir)


class tmovParams:
    def __init__(self, atype, btype, ctype, bias_type, dst_type, quant_type, m, n, k, is_bias = 0, is_quant = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.bias_type = bias_type
        self.dst_type = dst_type
        self.quant_type = quant_type
        self.m = m
        self.n = n
        self.k = k
        self.is_bias = is_bias
        self.is_quant = is_quant

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMOVTest.case_bias1",  # 此名称要和TEST_F(TMATMULTest, case1)定义的名称一致
        "TMOVTest.case_bias2",
        "TMOVTest.case_bias3",
        "TMOVTest.case_bias4",
        "TMOVTest.case_bias5",
        "TMOVTest.case_bias_dynamic6",
        "TMOVTest.case_bias_dynamic7",
        "TMOVTest.case_bias_dynamic8",

        "TMOVTest.case_fixpipe1",
        "TMOVTest.case_fixpipe2",
        "TMOVTest.case_fixpipe3",
        "TMOVTest.case_fixpipe4",
        "TMOVTest.case_fixpipe5",
    ]

    case_params_list = [
        # L1_TO_BIAS
        tmovParams(np.float16, np.float16, np.float32, np.float32, np.float32, np.uint64, 64, 32, 96, 1, 0),    # float32 -> float32
        tmovParams(np.float32, np.float32, np.float32, np.float16, np.float32, np.uint64, 128, 64, 128, 1, 0),  # half -> float32
        tmovParams(np.float32, np.float32, np.float32, bfloat16, np.float32, np.uint64, 64, 32, 80, 1, 0),  # bfloat16 -> float32
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int32, np.uint64, 128, 64, 96, 1, 0),               # int32 -> int32
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int32, np.uint64, 31, 63, 32, 1, 0),                # 非对齐分型 int32 -> int32
        tmovParams(np.float16, np.float16, np.float32, np.float16, np.float32, np.uint64, 64, 32, 80, 1, 0),    # 动态tile float32 -> float32
        tmovParams(np.float32, np.float32, np.float32, bfloat16, np.float32, np.uint64, 112, 48, 96, 1, 0), # 动态tile bfloat16 -> float32
        tmovParams(np.float32, np.float32, np.float32, bfloat16, np.float32, np.uint64, 15, 63, 96, 1, 0),  # 动态tile 非对齐分型 bfloat16 -> float32

        # L1_TO_FB: quant
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int8, np.uint64, 32, 128, 32, 0, 1),                # int32 -> int8
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.float16, np.uint64, 96, 64, 32, 0, 1),              # int32 -> half
        tmovParams(np.int8, np.int8, np.int32, np.int32, bfloat16, np.uint64, 128, 64, 96, 0, 1),           # int32 -> bfloat16
        tmovParams(np.float32, np.float32, np.float32, np.float32, np.int8, np.uint64, 112, 48, 96, 0, 1),      # float32 -> int8
        tmovParams(np.float32, np.float32, np.float32, np.float32, np.int8, np.uint64, 31, 31, 96, 0, 1),       # 非对齐分型 float32 -> int8
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)