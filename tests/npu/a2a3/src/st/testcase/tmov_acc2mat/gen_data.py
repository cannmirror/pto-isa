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
import copy
import struct
import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)


def extract_quant_params(quant_gm):
    """
    从uint64类型的quant_gm中提取M1、offset、sign参数
    param:
        quant_g：uint64类型的整数
    return:
        M1：自定义格式(1,8,10)的浮点数
        offset：9位整数
        sign：1位布尔值（0或1）
    """
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0xFFFFF  # 提取M1的20位[31:13]，0xFFFFF是20位掩码
    offset = (quant_gm >> 37) & 0x1FF # 提取offset的9位[45:37]，0x1FF是9位掩码
    sign = (quant_gm >> 46) & 0x1  # 提取sign的一位[46]，0x1是1位掩码
    n = (quant_gm >> 32) & 0xF
    # 解析M1为(1,8,10)格式的浮点数
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127 # 假设指数偏倚量为127，与float32一致
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))
    return m1, offset, sign, n


def saturation(value, min_val, max_val, target_type):
    """
    将输入的浮点数进行饱和处理，并转换为目标类型
    """
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type).astype(target_type)


def qf2b8_pre(data, quant_gm):
    """
    float32 -> int8
    int32 ->int8/uint8
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    tmp1 = saturation(data.astype(np.float32) * m1, -256, 255, np.int16) + offset
    if sign:
        return saturation(tmp1, -128, 127, np.int8)
    else:
        return saturation(tmp1, 0, 255, np.uint8)


def qf2f16_pre(data, quant_gm):
    """
    float32 -> float16
    int32 -> float16
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16)


def qf2bf16_pre(data, quant_gm):
    """
    float32 -> bfloat16
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, 0x0080, 0x7F80, bfloat16)


def qs2s16_pre(data, quant_gm):
    """
    int32 -> int16
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    tmp1 = data >> (n + 1)
    return saturation(tmp1, -32768, 32767, np.int16)


def vector_quant_non_int16(golden, dst_type, n, m, quant_type):
    temp_quant_tensor = np.random.randint(1, 3, n).astype(np.float32)
    temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
    for i, _ in enumerate(temp_quant_tensor_api):
        temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor[i]))[0]
        if dst_type == np.int8:
            temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
    quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
    quant_tensor = quant_tensor.astype(quant_type)
    quant_tensor.tofile("./quant_gm.bin")
    quant_golden = np.zeros((m, n), dtype=dst_type)
    for i in range(m):
        for j in range(n):
            if dst_type in (np.int8, np.uint8):
                quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
            elif dst_type == np.float16:
                quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
            elif dst_type == bfloat16:
                quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
    return quant_golden


def vector_quant_int16(golden, dst_type, n, m, quant_type):
    temp_quant_tensor = np.random.randint(1, 9, n).astype(np.int8)
    value = temp_quant_tensor - 1
    quant_tensor = (value.astype(quant_type) << 32)
    quant_tensor.tofile("./quant_gm.bin")
    quant_golden = np.zeros((m, n), dtype=dst_type)
    for i in range(m):
        for j in range(n):
            quant_golden[i, j] = qs2s16_pre(golden[i, j], quant_tensor[j])
    return quant_golden


def scalar_quant_non_int16(golden, dst_type, scalar):
    golden = golden * scalar
    if dst_type == np.int8:
        golden = saturation(golden, -128, 127, np.int8)
    elif dst_type == np.uint8:
        golden = saturation(golden, 0, 255, np.uint8)
    return golden


def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    c_type = param.ctype
    m, k, n = param.m, param.k, param.n
    is_v_quant, is_s_quant, dst_type, scalar = param.is_v_quant, param.is_s_quant, param.dst_type, param.scalar

    if dst_type == np.int8:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(a_type)
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(a_type)
    else:
        x1_gm = np.random.randint(1, 3, [m, k]).astype(a_type)
        x2_gm = np.random.randint(1, 3, [k, n]).astype(b_type)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    golden = np.matmul(x1_gm.astype(c_type), x2_gm.astype(c_type)).astype(c_type)

    if is_v_quant and dst_type != np.int16:
        golden = vector_quant_non_int16(golden, dst_type, n, m, param.quant_type)
    elif is_v_quant and dst_type == np.int16:
        golden = vector_quant_int16(golden, dst_type, n, m, param.quant_type)
    elif is_s_quant and dst_type != np.int16:
        golden = scalar_quant_non_int16(golden, dst_type, scalar)
    elif is_s_quant and dst_type == np.int16:
        scalar = int(scalar)
        golden = golden >> scalar
        golden = saturation(golden, -32768, 32767, np.int16)
    block_cols = 16
    if (dst_type == np.int8 or dst_type == np.uint8):
        block_cols = 32
    golden = golden.reshape(
        (int(m / 16), 16, int(n / block_cols), block_cols)).transpose(2, 0, 1, 3).astype(dst_type)
    golden.astype(dst_type).tofile("./golden.bin")


class TmovParams:
    def __init__(self, atype, btype, dst_type, m, k, n,
                 is_v_quant=False, is_s_quant=False,
                 quant_type=None, scalar=1):
        self.atype = atype
        self.btype = btype
        self.ctype = np.float32
        if (atype == np.int8):
            self.ctype = np.int32
        self.m = m
        self.k = k
        self.n = n
        self.is_v_quant = is_v_quant
        self.is_s_quant = is_s_quant
        self.dst_type = dst_type
        if (quant_type):
            self.quant_type = quant_type
        self.scalar = scalar

if __name__ == "__main__":
    case_name_list = [
        ##fp32->half
        "TMOVTest.case_nz2nz_1",
        ##fp32->bf16
        "TMOVTest.case_nz2nz_2",
        ##int32->half
        "TMOVTest.case_nz2nz_sc_quant_3",
        "TMOVTest.case_nz2nz_fb_quant_4",
        ##float->int8
        "TMOVTest.case_nz2nz_sc_quant_5",
        "TMOVTest.case_nz2nz_fb_quant_6",
        ##int32->int8
        "TMOVTest.case_nz2nz_sc_quant_7",
        "TMOVTest.case_nz2nz_fb_quant_8",
        ##int32->uint8
        "TMOVTest.case_nz2nz_sc_quant_9",
        "TMOVTest.case_nz2nz_fb_quant_10",
        ##int32->int16
        "TMOVTest.case_nz2nz_sc_quant_11",
        "TMOVTest.case_nz2nz_fb_quant_12",
    ]

    case_params_list = [
        ##fp32->half
        TmovParams(np.float16, np.float16, np.float16, 64, 128, 128),
        ##fp32->bf16
        TmovParams(np.float16, np.float16, bfloat16, 48, 128, 64),
        ##int32->half
        TmovParams(np.int8, np.int8, np.float16, 48, 64, 128, False, True, None, 2),
        TmovParams(np.int8, np.int8, np.float16, 80, 128, 64, True, False, np.uint64),
        ##float->int8
        TmovParams(np.float16, np.float16, np.int8, 48, 64, 128, False, True, None, 2),
        TmovParams(np.float16, np.float16, np.int8, 80, 128, 64, True, False, np.uint64),
        ##int32->int8
        TmovParams(np.int8, np.int8, np.int8, 48, 64, 128, False, True, None, 2),
        TmovParams(np.int8, np.int8, np.int8, 80, 128, 64, True, False, np.uint64),
        ##int32->uint8
        TmovParams(np.int8, np.int8, np.uint8, 48, 64, 128, False, True, None, 1),
        TmovParams(np.int8, np.int8, np.uint8, 80, 128, 64, True, False, np.uint64),
        ##int32->int16
        TmovParams(np.int8, np.int8, np.int16, 48, 64, 128, False, True, None, 2),
        TmovParams(np.int8, np.int8, np.int16, 80, 128, 64, True, False, np.uint64),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)