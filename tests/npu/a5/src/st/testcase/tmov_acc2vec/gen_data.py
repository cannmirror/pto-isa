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

def ceil_div(num_1, num_2):
    if num_2 == 0:
        return 0
    return (num_1 + num_2 - 1) // num_2

def extract_quant_params(quant_gm):
    """
    Extract the parameters M1, offset, and sign from the quant_gm of type uint64.
    Args:
        quant_g: An integer of type uint64
    Return:
        m1: A floating-point number in custom format (1,8,10)
        offset: A 9-bit integer
        sign: A 1-bit boolean value (0 or 1)
    """
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0x7FFFF
    offset = (quant_gm >> 37) & 0x1FF
    sign = (quant_gm >> 46) & 0x1

    # Parse M1 into a floating-point number in (1,8,10) format.
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127  # Assuming the exponent bias is 127, which aligns with float32.
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return m1, offset, sign

def saturation(value, min_val, max_val, target_type):
    """
    Perform saturation processing on the input floating-point number and convert it to the target type.
    """
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type)

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
    a_type = param.atype
    b_type = param.btype
    c_type = param.ctype

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, False, False, False
    sFractalSize = param.sFractalSize if hasattr(param, 'sFractalSize') else 512
    dst_format = param.dst_format if hasattr(param, 'dst_format') else 'ND'
    is_v_quant, is_s_quant, dst_type, is_relu = param.isVQuant, param.isSQuant, param.dstType, param.isRelu
    
    if dst_type == np.int8:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(a_type)
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(a_type)
    else:
        x1_gm = np.random.randint(1, 5, [m, k]).astype(a_type)
        x2_gm = np.random.randint(1, 5, [k, n]).astype(b_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(c_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden = np.matmul(x1_gm.astype(c_type), x2_gm.astype(c_type)).astype(c_type)

    # fixpipe
    if is_v_quant:
        quant_type = param.quantType
        temp_quant_tensor = np.random.randint(1, 5, n).astype(np.float32)
        temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
        for i, _ in enumerate(temp_quant_tensor_api):
            temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor[i]))[0]
            if dst_type == np.int8:
                temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
        quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
        quant_tensor = quant_tensor.astype(quant_type)
        quant_tensor.tofile("./quant_gm.bin")
        quant_golden = np.zeros((m, n), dtype = dst_type)
        for i in range(m):
            for j in range(n):
                if dst_type == np.int8:
                    quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
                elif dst_type == np.float16:
                    quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
                elif dst_type == bfloat16:
                    quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
                else:
                    quant_golden[i, j] = golden[i, j] * quant_tensor[j]
        golden = quant_golden
    elif is_s_quant:
        scalar = param.scalar
        golden = golden * scalar
        if dst_type == np.int8:
            golden = saturation(golden, -128, 127, np.int8)
        elif dst_type == np.uint8:
            golden = saturation(golden, 0, 255, np.uint8)
    if is_relu:
        golden = np.maximum(golden, 0)

    if dst_format == 'NZ':
        if dst_type == np.float32 and sFractalSize == 512:
            block_cols = 8
        elif dst_type == np.int8 and sFractalSize == 512:
            block_cols = 32
        else:
            block_cols = 16
        assert(m % 16) == 0, "M should be 16 aligned when matrix C is NZ format"
        assert(n % block_cols) == 0, "N should be aligned when matrix C is NZ format"
        golden = golden.reshape((int(m/16), 16, int(n/block_cols),block_cols)).transpose(2, 0, 1, 3).astype(dst_type)
    elif dst_format == 'DN':
        golden = golden.transpose()
    golden.astype(dst_type).tofile("./golden.bin")


class tmovParams:
    def __init__(self, atype, btype, dstType, m, k, n, dst_format='ND', sFractalSize=512, isVQuant=False, isSQuant=False,
                 isRelu=False, quantType=None, scalar=1):
        self.atype = atype
        self.btype = btype
        self.ctype = np.float32
        if (atype == np.int8):
            self.ctype = np.int32
        self.dstType = dstType
        self.m = m
        self.k = k
        self.n = n
        self.dst_format = dst_format
        self.sFractalSize = sFractalSize
        self.isVQuant = isVQuant
        self.isSQuant = isSQuant
        self.isRelu = isRelu
        if (isVQuant):
            self.quantType = quantType
        if (isSQuant):
            self.scalar = scalar

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMOVTest.case_nz2nd_1",
        "TMOVTest.case_nz2nd_2",
        "TMOVTest.case_nz2nd_3",
        "TMOVTest.case_nz2nd_4",
        "TMOVTest.case_nz2nd_5",
        "TMOVTest.case_nz2nd_6",
        "TMOVTest.case_nz2nd_7",
        "TMOVTest.case_nz2nd_8",

        "TMOVTest.case_nz2nz_1",
        "TMOVTest.case_nz2nz_2",
        "TMOVTest.case_nz2nz_3",
        "TMOVTest.case_nz2nz_4",
        "TMOVTest.case_nz2nz_5",
        "TMOVTest.case_nz2nz_6",
        # Split
        "TMOVTest.case_nz2nz_7",
        "TMOVTest.case_nz2nz_8",
        "TMOVTest.case_nz2nz_9",
        "TMOVTest.case_nz2nz_10",

        "TMOVTest.case_nz2dn_1",
        "TMOVTest.case_nz2dn_2",
        "TMOVTest.case_nz2dn_3",
        "TMOVTest.case_nz2dn_4",
        # Quant pre
        "TMOVTest.case_nz2nz_fb_quant_1",
        "TMOVTest.case_nz2nz_fb_quant_2",
        "TMOVTest.case_nz2nz_fb_quant_3",
        "TMOVTest.case_nz2nz_fb_quant_4",

        "TMOVTest.case_nz2nz_sc_quant_1",
        "TMOVTest.case_nz2nz_sc_quant_2",
        "TMOVTest.case_nz2nz_sc_quant_3",
        "TMOVTest.case_nz2nz_sc_quant_4",

        "TMOVTest.case_nz2nd_fb_quant_1",
        "TMOVTest.case_nz2nd_fb_quant_2",
        "TMOVTest.case_nz2nd_fb_quant_3",
        "TMOVTest.case_nz2nd_fb_quant_4",
        "TMOVTest.case_nz2nd_fb_quant_5",

        "TMOVTest.case_nz2nd_sc_quant_1",
        "TMOVTest.case_nz2nd_sc_quant_2",
        "TMOVTest.case_nz2nd_sc_quant_3",
        "TMOVTest.case_nz2nd_sc_quant_4",

        "TMOVTest.case_nz2dn_fb_quant_1",
        "TMOVTest.case_nz2dn_fb_quant_2",
        "TMOVTest.case_nz2dn_fb_quant_3",
        "TMOVTest.case_nz2dn_fb_quant_4",

        "TMOVTest.case_nz2dn_sc_quant_1",
        "TMOVTest.case_nz2dn_sc_quant_2",
        "TMOVTest.case_nz2dn_sc_quant_3",
        "TMOVTest.case_nz2dn_sc_quant_4",
    ]

    case_params_list = [
        tmovParams(np.float16, np.float16, np.float32, 63, 127, 129, 'ND', 512, False, False, True),   # f32 -> f32
        tmovParams(np.float16, np.float16, np.float16, 112, 80, 112, 'ND', 512, False, False, True),   # f32 -> f16
        tmovParams(np.float16, np.float16, np.float32, 63, 63, 63),
        tmovParams(np.float16, np.float16, np.float32, 111, 48, 88),
        tmovParams(np.float16, np.float16, np.float32, 65, 40, 80),     # sub block id = 1
        tmovParams(np.float16, np.float16, bfloat16, 111, 47, 96),     # f32 -> bf16
        tmovParams(np.float16, np.float16, np.float32, 96, 32, 48),     # split m
        tmovParams(np.float16, np.float16, np.float32, 48, 32, 128),     # split n

        tmovParams(np.float16, np.float16, np.float32, 80, 112, 48, 'NZ'),  # nz2nz,f16@f16->f32
        tmovParams(np.float16, np.float16, np.float32, 112, 128, 80, 'NZ'),  # nz2nz,f16@f16->f32
        tmovParams(np.float16, np.float16, np.float16, 96, 80, 112, 'NZ'),  # nz2nz,f16@f16->f32->f16 in ub
        tmovParams(np.float16, np.float16, np.float32, 80, 112, 96, 'NZ', 1024), # nz2nz,f16@f16->f32  1024
        tmovParams(np.float32, np.float32, np.float32, 112, 96, 80, 'NZ', 512, False, False, True), # nz2nz,f32@f32->f32
        tmovParams(np.float16, np.float16, bfloat16, 48, 112, 48, 'NZ', 512, False, False, True),     # nz2nz,f16@f16-> f32 -> bf16
        # Split
        tmovParams(np.float32, np.float32, np.float32, 32, 16, 32, 'NZ', 1024),  # nz2nz.f32@f32->f32.split n
        tmovParams(np.float16, np.float16, np.float32, 128, 16, 64, 'NZ', 1024),  # nz2nz.f16@f16->f32.split n
        tmovParams(np.float16, np.float16, np.float32, 32, 16, 32, 'NZ', 1024),  # nz2nz.f16@f16->f32.split m
        tmovParams(np.float32, np.float32, np.float32, 128, 128, 64, 'NZ', 1024),  # nz2nz.f32@f32->f32.split m

        tmovParams(np.float32, np.float32, np.float32, 80, 40, 66, 'DN'),      # f32 -> f32
        tmovParams(np.float16, np.float16, np.float16, 88, 48, 95, 'DN'),      # f32 -> f16
        tmovParams(np.float16, np.float16, bfloat16, 48, 31, 31, 'DN', 512, False, False, True),  # f32 -> bf16
        tmovParams(np.float16, np.float16, np.float32, 88, 48, 95, 'DN', 512, False, False, True),      # f32 -> f32

        tmovParams(np.int8, np.int8, np.int8, 32, 32, 128, 'NZ', 512, True, False, False, np.uint64),  # nz2nz.b8@b8->b32 in L0C, quant (int32 -> int8)
        tmovParams(np.int8, np.int8, np.float16, 128, 64, 128, 'NZ', 512, True, False, False, np.uint64),  # nz2nz.vector quant (int32 -> half)
        tmovParams(np.float32, np.float32, np.int8, 64, 32, 128, 'NZ', 512, True, False, True, np.uint64),  # nz2nz.vector quant (float -> int8)
        tmovParams(np.float32, np.float32, np.float16, 64, 32, 64, 'NZ', 512, True, False, True, np.uint64),  # nz2nz.vector quant (float -> half)

        tmovParams(np.float32, np.float32, np.float16, 128, 32, 64, 'NZ', 512, False, True, True, None, 2),     # nz2nz.scalar quant (float -> half)
        tmovParams(np.int8, np.int8, np.float16, 32, 128, 64, 'NZ', 512, False, True, True, None, 4),           # nz2nz.scalar quant (int32 -> half)
        tmovParams(np.int8, np.int8, np.int8, 32, 32, 128, 'NZ', 512, False, True, False, None, 5),              # nz2nz.scalar quant (int32 -> int8)
        tmovParams(np.float32, np.float32, np.int8, 32, 32, 64, 'NZ', 512, False, True, False, None, 7),         # nz2nz.scalar quant (float -> int8)

        tmovParams(np.int8, np.int8, np.int8, 30, 48, 64, 'ND', 512, True, False, False, np.uint64),              # vector quant (int32 -> int8)
        tmovParams(np.int8, np.int8, np.float16, 60, 128, 32, 'ND', 512, True, False, False, np.uint64),           # vector quant (int32 -> half)
        tmovParams(np.int8, np.int8, bfloat16, 128, 64, 96, 'ND', 512, True, False, False, np.uint64),             # vector quant (int32 -> bf16)
        tmovParams(np.float32, np.float32, np.int8, 60, 128, 64, 'ND', 512, True, False, True, np.uint64),        # vector quant (float -> int8)
        tmovParams(np.float32, np.float32, np.float16, 31, 128, 128, 'ND', 512, True, False, True, np.uint64),    # vector quant (float -> half)

        tmovParams(np.float32, np.float32, np.float16, 128, 48, 96, 'ND', 512, False, True, True, None, 2),   # scalar quant (float -> half)
        tmovParams(np.float32, np.float32, np.int8, 60, 128, 64, 'ND', 512, False, True, True, None, 5),      # scalar quant (float -> int8)
        tmovParams(np.int8, np.int8, np.float16, 30, 48, 64, 'ND', 512, False, True, False, None, 3),         # scalar quant (int32 -> half)
        tmovParams(np.int8, np.int8, np.int8, 60, 128, 32, 'ND', 512, False, True, False, None, 1),             # scalar quant (int32 -> int8)

        tmovParams(np.int8, np.int8, np.int8, 96, 128, 60, 'DN', 512, True, False, False, np.uint64),         # vector quant (int32 -> int8)
        tmovParams(np.int8, np.int8, np.float16, 32, 48, 64, 'DN', 512, True, False, False, np.uint64),       # vector quant (int32 -> half)
        tmovParams(np.float16, np.float16, np.int8, 32, 128, 60, 'DN', 512, True, False, True, np.uint64),   # vector quant (float -> int8)
        tmovParams(np.float16, np.float16, np.float16, 64, 64, 90, 'DN', 512, True, False, True, np.uint64),  # vector quant (float -> half)

        tmovParams(np.float32, np.float32, np.float16, 80, 40, 66, 'DN', 512, False, True, True, None, 2),   # scalar quant (float -> half)
        tmovParams(np.float32, np.float32, np.int8, 96, 128, 60, 'DN', 512, False, True, True, None, 5),      # scalar quant (float -> int8)
        tmovParams(np.int8, np.int8, np.float16, 32, 128, 64, 'DN', 512, False, True, False, None, 3),         # scalar quant (int32 -> half)
        tmovParams(np.int8, np.int8, np.int8, 64, 64, 90, 'DN', 512, False, True, False, None, 1),             # scalar quant (int32 -> int8)
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)