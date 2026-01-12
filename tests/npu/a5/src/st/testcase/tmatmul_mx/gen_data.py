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
import math
import numpy as np
import ml_dtypes
import en_dtypes

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = en_dtypes.float4_e1m2
fp4_e2m1x2 = en_dtypes.float4_e2m1

np.random.seed(19)


def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, 
                       ((0, pad_m), (0, pad_k)), 
                       mode='constant',
                       constant_values=0)
    else:
        padded = x1_mx_gm
    
    m_padded = m + pad_m
    k_padded = k + pad_k

    x1_scale_gm = padded.reshape((int(m_padded / block_size), block_size, 
                                 int(k_padded / c0_size_mx), c0_size_mx))
    x1_scale_gm = x1_scale_gm.transpose(0, 2, 1, 3)
    x1_scale_gm = x1_scale_gm.reshape(x1_scale_gm.shape[0] * x1_scale_gm.shape[1], 
                                     x1_scale_gm.shape[2] * x1_scale_gm.shape[3])

    return x1_scale_gm


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, 
                       ((0, pad_k), (0, pad_n)),
                       mode='constant',
                       constant_values=0)
    else:
        padded = x2_mx_gm
    
    k_padded, n_padded = padded.shape
    
    x2_scale_gm = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    x2_scale_gm = x2_scale_gm.reshape(x2_scale_gm.shape[1] * x2_scale_gm.shape[3], 
                                      x2_scale_gm.shape[0] * x2_scale_gm.shape[2])

    return x2_scale_gm


def pack_two_fp4(scale_matrix):
    scale_matrix_row = scale_matrix.shape[0]
    scale_matrix_col = scale_matrix.shape[1]
    scale_matrix_bin = scale_matrix.flatten()
    scale_matrix_high = scale_matrix_bin[::2].view(np.uint8)
    scale_matrix_low = scale_matrix_bin[1::2].view(np.uint8)
    low_bits = (scale_matrix_low & 0x0F) << 4
    high_bits = scale_matrix_high & 0x0F
    combined = low_bits | high_bits
    scale_matrix_bin = combined.reshape(scale_matrix_row, scale_matrix_col // 2)
    return scale_matrix_bin


def align_to_multiple(k, alignment=64):
    return (k + alignment - 1) // alignment * alignment


def gen_golden_data(case_name, param):

    a_type = param.atype
    b_type = param.btype
   
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False

    original_k = k
    if k % 64 != 0:
        k_aligned = align_to_multiple(k, 64)
    else:
        k_aligned = original_k

    if a_type == fp4_e2m1x2:
        if original_k % 64 != 0:
            x1_gm_original = np.random.randint(-7, 7, [m, original_k]).astype(a_type)
            x1_gm = np.zeros([m, k_aligned], dtype=a_type)
            x1_gm[:, :original_k] = x1_gm_original

        else:
            x1_gm = np.random.randint(-7, 7, [m, original_k]).astype(a_type)

        x1_gm_bin = pack_two_fp4(x1_gm)
        x1_gm_bin.tofile("./x1_gm.bin")
    elif a_type == fp4_e1m2x2:
        if original_k % 64 != 0:

            x1_gm_original = np.random.randint(-2, 2, [m, original_k]).astype(a_type)
            x1_gm = np.zeros([m, k_aligned], dtype=a_type)
            x1_gm[:, :original_k] = x1_gm_original

        else:
            x1_gm = np.random.randint(-2, 2, [m, original_k]).astype(a_type)

        x1_gm_bin = pack_two_fp4(x1_gm)
        x1_gm_bin.tofile("./x1_gm.bin")
    else:
        if k % 64 != 0:
            x1_gm_original = np.random.randint(1, 5, [m, original_k]).astype(a_type)
            x1_gm = np.zeros([m, k_aligned], dtype=a_type)
            x1_gm[:, :original_k] = x1_gm_original

        else:
            x1_gm = np.random.randint(1, 5, [m, original_k]).astype(a_type)

        x1_gm.tofile("./x1_gm.bin")

    if b_type == fp4_e2m1x2:
        x2_gm = np.random.randint(-7, 7, [k_aligned, n]).astype(b_type)
        if original_k % 64 != 0:
            x2_gm[original_k:, :] = 0

        x2_gm_bin = pack_two_fp4(x2_gm)
        x2_gm_bin.tofile("./x2_gm.bin")
    elif b_type == fp4_e1m2x2:
        x2_gm = np.random.randint(-2, 2, [k_aligned, n]).astype(b_type)
        if original_k % 64 != 0:
            x2_gm[original_k:, :] = 0
        x2_gm_bin = pack_two_fp4(x2_gm)
        x2_gm_bin.tofile("./x2_gm.bin")
    else:
        x2_gm = np.random.randint(1, 5, [k_aligned, n]).astype(b_type)
        if original_k % 64 != 0:
            x2_gm[original_k:, :] = 0
        x2_gm.tofile("./x2_gm.bin")
    
    x1_mx_gm = np.random.randint(127, 130, [m, math.ceil(k_aligned / 32)]).astype(np.uint8)
    x2_mx_gm = np.random.randint(127, 130, [math.ceil(k_aligned / 32), n]).astype(np.uint8)

    ###################### compute ########################
    x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
    x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
    x1_full = np.zeros([m, k_aligned], dtype=np.float64)
    x2_full = np.zeros([k_aligned, n], dtype=np.float64)

    for i in range(x1_gm.shape[1]):
        x1_full[:, i] = x1_gm[:, i] * x1_mx[:, i // 32]
        x2_full[i, :] = x2_gm[i, :] * x2_mx[i // 32, :]

    x1 = x1_full[:, :original_k] if original_k < k_aligned else x1_full
    x2 = x2_full[:original_k, :] if original_k < k_aligned else x2_full
    # x1_scale_gm, convert to zZ format
    x1_scale_gm = convert_x1_scale_format(x1_mx_gm, 16, 2)
    # x1_scale_gm, convert to nN format
    x2_scale_gm = convert_x2_scale_format(x2_mx_gm, 16, 2)

    x1_scale_gm.tofile("./x1_mx_gm.bin")
    x2_scale_gm.tofile("./x2_mx_gm.bin")
    if is_bias:
        bias_gm = np.random.randint(0, 1, [n, ]).astype(bias_type)
        bias_gm.tofile("./bias_gm.bin")
        golden = np.matmul(x1.astype(np.float64), x2.astype(np.float64)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1.astype(np.float64), x2.astype(np.float64)).astype(dst_type)

    golden.tofile("./golden.bin")


class TmatmulmxParams:

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
    case_name_list = [
        "TMATMULMXTest.case_e5m2_e5m2_128_64_64",
        "TMATMULMXTest.case_e4m3_e4m3_127_72_64",
        "TMATMULMXTest.case_e4m3_e5m2_128_110_63",
        "TMATMULMXTest.case_e2m1_e2m1_128_64_64",
        "TMATMULMXTest.case_e1m2_e2m1_117_64_60",
        "TMATMULMXTest.case_e2m1_e1m2_128_118_64",
        "TMATMULMXTest.case_e2m1_e1m2_115_64_30",
        "TMATMULMXTest.case_e4m3_e4m3_16_32_16",
        "TMATMULMXTest.case_e4m3_e5m2_10_50_54",
        "TMATMULMXTest.case_e2m1_e2m1_4_30_8",
        # bias test
        "TMATMULMXTest.case_e5m2_e4m3_115_64_30",
        "TMATMULMXTest.case_e4m3_e4m3_200_192_95",
        "TMATMULMXTest.case_e2m1_e1m2_35_128_56",
        # bias + acc test
        "TMATMULMXTest.case_e1m2_e1m2_47_128_62",
        "TMATMULMXTest.case_e4m3_e5m2_64_65_64",
        # gemv mode
        "TMATMULMXTest.case_e1m2_e1m2_1_64_62",
    ]

    case_params_list = [
        TmatmulmxParams(fp8_e5m2, fp8_e5m2, np.float32, 128, 64, 64, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 127, 72, 64, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 128, 110, 63, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 128, 64, 64, False),
        TmatmulmxParams(fp4_e1m2x2, fp4_e2m1x2, np.float32, 117, 64, 60, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 128, 118, 64, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 115, 64, 30, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 16, 32, 16, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 10, 50, 54, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 4, 30, 8, False),
        # bias test
        TmatmulmxParams(fp8_e5m2, fp8_e4m3fn, np.float32, 115, 64, 30, True),
        TmatmulmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 200, 192, 95, True),
        TmatmulmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 35, 128, 56, True),
        # bias + acc test
        TmatmulmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 47, 128, 62, True),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 64, 65, 64, True),
        # gemv mode
        TmatmulmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 1, 64, 62, True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)