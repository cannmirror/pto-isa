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
fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
np.random.seed(19)


def check(x, n):
    if len(x) < n:
        x = '0' * (n - len(x)) + x
    elif len(x) > n:
        x = x[1:]
    return x


def cast(c, dtype):
    if dtype == 'fp16':
        c = np.array(c).astype(np.float16)
    elif dtype == 'fp32':
        c = np.array(c).astype(np.float32)
    return c


def hif8_convert(input):
    d, e = '', ''
    s, m = input[0], input[5:]
    m1, m2, m3 = int(input[5]), int(input[6]), int(input[7])
    if input[1] == '1' or input[2] == '1':
        d, e = input[1:3], input[3:5]
    elif input[3] == '1':
        d, e = input[1:4], input[4]
    else:
        d, e = input[1:5], ''
    f1 = -1 if s == '1' else 1
    f2 = 1
    if d == '0000':
        if s == '1':
            if m == '000':
                return np.nan
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23) * f1
        else:
            if m == '000':
                return 0
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23)
        return input
    elif d == '0001':
        f2 = 0
        input = (1 + (m1 * 4 + m2 * 2 + m3) / 8) * 2 ** f2 * f1
        return input
    elif d == '001':
        f2 = -1 if e == '1' else 1
        input = (1 + (m1 * 4 + m2 * 2 + m3) / 8) * 2 ** f2 * f1
        return input
    elif d == '01':
        f2 = -1 if int(input[3]) == 1 else 1
        input = (1 + (m1 * 4 + m2 * 2 + m3) / 8) * 2 ** (f2 * (2 + int(input[4]))) * f1
        return input
    elif d == '10':
        f2 = -1 if int(input[3]) == 1 else 1
        input = (1 + (m2 * 2 + m3) / 4) * 2 ** (f2 * (4 + int(input[4]) * 2 + int(input[5]))) * f1
        return input
    elif d == '11':
        f2 = -1 if int(input[3]) == 1 else 1
        if e == '01' and m == '111':
            return f1 * np.inf
        input = (1 + m3 / 2) * 2 ** (f2 * (8 + int(input[4]) * 4 + int(input[5]) * 2 + int(input[6]))) * f1
        return input


def get_hif8_golden(x1_gm, x2_gm, start_m, start_k, start_n, dst_type):
    s1 = x1_gm.reshape(-1)
    s2 = x2_gm.reshape(-1)
    s1_len = len(s1)
    s2_len = len(s2)
    re1 = [0] * s1_len
    re2 = [0] * s2_len
    for i in range(s1_len):
        temp = bin(s1[i])
        temp = temp.split('b')[1]
        temp = check(temp, 8)
        re1[i] = hif8_convert(temp)
    s1 = cast(re1, 'fp32')
    for i in range(s2_len):
        temp = bin(s2[i])
        temp = temp.split('b')[1]
        temp = check(temp, 8)
        re2[i] = hif8_convert(temp)
    s2 = cast(re2, 'fp32')
    x1_gm = s1.reshape(x1_gm.shape)
    x2_gm = s2.reshape(x2_gm.shape)
    x1_slice = x1_gm[start_m:, start_k:]
    x2_slice = x2_gm[start_k:, start_n:]
    golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)
    return golden


def create_padded_tensors(
    x1_gm, x2_gm, m, n, k, base_m, base_n, base_k, src_type=np.int8, 
    rand_range_right=(1, 5), 
    rand_range_down=(1, 5), 
    rand_range_corner=(1, 5)):
    assert base_m >= m, f"base_m ({base_m}) mast be >= m ({m})"
    assert base_n >= n, f"base_n ({base_n}) mast be >= n ({n})"
    assert base_k >= k, f"base_k ({base_k}) mast be >= k ({k})"
    # x1_gm_padded：base_m, base_k
    x1_gm_padded = np.zeros((base_m, base_k), dtype=np.int32).astype(src_type)
    # origin data
    x1_gm_padded[:m, :k] = x1_gm
    # Right-side random value padding (k-direction extension)
    right_fill = np.random.randint(rand_range_right[0], rand_range_right[1],
                                    size=(m, base_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[:m, k:base_k] = right_fill
    # Add 0 to the bottom (extended in the m direction)
    x1_gm_padded[m:base_m, :k] = 0

    # Add random value in the bottom right corner
    corner_fill = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                    size=(base_m - m, base_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[m:base_m, k:base_k] = corner_fill
    #x2_gm_padded：base_k, base_n
    x2_gm_padded = np.zeros((base_k, base_n), dtype=np.int32).astype(src_type)
    x2_gm_padded[:k, :n] = x2_gm
    down_fill = np.random.randint(rand_range_down[0], rand_range_down[1],
                                    size=(base_k - k, n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:base_k, :n] = down_fill
    x2_gm_padded[:k, n:base_n] = 0
    corner_fill2 = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                     size=(base_k - k, base_n - n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:base_k, n:base_n] = corner_fill2
    return x1_gm_padded, x2_gm_padded


def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, start_m, start_k, start_n, is_bias, is_atrans, is_btrans, base_m, base_k, base_n = \
        param.m, param.k, param.n, param.start_m, param.start_k, param.start_n, False, param.is_atrans, \
        param.is_btrans, param.base_m, param.base_k, param.base_n

    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    # get slice
    x1_slice = x1_gm[start_m:, start_k:]  # from (rowIdx1, colIdx1) to the end
    x2_slice = x2_gm[start_k:, start_n:]  # from (rowIdx2, colIdx2) to the end
    golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)
    # hifloat8_t processing
    if (param.atype == np.uint8):
        golden = get_hif8_golden(x1_gm, x2_gm, start_m, start_k, start_n, dst_type)
    # padding for unaligned data
    if base_m > 0 or base_n > 0 or base_k > 0:
        base_m = base_m if base_m > 0 else m
        base_n = base_n if base_n > 0 else n
        base_k = base_k if base_k > 0 else k
        x1_gm, x2_gm = create_padded_tensors(x1_gm, x2_gm, m, n, k, base_m, base_n, base_k, src_type, \
                    rand_range_right=(1, 5), rand_range_down=(1, 5), rand_range_corner=(1, 5))
    if is_atrans:
        x1_gm = x1_gm.transpose()
    if not is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class textractParams:
    def __init__(self, atype, btype, ctype, m, k, n, start_m, start_k, start_n, \
        is_atrans = 0, is_btrans = 0, base_m = 0, base_k = 0, base_n = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.start_m = start_m
        self.start_k = start_k
        self.start_n = start_n
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans
        self.base_m = base_m
        self.base_k = base_k
        self.base_n = base_n

if __name__ == "__main__":
    case_name_list = [
        "TEXTRACTTest.case11",
        "TEXTRACTTest.case12",
        "TEXTRACTTest.case13",
        "TEXTRACTTest.case14",
        "TEXTRACTTest.case21",
        "TEXTRACTTest.case22",
        "TEXTRACTTest.case23",
        "TEXTRACTTest.case24",
        "TEXTRACTTest.case31",
        "TEXTRACTTest.case32",
        "TEXTRACTTest.case33",
        "TEXTRACTTest.case34",
        "TEXTRACTTest.case41",
        "TEXTRACTTest.case42",
        "TEXTRACTTest.case43",
        "TEXTRACTTest.case44",
        "TEXTRACTTest.case51",
        "TEXTRACTTest.case52",
        "TEXTRACTTest.case53",
        "TEXTRACTTest.case54",
        "TEXTRACTTest.case61",
        "TEXTRACTTest.case62",
        "TEXTRACTTest.case63",
        "TEXTRACTTest.case64",
        "TEXTRACTTest.case71",
        "TEXTRACTTest.case72",
        "TEXTRACTTest.case73",
        "TEXTRACTTest.case74",
    ]

    case_params_list = [
        # float16
        textractParams(np.float16, np.float16, np.float32, 63, 48, 66, 0, 0, 0, 0, 0, 128, 64, 256),
        textractParams(np.float16, np.float16, np.float32, 68, 93, 97, 0, 0, 0, 1, 1, 128, 128, 128),
        textractParams(np.float16, np.float16, np.float32, 75, 201, 79, 16, 16, 16, 0, 0, 80, 256, 80),
        textractParams(np.float16, np.float16, np.float32, 59, 232, 61, 16, 16, 16, 1, 1, 64, 256, 64),
        # float32
        textractParams(np.float32, np.float32, np.float32, 68, 70, 69, 0, 0, 0, 0, 0, 80, 128, 80),
        textractParams(np.float32, np.float32, np.float32, 20, 22, 21, 0, 0, 0, 1, 1, 64, 96, 64),
        textractParams(np.float32, np.float32, np.float32, 49, 119, 63, 16, 32, 16, 0, 0, 64, 128, 64),
        textractParams(np.float32, np.float32, np.float32, 127, 60, 102, 16, 16, 32, 1, 1, 128, 64, 128),
        # int8
        textractParams(np.int8, np.int8, np.int32, 97, 231, 83, 0, 0, 0, 0, 0, 128, 256, 128),
        textractParams(np.int8, np.int8, np.int32, 71, 188, 82, 0, 0, 0, 1, 1, 128, 256, 128),
        textractParams(np.int8, np.int8, np.int32, 63, 112, 98, 32, 32, 32, 0, 0, 64, 128, 128),
        textractParams(np.int8, np.int8, np.int32, 106, 125, 60, 32, 32, 32, 1, 1, 128, 128, 64),
        # bfloat16
        textractParams(bfloat16, bfloat16, np.float32, 23, 24, 25, 0, 0, 0, 0, 0, 96, 64, 96),
        textractParams(bfloat16, bfloat16, np.float32, 23, 24, 25, 0, 0, 0, 1, 1, 96, 64, 96),
        textractParams(bfloat16, bfloat16, np.float32, 39, 40, 41, 16, 16, 16, 0, 0, 96, 64, 96),
        textractParams(bfloat16, bfloat16, np.float32, 39, 40, 41, 16, 16, 16, 1, 1, 96, 64, 96),
        # hif8
        textractParams(np.uint8, np.uint8, np.float32, 46, 40, 45, 0, 0, 0, 0, 0, 128, 96, 128),
        textractParams(np.uint8, np.uint8, np.float32, 46, 40, 45, 0, 0, 0, 1, 1, 128, 96, 128),
        textractParams(np.uint8, np.uint8, np.float32, 78, 72, 77, 32, 32, 32, 0, 0, 128, 96, 128),
        textractParams(np.uint8, np.uint8, np.float32, 78, 72, 77, 32, 32, 32, 1, 1, 128, 96, 128),
        # fp8_e4m3fn
        textractParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 46, 40, 45, 0, 0, 0, 0, 0, 128, 96, 128),
        textractParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 46, 40, 45, 0, 0, 0, 1, 1, 128, 96, 128),
        textractParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 78, 72, 77, 32, 32, 32, 0, 0, 128, 96, 128),
        textractParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 78, 72, 77, 32, 32, 32, 1, 1, 128, 96, 128),
        # fp8_e5m2
        textractParams(fp8_e5m2, fp8_e5m2, np.float32, 46, 40, 45, 0, 0, 0, 0, 0, 128, 96, 128),
        textractParams(fp8_e5m2, fp8_e5m2, np.float32, 46, 40, 45, 0, 0, 0, 1, 1, 128, 96, 128),
        textractParams(fp8_e5m2, fp8_e5m2, np.float32, 78, 72, 77, 32, 32, 32, 0, 0, 128, 96, 128),
        textractParams(fp8_e5m2, fp8_e5m2, np.float32, 78, 72, 77, 32, 32, 32, 1, 1, 128, 96, 128),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
