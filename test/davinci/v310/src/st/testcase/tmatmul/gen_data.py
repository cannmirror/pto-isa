#!/usr/bin/python3
# coding=utf-8

import os
import struct
import math
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

def HF8(input):
    if len(input) < 8:
        print("输入不满足8bit位，没法转换，请补齐")
        exit(-1)

    if len(input) > 8:
        print("输入超越8bit位，没法转换，请确认输入")
        exit(-1)
    d = ''
    e = ''
    s = input[0]
    m = input[5:]
    m1 = int(input[5])
    m2 = int(input[6])
    m3 = int(input[7])
    if input[1] == '1' or input[2] == '1':
        d = input[1:3]
        e = input[3:5]
    elif input[3] == '1':
        d = input[1:4]
        e = input[4]
    else:
        d = input[1:5]
        e = ''
    f1 = 1
    f2 = 1
    if d == '0000':
        if s == '1':
            f1 = -1
            if m == '000':
                return np.nan
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23) * f1
        else:
            if m == '000':
                return 0
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23)
        return input
    elif d == '0001':
        if s == '1':
            f1 = -1
        f2 = 0
        input = (1 + (m1 * 4 + m2 * 2 + m3) / 8) * 2 ** f2 * f1
        return input
    elif d == '001':
        if s == '1':
            f1 = -1
        if e == '1':
            f2 = -1
        input = (1 + (m1 * 4 + m2 * 2 + m3) / 8) * 2 ** f2 * f1
        return input
    elif d == '01':
        if s == '1':
            f1 = -1
        e1 = int(input[3])
        e2 = int(input[4])
        if e1 == 1:
            f2 = -1
        input = (1 + (m1 * 4 + m2 * 2 + m3) / 8) * 2 ** (f2 * (2 + e2)) * f1
        return input
    elif d == '10':
        if s == '1':
            f1 = -1
        e1 = int(input[3])
        e2 = int(input[4])
        e3 = int(input[5])
        if e1 == 1:
            f2 = -1
        input = (1 + (m2 * 2 + m3) / 4) * 2 ** (f2 * (4 + e2 * 2 + e3)) * f1
        return input
    elif d == '11':
        if s == '1':
            f1 = -1
        e1 = int(input[3])
        e2 = int(input[4])
        e3 = int(input[5])
        e4 = int(input[6])
        if e1 == 1:
            f2 = -1
        if e == '01' and m == '111':
            return f1 * np.inf
        input = (1 + m3/2) * 2 ** (f2 * (8 + e2 * 4 + e3 * 2 + e4)) * f1
        return input

def gen_golden_data(case_name, param):
    is_hifloat = False
    a_type = param.atype
    b_type = param.btype
    if (a_type == np.uint8):
        is_hifloat = True
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False 
    x1_gm = np.random.randint(1, 5, [m, k]).astype(a_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(b_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(bias_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if (is_hifloat):
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
            re1[i] = HF8(temp)
        s1 = cast(re1, 'fp32')
        for i in range(s2_len):
            temp = bin(s2[i])
            temp = temp.split('b')[1]
            temp = check(temp, 8)
            re2[i] = HF8(temp)            
        s2 = cast(re2, 'fp32')
        x1_gm = s1.reshape(x1_gm.shape)
        x2_gm = s2.reshape(x2_gm.shape)

    if is_bias:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type)

    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.is_bias = is_bias
        if bias_type:
            self.bias_type = bias_type
        else:
            self.bias_type = ctype

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case1", # 此名称要和TEST_F(TMATMULTest, case1)定义的名称一致
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULTest.case4",
        "TMATMULTest.case5",
        "TMATMULTest.case6",
        "TMATMULTest.case7",
        "TMATMULTest.case8",
        "TMATMULTest.case9",
        "TMATMULTest.case10",

        "TMATMULBIASTest.case1",
        "TMATMULBIASTest.case2",
        "TMATMULBIASTest.case3",
        "TMATMULBIASTest.case4",
        "TMATMULBIASTest.case5",
        "TMATMULBIASTest.case6",
        "TMATMULBIASTest.case7",
        "TMATMULBIASTest.case8",
        "TMATMULBIASTest.case9",
        "TMATMULBIASTest.case10",
        "TMATMULBIASTest.case11",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 127, 128, 64, False), # half * half -> float 128
        tmatmulParams(np.int8, np.int8, np.int32, 128, 127, 64, False),
        tmatmulParams(np.float16, np.float16, np.float32, 127, 128, 61, False),
        tmatmulParams(np.float32, np.float32, np.float32, 127, 127, 63, False),
        tmatmulParams(bfloat16, bfloat16, np.float32, 128, 128, 64, False),
        tmatmulParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 128, 128, 64, False),
        tmatmulParams(fp8_e4m3fn, fp8_e5m2, np.float32, 128, 128, 64, False),
        tmatmulParams(fp8_e5m2, fp8_e4m3fn, np.float32, 128, 128, 64, False),
        tmatmulParams(fp8_e5m2, fp8_e5m2, np.float32, 128, 128, 64, False),
        tmatmulParams(np.uint8, np.uint8, np.float32, 128, 128, 64, False),

        tmatmulParams(np.int8, np.int8, np.int32, 128, 128, 64, True),
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float32, 128, 127, 64, True, bfloat16),
        tmatmulParams(bfloat16, bfloat16, np.float32, 128, 128, 63, True, bfloat16),
        tmatmulParams(np.float16, np.float16, np.float32, 127, 128, 63, True),
        tmatmulParams(np.float32, np.float32, np.float32, 127, 128, 63, True),
        tmatmulParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 128, 128, 64, True),
        tmatmulParams(fp8_e4m3fn, fp8_e5m2, np.float32, 128, 128, 64, True),
        tmatmulParams(fp8_e5m2, fp8_e4m3fn, np.float32, 128, 128, 64, True),
        tmatmulParams(fp8_e5m2, fp8_e5m2, np.float32, 128, 128, 64, True),
        tmatmulParams(np.uint8, np.uint8, np.float32, 128, 128, 64, True),     
    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)    