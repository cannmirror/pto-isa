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
np.random.seed(19)

def matmul_reference(a, b, out_dtype):
    """
    Reference matmul that avoids BLAS calls (some macOS Python distributions may
    ship a broken/unsupported BLAS backend that returns incorrect results).

    a: (m, k)
    b: (k, n)
    returns: (m, n)
    """
    a = a.astype(out_dtype, copy=False)
    b = b.astype(out_dtype, copy=False)
    # (m, k, 1) * (1, k, n) -> (m, k, n) -> sum over k
    return (a[:, :, None] * b[None, :, :]).sum(axis=1, dtype=out_dtype)

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, False, False, False
    repeats = param.repeats

    x1_gm = np.random.randint(1, 5, [repeats, m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [repeats, k, n]).astype(src_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(dst_type)
    golden=np.zeros([m,n], dst_type)

    for i in range(repeats):
        golden = golden + matmul_reference(x1_gm[i], x2_gm[i], dst_type).astype(dst_type)

        if is_atrans:
            x1_gm[i] = x1_gm[i].transpose()
        if is_btrans:
            x2_gm[i] = x2_gm[i].transpose()

    if is_bias:
        golden += bias_gm

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, repeats):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.repeats = repeats


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case1", # 此名称要和TEST_F(TMATMULTest, case1)定义的名称一致
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULTest.case4",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, 1),
        tmatmulParams(np.int8, np.int8, np.int32, 128, 128, 64, 1),
        tmatmulParams(np.float16, np.float16, np.float32, 128, 128, 64, 5),
        tmatmulParams(np.float32, np.float32, np.float32, 32, 16, 32, 1),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
