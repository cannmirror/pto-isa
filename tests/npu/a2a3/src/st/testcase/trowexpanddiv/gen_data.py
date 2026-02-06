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
np.random.seed(2025)


def gen_golden_data(case_name, param):
    dtype = param.datatype

    vr, vc = param.valid_row, param.valid_col
    src1vc = 1
    if param.is_rowmajor:
        src1vc = 32 // np.dtype(dtype).itemsize

    input1 = np.random.random(vr * vc).astype(dtype)
    input2 = np.random.random(vr * src1vc).astype(dtype)
    golden = np.zeros(vr * vc).astype(dtype)

    if param.src0eqdst:
        for i in range(vr):
            for j in range(vc):
                golden[i * vc + j] = input1[i * vc + j] / input2[i * src1vc + j % src1vc]
    else:
        for i in range(vr):
            for j in range(vc):
                golden[i * vc + j] = input2[i * src1vc + j % src1vc] / input1[i * vc + j]
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TRowExpandDiv:
    def __init__(self, datatype, valid_row, valid_col, row, col, src0eqdst, is_rowmajor):
        self.datatype = datatype
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.row = row
        self.col = col
        self.src0eqdst = src0eqdst
        self.is_rowmajor = is_rowmajor


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TROWEXPANDDIVTest.case1",
        "TROWEXPANDDIVTest.case2",
        "TROWEXPANDDIVTest.case3",
        "TROWEXPANDDIVTest.case4",
        "TROWEXPANDDIVTest.case5",
        "TROWEXPANDDIVTest.case6",
        "TROWEXPANDDIVTest.case7",
        "TROWEXPANDDIVTest.case8",
        "TROWEXPANDDIVTest.case9",
        "TROWEXPANDDIVTest.case10",
        "TROWEXPANDDIVTest.case11",
        "TROWEXPANDDIVTest.case12",
        "TROWEXPANDDIVTest.case13",
        "TROWEXPANDDIVTest.case14",
    ]

    case_params_list = [
        TRowExpandDiv(np.float32, 16, 16, 16, 16, True, False),
        TRowExpandDiv(np.float32, 16, 16, 32, 32, True, False),
        TRowExpandDiv(np.float16, 16, 16, 16, 16, True, False),
        TRowExpandDiv(np.float16, 16, 16, 32, 32, True, False),
        TRowExpandDiv(np.float32, 1, 16384, 1, 16384, True, False),
        TRowExpandDiv(np.float32, 2048, 1, 2048, 8, True, False),
        TRowExpandDiv(np.float32, 16, 16, 16, 16, True, True),
        TRowExpandDiv(np.float32, 16, 16, 32, 32, True, True),
        TRowExpandDiv(np.float16, 16, 16, 16, 16, True, True),
        TRowExpandDiv(np.float16, 16, 16, 32, 32, True, True),
        TRowExpandDiv(np.float32, 1, 16384, 1, 16384, True, True),
        TRowExpandDiv(np.float32, 2048, 1, 2048, 8, True, True),
        TRowExpandDiv(np.float32, 16, 16, 16, 16, False, False),
        TRowExpandDiv(np.float32, 16, 16, 16, 16, False, True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
