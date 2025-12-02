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


def gen_golden_data_tpartadd(case_name, param):
    dtype = param.dtype

    dstRow, dstCol = [param.dstVR, param.dstVC]
    src0Row, src0Col = [param.src0VR, param.src0VC]
    src1Row, src1Col = [param.src1VR, param.src1VC]

    # Generate random input arrays
    input1 = np.random.random(src0Row*src0Col).astype(dtype)
    input2 = np.random.random(src1Row*src1Col).astype(dtype)

    # Perform the addbtraction
    golden = np.zeros([dstRow*dstCol]).astype(dtype)
    if dstRow == src0Row and dstRow == src1Row and dstCol == src0Col and dstCol == src1Col:
        golden = input1 + input2
    elif dstRow > src0Row and dstRow == src1Row and dstCol == src0Col and dstCol == src1Col:
        for i in range(0, src0Row*src0Col):
            golden[i] = input1[i]+input2[i]
        for i in range(src0Row*src0Col, dstRow*dstCol):
            golden[i] = input2[i]
    elif dstRow == src0Row and dstRow > src1Row and dstCol == src0Col and dstCol == src1Col:
        for i in range(0, src1Row*src1Col):
            golden[i] = input1[i]+input2[i]
        for i in range(src1Row*src1Col, dstRow*dstCol):
            golden[i] = input1[i]
    elif dstRow == src0Row and dstRow == src1Row and dstCol > src0Col and dstCol == src1Col:
        for i in range(0, src0Row):
            for j in range(0, src0Col):
                golden[j + i*dstCol] = input1[j + i*src0Col] + input2[j + i*src1Col]
            for j in range(src0Col, dstCol):
                golden[j + i*dstCol] = input2[j + i*src1Col]
    elif dstRow == src0Row and dstRow == src1Row and dstCol == src0Col and dstCol > src1Col:
        for i in range(0, src1Row):
            for j in range(0, src1Col):
                golden[j + i*dstCol] = input1[j + i*src0Col] + input2[j + i*src1Col]
            for j in range(src1Col, dstCol):
                golden[j + i*dstCol] = input1[j + i*src0Col]
    # Apply valid region constraints
    output = np.zeros([dstRow*dstCol]).astype(dtype)

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class tpartaddParams:
    def __init__(self, dtype, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC):
        self.dtype = dtype
        self.dstVR = dstVR
        self.dstVC = dstVC
        self.src0VR = src0VR
        self.src0VC = src0VC
        self.src1VR = src1VR
        self.src1VC = src1VC


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int16: 'int16',
        np.int32: 'int32',
    }[param.dtype]
    return f"TPARTADDTest.case_{dtype_str}_{param.dstVR}x{param.dstVC}_{param.src0VR}x{param.src0VC}_{param.src1VR}x{param.src1VC}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tpartaddParams(np.float32, 64, 64, 64, 64, 64, 64),
        tpartaddParams(np.float32, 64, 64, 8, 64, 64, 64),
        tpartaddParams(np.float32, 64, 64, 64, 8, 64, 64),
        tpartaddParams(np.float32, 64, 64, 64, 64, 8, 64),
        tpartaddParams(np.float32, 64, 64, 64, 64, 64, 8),
        tpartaddParams(np.float16, 8, 48, 8, 16, 8, 48),
        tpartaddParams(np.float16, 8, 768, 8, 512, 8, 768),
        tpartaddParams(np.int16, 8, 48, 8, 48, 8, 16),
        tpartaddParams(np.int32, 64, 64, 8, 64, 64, 64),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpartadd(case_name, param)
        os.chdir(original_dir)