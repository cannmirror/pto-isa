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


def gen_golden_data(param):
    data_type = param.data_type
    src0_validrow, src0_validcol = param.src0_validrow, param.src0_validcol
    src1_validrow, src1_validcol = param.src1_validrow, param.src1_validcol
    input0_arr = np.random.rand(param.src0_row, param.src0_col) * 10
    input0_arr = input0_arr.astype(data_type)
    input1_arr = np.random.rand(param.src1_row, param.src1_col) * 10
    input1_arr = input1_arr.astype(data_type)
    output_arr = np.zeros((param.dst_row, param.dst_col), dtype=data_type)
    
    for i in range(param.dst_validrow):
        for j in range(param.dst_validcol):
            iin_src0 = i < src0_validrow and j < src0_validcol
            in_src1 = i < src1_validrow and j < src1_validcol
            out_src0 = i >= src0_validrow or j >= src0_validcol
            out_src1 = i >= src1_validrow or j >= src1_validcol
            if iin_src0 and in_src1:
                output_arr[i, j] = max(input0_arr[i, j], input1_arr[i, j])
            elif out_src0 and in_src1:
                output_arr[i, j] = input1_arr[i, j]
            elif iin_src0 and out_src1:
                output_arr[i, j] = input0_arr[i, j]

    input0_arr.tofile('input0.bin')
    input1_arr.tofile('input1.bin')
    output_arr.tofile('golden.bin')


class TestParams:
    def __init__(self, name, data_type, dst_row, dst_col, src0_row, src0_col, src1_row, src1_col, dst_validrow,
                 dst_validcol, src0_validrow, src0_validcol, src1_validrow, src1_validcol):
        self.name = name
        self.data_type = data_type
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col
        self.dst_validrow = dst_validrow
        self.dst_validcol = dst_validcol
        self.src0_validrow = src0_validrow
        self.src0_validcol = src0_validcol
        self.src1_validrow = src1_validrow
        self.src1_validcol = src1_validcol


if __name__ == "__main__":
    case_params_list = [
        TestParams('TPARTMAXTest.test0', np.int16, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32),
        TestParams('TPARTMAXTest.test1', np.float16, 22, 32, 22, 32, 16, 32, 22, 32, 22, 32, 16, 32),
        TestParams('TPARTMAXTest.test2', np.float32, 22, 40, 22, 40, 22, 32, 22, 40, 22, 40, 22, 32),
        TestParams('TPARTMAXTest.test3', np.int32, 22, 40, 22, 40, 8, 40, 22, 40, 22, 40, 8, 40),
        TestParams('TPARTMAXTest.test4', np.float32, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128),
        TestParams('TPARTMAXTest.testEmpty0', np.int16, 16, 32, 16, 16, 16, 32, 16, 32, 16, 0, 16, 32),
        TestParams('TPARTMAXTest.testEmpty1', np.float16, 16, 32, 16, 32, 16, 32, 16, 32, 0, 32, 16, 32),
        TestParams('TPARTMAXTest.testEmpty2', np.float32, 16, 32, 16, 32, 16, 16, 16, 32, 16, 32, 16, 0),
        TestParams('TPARTMAXTest.testEmpty3', np.int32, 16, 32, 16, 32, 16, 32, 16, 32, 16, 32, 0, 32),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)