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
    input_arr0 = np.random.uniform(low=-8, high=8, size=(param.s0rows, param.s0cols)).astype(data_type)
    input_arr1 = np.random.uniform(low=-8, high=8, size=(param.s1rows, param.s1cols)).astype(data_type)
    output_arr = np.full((param.drows, param.dcols), -np.inf)
    output_arr[:param.s0rows, :param.s0cols] = input_arr0[:param.s0rows, :param.s0cols]
    output_arr[:param.s1rows, :param.s1cols] = input_arr1[:param.s1rows, :param.s1cols]
    src0 = input_arr0[0:min(param.s0rows, param.s1rows), 0:min(param.s0cols, param.s1cols)]
    src1 = input_arr1[0:min(param.s0rows, param.s1rows), 0:min(param.s0cols, param.s1cols)]
    res = src0 * (src0 > src1) + src1 * (src0 <= src1)
    rows, cols = res.shape
    output_arr[:rows, :cols] = res
    output_arr = output_arr.astype(data_type)
    input_arr0.tofile('input0.bin')
    input_arr1.tofile('input1.bin')
    output_arr.tofile('golden.bin')


class TestParams:
    def __init__(self, name, data_type, dparam, s0param, s1param):
        self.name = name
        self.data_type = data_type
        self.drows, self.dcols, self.d_row_stride = dparam
        self.s0rows, self.s0cols, self.s0_row_stride = s0param
        self.s1rows, self.s1cols, self.s1_row_stride = s1param

if __name__ == "__main__":
    case_list = [
        TestParams('TPARTMAXTest.test0', np.float32, (16, 32, 32), (16, 16, 16), (16, 32, 32)),
        TestParams('TPARTMAXTest.test1', np.float32, (22, 32, 32), (22, 32, 32), (16, 32, 32)),
        TestParams('TPARTMAXTest.test2', np.float32, (22, 40, 40), (22, 40, 40), (22, 32, 32)),
        TestParams('TPARTMAXTest.test3', np.float32, (22, 40, 40), (22, 40, 40), (8, 40, 40)),
        TestParams('TPARTMAXTest.test4', np.float32, (64, 128, 128), (64, 128, 128), (64, 128, 128)),
        TestParams('TPARTMAXTest.testEmpty0', np.float32, (16, 32, 32), (16, 0, 8), (16, 32, 32)),
        TestParams('TPARTMAXTest.testEmpty1', np.float32, (16, 32, 32), (0, 32, 32), (16, 32, 32)),
        TestParams('TPARTMAXTest.testEmpty2', np.float32, (16, 32, 32), (16, 32, 32), (16, 0, 8)),
        TestParams('TPARTMAXTest.testEmpty3', np.float32, (16, 32, 32), (16, 32, 32), (0, 32, 32)),
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        orig_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(orig_dir)
