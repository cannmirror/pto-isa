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
    src_type = param.datatype
    dst_type = param.datatype
    rows = param.row
    cols = param.col

    input_arr = np.random.uniform(low=-8, high=8, size=(rows, cols)).astype(src_type)
    result_arr = input_arr.sum(axis=1, keepdims=True)
    output_arr = np.zeros((rows, cols), dtype=dst_type)
    for i in range(cols):
        output_arr[i, 0] = result_arr[i, 0]
    input_arr.tofile('input0.bin')
    output_arr.tofile('golden.bin')


class TrowsumParams:
    def __init__(self, name, datatype, row, col):
        self.name = name
        self.datatype = datatype
        self.row = row
        self.col = col

if __name__ == "__main__":
    case_list = [
        TrowsumParams("TROWSUMTest.test1", np.float32, 16, 16),
        TrowsumParams("TROWSUMTest.test2", np.float16, 16, 16),
        TrowsumParams("TROWSUMTest.test3", np.float32, 666, 666)
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
