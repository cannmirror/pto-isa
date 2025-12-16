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


def gen_golden_data(param):
    test_type = param.test_type
    rows = param.rows
    cols = param.cols

    input_arr = np.random.uniform(low = -10, high = 10, size = (rows, cols)).astype(test_type)
    input_arr.tofile("input_arr.bin")
    output_arr = input_arr
    nz_block_row = 16
    c0_size = 16
    output_arr = input_arr.reshape(int(rows / nz_block_row), nz_block_row,
        int(cols / c0_size), c0_size).transpose(2, 0, 1, 3).astype(test_type)
    output_arr.tofile("golden_output.bin")


class TmovUb2L1Params:
    def __init__(self, test_type, rows, cols):
        self.test_type = test_type
        self.rows = rows
        self.cols = cols

if __name__ == "__main__":
    case_name_list = [
        "TMovUb2l1Test.case1",
    ]

    case_params_list = [
        TmovUb2L1Params(np.float16, 16, 32),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)