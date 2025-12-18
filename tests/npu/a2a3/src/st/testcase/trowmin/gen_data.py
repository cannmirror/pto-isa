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
np.random.seed(23)


def gen_golden_data(param):
    input_arr = np.random.uniform(low=-16, high=16, size=(param.row, param.col)).astype(param.data_type)
    output_arr = np.full((param.valid_row), np.finfo(param.data_type).max).astype(param.data_type)
    for i in range(param.valid_row):
        output_arr[i] = np.min(input_arr[i][:param.valid_col])
    input_arr.tofile('input.bin')
    output_arr.tofile('golden.bin')


class TRowMinParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col


if __name__ == "__main__":
    case_params_list = [
        TRowMinParams("TROWMINTest.case1", np.float32, 127, 127, 64, 64 - 1),
        TRowMinParams("TROWMINTest.case2", np.float32, 63, 63, 64, 64),
        TRowMinParams("TROWMINTest.case3", np.float32, 31, 31, 64 * 2, 64 * 2 - 1),
        TRowMinParams("TROWMINTest.case4", np.float32, 15, 15, 64 * 3, 64 * 3),
        TRowMinParams("TROWMINTest.case5", np.float32, 7, 7, 64 * 7, 64 * 7 - 1),
        TRowMinParams("TROWMINTest.case6", np.float16, 256, 256, 16, 16 - 1),
        TRowMinParams("TROWMINTest.case7", np.float32, 30, 30, 216, 216),
        TRowMinParams("TROWMINTest.case8", np.float32, 30, 30, 216, 24),
        TRowMinParams("TROWMINTest.case9", np.float32, 30, 11, 216, 216),
        TRowMinParams("TROWMINTest.case10", np.float32, 30, 11, 216, 24),
        TRowMinParams("TROWMINTest.case11", np.float32, 238, 238, 40, 40),
        TRowMinParams("TROWMINTest.case12", np.float32, 238, 238, 40, 16),
        TRowMinParams("TROWMINTest.case13", np.float32, 238, 121, 40, 40),
        TRowMinParams("TROWMINTest.case14", np.float32, 238, 121, 40, 16),
        TRowMinParams("TROWMINTest.case15", np.float32, 64, 64, 128, 128),
        TRowMinParams("TROWMINTest.case16", np.float32, 32, 32, 256, 256),
        TRowMinParams("TROWMINTest.case17", np.float32, 16, 16, 512, 512),
        TRowMinParams("TROWMINTest.case18", np.float32, 8, 8, 1024, 1024)
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)