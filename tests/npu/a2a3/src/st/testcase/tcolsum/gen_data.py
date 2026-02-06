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


def gen_golden_data(case_name, param):
    data_type = param.data_type
    cols = param.col
    src_row = param.src_row
    src_valid_row = param.src_valid_row

    input = np.random.rand(src_valid_row, cols) * 10
    input = input.astype(data_type)
    golden = np.zeros((1, cols), dtype=data_type)
    golden[0] = np.sum(input, axis=0)
    input.tofile('input.bin')
    golden.tofile('golden.bin')


class TColSum:
    def __init__(self, data_type, col, src_row, src_valid_row):
        self.data_type = data_type
        self.col = col
        self.src_row = src_row
        self.src_valid_row = src_valid_row

if __name__ == "__main__":

    case_name_list = [
        "TCOLSUMTest.case1",
        "TCOLSUMTest.case2",
        "TCOLSUMTest.case3",
        "TCOLSUMTest.case4",
        "TCOLSUMTest.case5",
        "TCOLSUMTest.case6",
        "TCOLSUMTest.case7",
        "TCOLSUMTest.case8",
        "TCOLSUMTest.case9",
        "TCOLSUMTest.case10",
        "TCOLSUMTest.case11",
        "TCOLSUMTest.case12",
        "TCOLSUMTest.case13",
        "TCOLSUMTest.case14",
        "TCOLSUMTest.case15",
        "TCOLSUMTest.case16",
        "TCOLSUMTest.case17",
        "TCOLSUMTest.case18",
        "TCOLSUMTest.case19",
        "TCOLSUMTest.case20",
        "TCOLSUMTest.case21",
        "TCOLSUMTest.case22",
        "TCOLSUMTest.case23",
        "TCOLSUMTest.case24",
        "TCOLSUMTest.case25",
        "TCOLSUMTest.case26",
    ]

    case_params_list = [
        TColSum(np.int16, 16, 16, 8),
        TColSum(np.int32, 16, 16, 8),
        TColSum(np.float32, 16, 16, 8),
        TColSum(np.int16, 128, 16, 8),
        TColSum(np.int32, 64, 16, 8),
        TColSum(np.float32, 64, 16, 8),
        TColSum(np.int16, 512, 16, 8),
        TColSum(np.int32, 256, 16, 8),
        TColSum(np.float32, 256, 16, 8),
        TColSum(np.int16, 512, 16, 7),
        TColSum(np.int32, 256, 32, 31),
        TColSum(np.float32, 256, 32, 31),
        TColSum(np.float32, 256, 16, 1),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        if (i > len(case_name_list) // 2 - 1):
            gen_golden_data(case_name, case_params_list[i - len(case_name_list) // 2])
        else:
            gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)