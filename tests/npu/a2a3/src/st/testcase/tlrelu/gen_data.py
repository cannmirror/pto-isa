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
import struct
import numpy as np

np.random.seed(42)


def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col

    input_arr = np.random.uniform(low=-8, high=8, size=(rows, cols)).astype(data_type)
    scalar = np.random.uniform(low=0.001, high=0.1, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    for i in range(rows):
        for j in range(cols):
            if input_arr[i, j] < 0:
                output_arr[i, j] = input_arr[i, j] * scalar[0, 0]
            else:
                output_arr[i, j] = input_arr[i, j]
    input_arr.tofile("input.bin")
    with open("scalar.bin", "wb") as f:
        f.write(struct.pack("f", np.float32(scalar[0, 0])))
    output_arr.tofile("golden.bin")


class TLRELUParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TLRELUParams("TLRELUTest.case1", np.float32, 32, 64, 32, 64),
        TLRELUParams("TLRELUTest.case2", np.float16, 63, 64, 63, 64),
        TLRELUParams("TLRELUTest.case3", np.float32, 31, 128, 31, 128),
        TLRELUParams("TLRELUTest.case4", np.float16, 15, 64 * 3, 15, 64 * 3),
        TLRELUParams("TLRELUTest.case5", np.float32, 7, 64 * 7, 7, 64 * 7),
        TLRELUParams("TLRELUTest.case6", np.float32, 256, 16, 256, 16),
        TLRELUParams("TLRELUTest.case7", np.float32, 32, 128, 32, 64),
        TLRELUParams("TLRELUTest.case8", np.float16, 63, 128, 63, 64),
        TLRELUParams("TLRELUTest.case9", np.float32, 31, 256, 31, 128),
        TLRELUParams("TLRELUTest.case10", np.float16, 15, 192, 15, 64 * 3),
        TLRELUParams("TLRELUTest.case11", np.float32, 7, 512, 7, 64 * 7),
        TLRELUParams("TLRELUTest.case12", np.float32, 256, 32, 256, 16),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
