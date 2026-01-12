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
from enum import Enum
np.random.seed(19)
np.set_printoptions(threshold=np.inf)


class DataFormat(Enum):
    AND2ZZ = 1
    ADN2ZZ = 2
    AZZ2ZZ = 3
    BND2NN = 4
    BDN2NN = 5
    BNN2NN = 6


def update_golden(golden, param, tile_rows, tile_cols):
    src_type = param.atype
    c0_size = 2
    if param.load_type == DataFormat['AND2ZZ'].value:
        assert (tile_rows % 16) == 0, "tile_rows should be 16 aligned when matrix A is AND2ZZ format"
        assert (tile_cols % c0_size) == 0, "tile_cols should be c0_size(2) aligned when matrix A is AND2ZZ format"
        golden = golden.reshape(
            (int(tile_rows / 16), 16, int(tile_cols / c0_size), c0_size)).transpose(0, 2, 1, 3).astype(src_type)
    elif param.load_type == DataFormat['ADN2ZZ'].value:
        assert (tile_rows % 16) == 0, "tile_rows should be 16 aligned when matrix A is ADN2ZZ format"
        assert (tile_cols % c0_size) == 0, "tile_cols should be c0_size(2) aligned when matrix A is ADN2ZZ format"
        golden = golden.reshape(
            (int(tile_rows / 16), 16, int(tile_cols / c0_size), c0_size)).transpose(0, 2, 1, 3).astype(src_type)
    elif param.load_type == DataFormat['BND2NN'].value:
        assert (tile_rows % c0_size) == 0, "tile_rows should be c0_size(2) aligned when matrix B is BND2NN format"
        assert (tile_cols % 16) == 0, "tile_cols should be 16 aligned when matrix B is BND2NN format"
        golden = golden.reshape(
            (int(tile_rows // 2), 2, int(tile_cols // 16), 16)).transpose(2, 0, 3, 1).astype(src_type)
    elif param.load_type == DataFormat['BDN2NN'].value:
        assert (tile_rows % c0_size) == 0, "tile_rows should be c0_size(2) aligned when matrix B is BDN2NN format"
        assert (tile_cols % 16) == 0, "tile_cols should be 16 aligned when matrix B is BDN2NN format"
        golden = golden.reshape(
            (int(tile_rows // 2), 2, int(tile_cols // 16), 16)).transpose(2, 0, 3, 1).astype(src_type)
    return golden


def gen_golden_data(param):
    src_type = param.atype

    whole_shape3 = param.ws3
    whole_shape4 = param.ws4

    valid_row, valid_col, tile_rows, tile_cols = param.valid_row, param.valid_col, param.tile_rows, param.tile_cols
    x1_gm = np.random.randint(1, 5, [valid_row, valid_col]).astype(src_type)
    golden = np.zeros([tile_rows, tile_cols]).astype(src_type)

    if param.load_type == DataFormat['AND2ZZ'].value:
        x1_gm = np.random.randint(1, 5, [whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([tile_rows, tile_cols]).astype(src_type)  # L1中Tile大小
        min_m = min(valid_row, golden.shape[0])
        min_k = min(valid_col, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
    elif param.load_type == DataFormat['ADN2ZZ'].value:
        x1_gm = np.tile(np.arange(whole_shape3)[:, np.newaxis], (1, whole_shape4)).astype(src_type)
        golden = np.zeros([tile_rows, tile_cols]).astype(src_type)  # L1中Tile大小
        min_m = min(valid_row, golden.shape[0])
        min_k = min(valid_col, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
        x1_gm = x1_gm.reshape((whole_shape3, whole_shape4 // 2, 2)).transpose(1, 0, 2).astype(src_type)
    elif param.load_type == DataFormat['BND2NN'].value:
        x1_gm = np.random.randint(1, 5, [whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([tile_rows, tile_cols]).astype(src_type)  # L1中Tile大小
        min_m = min(valid_row, golden.shape[0])
        min_k = min(valid_col, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
        x1_gm = x1_gm.reshape((whole_shape3 // 2, 2, whole_shape4)).transpose(0, 2, 1).astype(src_type)
    elif param.load_type == DataFormat['BDN2NN'].value:
        x1_gm = np.tile(np.arange(whole_shape3)[:, np.newaxis], (1, whole_shape4)).astype(src_type)
        golden = np.zeros([tile_rows, tile_cols]).astype(src_type)  # L1中Tile大小
        min_m = min(valid_row, golden.shape[0])
        min_k = min(valid_col, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
        x1_gm = x1_gm.reshape((whole_shape3, whole_shape4)).transpose(1, 0).astype(src_type)

    x2_gm = np.random.randint(1, 5, [valid_row, valid_col]).astype(src_type)
    golden = update_golden(golden, param, tile_rows, tile_cols)
    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class TmatmulParams:
    def __init__(self, case_name, atype, btype, ctype, shape0, shape1, shape2,
                 valid_row, valid_col, ws0, ws1, ws2, ws3, ws4, tile_rows, tile_cols, load_type):
        self.case_name = case_name
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2

        self.ws0 = ws0
        self.ws1 = ws1
        self.ws2 = ws2
        self.ws3 = ws3
        self.ws4 = ws4

        self.tile_rows = tile_rows  # L1 row
        self.tile_cols = tile_cols  # L1 col
        self.load_type = load_type

if __name__ == "__main__":

    case_params_list = [
        # for scaleA, K % 2 == 0
        # AND2ZZ
        TmatmulParams("TLOADMXTest.1_1_1_16_4_uint8_AND2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1, 16,
                      4, 1, 1, 1, 16, 4, 16, 4, DataFormat['AND2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_16_64_uint8_AND2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      16, 64, 1, 1, 1, 16, 64, 32, 158, DataFormat['AND2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_32_128_uint8_AND2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      32, 128, 1, 1, 1, 160, 128, 64, 1008, DataFormat['AND2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_128_128_uint8_AND2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1, 128,
                      128, 1, 1, 1, 128, 128, 128, 128, DataFormat['AND2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_64_128_uint8_AND2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      31, 118, 1, 1, 1, 34, 126, 64, 128, DataFormat['AND2ZZ'].value),                      
        # ADN2ZZ
        TmatmulParams("TLOADMXTest.1_1_1_16_4_uint8_ADN2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      1, 2, 1, 1, 1, 1, 65534, 16, 8, DataFormat['ADN2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_16_64_uint8_ADN2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1, 16,
                      64, 1, 1, 1, 16, 64, 16, 64, DataFormat['ADN2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_32_128_uint8_ADN2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1, 32,
                      128, 1, 1, 1, 32, 128, 32, 128, DataFormat['ADN2ZZ'].value),                      
        TmatmulParams("TLOADMXTest.1_1_1_128_128_uint8_ADN2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1, 27,
                      126, 1, 1, 1, 128, 128, 128, 128, DataFormat['ADN2ZZ'].value),
        TmatmulParams("TLOADMXTest.1_1_1_64_128_uint8_ADN2ZZ", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      31, 118, 1, 1, 1, 34, 126, 64, 128, DataFormat['ADN2ZZ'].value),
        # for scaleB, valid_row % 2 == 0
        # BND2NN
        TmatmulParams("TLOADMXTest.1_1_1_4_64_uint8_BND2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1, 4,
                      64, 1, 1, 1, 4, 64, 4, 64, DataFormat['BND2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_16_64_uint8_BND2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1, 16,
                      64, 1, 1, 1, 16, 64, 16, 64, DataFormat['BND2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_32_128_uint8_BND2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      32, 127, 1, 1, 1, 32, 128, 32, 256, DataFormat['BND2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_128_128_uint8_BND2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1, 128,
                      128, 1, 1, 1, 128, 128, 128, 128, DataFormat['BND2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_128_64_uint8_BND2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      116, 34, 1, 1, 1, 130, 60, 128, 64, DataFormat['BND2NN'].value),
        # BDN2NN
        TmatmulParams("TLOADMXTest.1_1_1_4_64_uint8_BDN2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1, 4,
                      64, 1, 1, 1, 4, 64, 4, 64, DataFormat['BDN2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_16_64_uint8_BDN2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1, 16,
                      64, 1, 1, 1, 16, 64, 16, 64, DataFormat['BDN2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_32_128_uint8_BDN2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      2, 128, 1, 1, 1, 32, 128, 4, 1088, DataFormat['BDN2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_128_128_uint8_BDN2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1, 30,
                      127, 1, 1, 1, 128, 128, 128, 128, DataFormat['BDN2NN'].value),
        TmatmulParams("TLOADMXTest.1_1_1_128_64_uint8_BDN2NN", np.uint8, np.uint8, np.uint8, 1, 1, 1,
                      116, 34, 1, 1, 1, 130, 60, 128, 64, DataFormat['BDN2NN'].value),
    ]

    for _, case_param in enumerate(case_params_list):
        casename = case_param.case_name
        if not os.path.exists(casename):
            os.makedirs(casename)
        original_dir = os.getcwd()
        os.chdir(casename)

        gen_golden_data(case_param)

        os.chdir(original_dir)
