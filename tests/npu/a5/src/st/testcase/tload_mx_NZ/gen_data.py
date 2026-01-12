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
np.set_printoptions(threshold=np.inf)


class DataFormat(Enum):
    ND2NZ = 1
    DN2NZ = 2
    ND2ND = 3
    NZ2NZ = 4
    DN2DN = 5
    DN2ZN = 6
    ZZ2ZZ = 7
    NN2NN = 8

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype
    #shape0,shape1,shape2,row,col表示gm真实搬运的数据
    shape0 = param.shape0
    shape1 = param.shape1
    shape2 = param.shape2
    #whole_shape表示gm的shape
    whole_shape0 = param.ws0
    whole_shape1 = param.ws1
    whole_shape2 = param.ws2
    whole_shape3 = param.ws3
    whole_shape4 = param.ws4
    c0_size = 2
    # [l1_row, l1_col]表示L1上申请Tile的大小
    row, col, l1_row, l1_col, is_atrans = param.m, param.k, param.basem, param.basek, False
    x1_gm = np.random.randint(1, 5, [row, col]).astype(src_type)
    golden = np.zeros([l1_row, l1_col]).astype(src_type)

    if param.load_type == DataFormat['ZZ2ZZ'].value or param.load_type == DataFormat['NN2NN'].value:
        x1_gm = np.random.randint(
            10, 125, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(np.uint8)

        submatrix = x1_gm[
            0:shape0,          # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,          # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,          # d2: 截取前shape2个元素（对应目标 d2=4）
            0:row,         # d3: 截取前M个元素（对应目标 d3=16）
            0:col         # d4: 截取K个元素（对应目标 d4=8）
        ]

        valid_row = 0
        valid_col = 0
        if param.load_type == DataFormat['ZZ2ZZ'].value:
            valid_row = shape0 * shape1 * row # shape0 >1 时，沿row方向拼接
            valid_col = shape2 * col
            assert (valid_col % c0_size) == 0, "valid_col in gm should be 2 aligned when matrix is ZZ format"
            assert (valid_row % 16) == 0, "valid_col in gm  should be 16 aligned when matrix is ZZ format"
        else:
            valid_row = shape2 * col
            valid_col = shape0 * shape1 * row # shape0 >1 时，沿col方向拼接
            assert (valid_col % 16) == 0, "valid_col in gm should be 16 aligned when matrix is NN format"
            assert (valid_row % c0_size) == 0, "valid_row in gm should be 2 aligned when matrix is NN format"

        golden = np.zeros([valid_row, valid_col]).reshape(shape0 * shape1, shape2, row, col).astype(src_type)
        new_submatrix = submatrix.reshape(
            submatrix.shape[0] * submatrix.shape[1], submatrix.shape[2], submatrix.shape[3], submatrix.shape[4])
        golden[:new_submatrix.shape[0], :new_submatrix.shape[1],
            :new_submatrix.shape[2], :new_submatrix.shape[3]] = new_submatrix

    x2_gm = np.random.randint(1, 5, [row, col]).astype(src_type)
    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class TmatmulParams:
    def __init__(self, atype, btype, ctype, shape0, shape1, shape2, m, k,
        ws0, ws1, ws2, ws3, ws4, basem, basek, load_type):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2

        self.ws0 = ws0
        self.ws1 = ws1
        self.ws2 = ws2
        self.ws3 = ws3
        self.ws4 = ws4

        self.basem = basem  # L1 row
        self.basek = basek  # L1 col
        self.load_type = load_type

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        # shape[0] == 1, L1Size = [valid_row, valid_col]
        "TLOADSCALETest.1_1_2_16_2_1_2_3_16_2_16_4_scale_ZZ2ZZ",
        "TLOADSCALETest.1_2_1_16_2_1_3_2_16_2_2_32_scale_NN2NN",
        #shape[0] == 1, L1Size > [valid_row, valid_col]
        "TLOADSCALETest.1_2_2_16_2_1_2_3_16_2_48_10_scale_ZZ2ZZ",
        "TLOADSCALETest.1_2_2_16_2_1_3_2_16_2_8_64_scale_NN2NN",
        "TLOADSCALETest.1_5_33_16_2_1_11_40_16_2_128_96_scale_ZZ2ZZ",
        "TLOADSCALETest.1_64_29_16_2_1_65_59_16_2_58_1088_scale_NN2NN",
        # shape[0] > 1, L1Size = [valid_row, valid_col]
        "TLOADSCALETest.3_1_2_16_2_3_2_3_16_2_48_4_scale_ZZ2ZZ",
        "TLOADSCALETest.4_2_1_16_2_4_3_2_16_2_2_128_scale_NN2NN",
        # shape[0] > 1, L1Size > [valid_row, valid_col]
        "TLOADSCALETest.4_3_3_16_2_4_10_5_16_2_192_10_scale_ZZ2ZZ",
        "TLOADSCALETest.7_5_3_16_2_7_7_11_16_2_12_560_scale_NN2NN",
    ]

    case_params_list = [
        # shape[0] == 1, L1Size = [valid_row, valid_col]
        TmatmulParams(np.uint8, np.uint8, np.uint8, 1, 1, 2, 16, 2,
                      1, 2, 3, 16, 2, 16, 4, DataFormat['ZZ2ZZ'].value),
        TmatmulParams(np.uint8, np.uint8,  np.uint8, 1, 2, 1, 16, 2,
                      1, 3, 2, 16, 2, 2, 32, DataFormat['NN2NN'].value),
        #shape[0] == 1, L1Size > [valid_row, valid_col]
        TmatmulParams(np.uint8, np.uint8, np.uint8, 1, 2, 2, 16, 2,
                      1, 2, 3, 16, 2, 48, 10, DataFormat['ZZ2ZZ'].value),
        TmatmulParams(np.uint8, np.uint8,  np.uint8, 1, 2, 2, 16, 2,
                      1, 3, 2, 16, 2, 8, 64, DataFormat['NN2NN'].value),
        TmatmulParams(np.uint8, np.uint8, np.uint8, 1, 5, 33, 16, 2,
                      1, 11, 40, 16, 2, 128, 96, DataFormat['ZZ2ZZ'].value),
        TmatmulParams(np.uint8, np.uint8,  np.uint8, 1, 64, 29, 16, 2,
                      1, 65, 59, 16, 2, 58, 1088, DataFormat['NN2NN'].value),
        # shape[0] > 1, L1Size = [valid_row, valid_col]
        TmatmulParams(np.uint8, np.uint8, np.uint8, 3, 1, 2, 16, 2,
                      3, 2, 3, 16, 2, 48, 4, DataFormat['ZZ2ZZ'].value),
        TmatmulParams(np.uint8, np.uint8,  np.uint8, 4, 2, 1, 16, 2,
                      4, 3, 2, 16, 2, 2, 128, DataFormat['NN2NN'].value),
        # shape[0] > 1, L1Size > [valid_row, valid_col]
        TmatmulParams(np.uint8, np.uint8, np.uint8, 4, 3, 3, 16, 2,
                      4, 10, 5, 16, 2, 192, 10, DataFormat['ZZ2ZZ'].value),
        TmatmulParams(np.uint8, np.uint8,  np.uint8, 7, 5, 3, 16, 2,
                      7, 7, 11, 16, 2, 12, 560, DataFormat['NN2NN'].value),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
