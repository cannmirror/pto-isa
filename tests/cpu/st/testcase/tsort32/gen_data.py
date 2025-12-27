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
    """
    生成黄金数据的修正版本
    """
    data_type = param.data_type
    row, col = param.row, param.col
    stride = 0
    if data_type == np.float16 or data_type == np.int16:
        stride = 4
    else:
        stride = 2
    
    # 生成随机数据
    total_elements = row * col
    input_arr = np.random.rand(row, col) * 10
    input_arr = input_arr.astype(data_type)
    index_arr = np.random.rand(row, col) * 10
    index_arr = index_arr.astype(np.uint32)
    output_arr = np.zeros((row, col * stride), dtype=data_type)

    for i in range(row):
        for j in range(0, col, 32):
            start = j
            end = min(j + 32, col)
            group_score = input_arr[i, start:end]
            group_index = index_arr[i, start:end]
            combined = list(zip(group_score, group_index))
            combined.sort(key=lambda x: (-x[0], x[1]))
            sorted_score = [x[0] for x in combined]
            sorted_index = [x[1] for x in combined]
            m = 0
            for k in range(start, end):
                if data_type in [np.float16, np.int16]:
                    output_arr[i, k * stride] = sorted_score[m]
                    output_arr[i, k * stride + 2] = sorted_index[m] & 0xFF
                    output_arr[i, k * stride + 3] = (sorted_index[m] >> 16) & 0xFF
                else:
                    output_arr[i, k * stride] = sorted_score[m]
                    output_arr[i, k * stride + 1] = sorted_index[m]
                m = m + 1
    # 保存输入文件
    input_arr.tofile('input0.bin')
    index_arr.tofile('input1.bin')
    output_arr.tofile('golden.bin')


class TestParams:
    def __init__(self, name, data_type, row, col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TestParams('TSORT32Test.test0', np.int16, 16, 16),
        TestParams('TSORT32Test.test1', np.float32, 8, 32),
        TestParams('TSORT32Test.test2', np.int32, 7, 32),
        TestParams('TSORT32Test.test3', np.float16, 32, 16),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
