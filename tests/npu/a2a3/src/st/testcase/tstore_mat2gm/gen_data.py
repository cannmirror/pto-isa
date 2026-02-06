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
    g_shape0 = param.g_shape0
    g_shape1 = param.g_shape1
    g_shape2 = param.g_shape2
    g_shape3 = param.g_shape3
    g_shape4 = param.g_shape4
    g_whole_shape0 = param.g_whole_shape0
    g_whole_shape1 = param.g_whole_shape1
    g_whole_shape2 = param.g_whole_shape2
    g_whole_shape3 = param.g_whole_shape3
    g_whole_shape4 = param.g_whole_shape4

    if param.format == "ND" or param.format == "NZ":
        input_arr = np.random.randint(-5, 5, size=(g_whole_shape0, g_whole_shape1,
                                    g_whole_shape2, g_whole_shape3, g_whole_shape4)).astype(data_type)
        output_arr = np.zeros(shape=(g_whole_shape0, g_whole_shape1,
                            g_whole_shape2, g_whole_shape3, g_whole_shape4), dtype=data_type)
        output_arr[0:g_shape0, 0: g_shape1, 0: g_shape2, 0: g_shape3, 0: g_shape4] \
                    = input_arr[0:g_shape0, 0: g_shape1, 0: g_shape2, 0: g_shape3, 0: g_shape4]
    elif param.format == "DN":
        input_arr = np.random.randint(-5, 5, size=(g_whole_shape0, g_whole_shape1,
                            g_whole_shape2, g_whole_shape4, g_whole_shape3)).astype(data_type)
        output_arr = np.zeros(shape=(g_whole_shape0, g_whole_shape1,
                            g_whole_shape2, g_whole_shape4, g_whole_shape3), dtype=data_type)
        output_arr[0:g_shape0, 0: g_shape1, 0: g_shape2, 0: g_shape4, 0: g_shape3] \
                    = input_arr[0:g_shape0, 0: g_shape1, 0: g_shape2, 0: g_shape4, 0: g_shape3]

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class GlobalTensorInfo:
    def __init__(self, case_name, data_type, format, g_shape0, g_shape1, g_shape2, g_shape3, g_shape4,
                g_whole_shape0, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4):
        self.case_name = case_name
        self.data_type = data_type
        self.format = format
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4

if __name__ == "__main__":
    case_params_list = [
        GlobalTensorInfo("TStoreMat2GMTest.case_nd1", np.int64, "ND", 1, 2, 1, 11, 32, 1, 3, 2, 93, 32),
        GlobalTensorInfo("TStoreMat2GMTest.case_nd2", np.float32, "ND", 1, 1, 1, 3, 128, 3, 3, 3, 32, 128),
        GlobalTensorInfo("TStoreMat2GMTest.case_nd3", np.int16, "ND", 2, 2, 1, 2, 32, 3, 3, 3, 111, 64),
        GlobalTensorInfo("TStoreMat2GMTest.case_nd4", np.int8, "ND", 1, 2, 1, 11, 32, 1, 3, 2, 93, 32),
        GlobalTensorInfo("TStoreMat2GMTest.case_nd5", np.float16, "ND", 1, 1, 1, 128, 128, 1, 1, 1, 256, 256),

        GlobalTensorInfo("TStoreMat2GMTest.case_dn1", np.int64, "DN", 2, 2, 1, 32, 2, 3, 3, 3, 64, 111),
        GlobalTensorInfo("TStoreMat2GMTest.case_dn2", np.float32, "DN", 1, 1, 1, 128, 3, 3, 3, 3, 128, 32),
        GlobalTensorInfo("TStoreMat2GMTest.case_dn3", np.int16, "DN", 2, 2, 1, 32, 2, 3, 3, 3, 64, 111),
        GlobalTensorInfo("TStoreMat2GMTest.case_dn4", np.int8, "DN", 1, 2, 1, 32, 11, 1, 3, 2, 32, 93),
        GlobalTensorInfo("TStoreMat2GMTest.case_dn5", np.float16, "DN", 1, 2, 2, 128, 311, 4, 3, 3, 256, 400),

        GlobalTensorInfo("TStoreMat2GMTest.case_nz1", np.float32, "NZ", 1, 5, 21, 16, 8, 1, 5, 21, 16, 8),
        GlobalTensorInfo("TStoreMat2GMTest.case_nz2", np.int16, "NZ", 2, 15, 11, 16, 16, 3, 23, 13, 16, 16),
        GlobalTensorInfo("TStoreMat2GMTest.case_nz3", np.int8, "NZ", 1, 16, 32, 16, 32, 1, 32, 32, 16, 32),
        GlobalTensorInfo("TStoreMat2GMTest.case_nz4", np.float16, "NZ", 2, 4, 5, 16, 16, 7, 7, 7, 16, 16),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params)
        os.chdir(original_dir)