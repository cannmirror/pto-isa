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

def gen_golden_data_tgatherb(case_name, param):
    dtype = param.dtype
    data_size = 1
    if dtype == np.float32 or dtype == np.int32 or dtype == np.uint32:
        data_size = 4
    elif dtype == np.float16 or dtype == np.int16 or dtype == np.uint16:
        data_size = 2
    elif dtype == np.int8 or dtype == np.uint8:
        data_size = 1
    else:
        ValueError(f"{dtype} unsupported data type!!")
    blockSizeElem = int(32/data_size)

    src_shape = [param.srcS1, param.srcS0]
    dst_shape = [param.dstS1, param.dstS0]
    offset_col = int(param.dstS0 / blockSizeElem)
    offset_shape = [param.dstS1, offset_col]
    offset_elt_num = param.dstS1 * offset_col
    dst_elt_num = param.dstS1 * param.dstS0

    src = np.arange(param.srcS1 * param.srcS0).astype(dtype)
    # src = np.random.uniform(low=0, high=10000, size=(param.srcS1*param.srcS0)).astype(dtype)
    offset = np.zeros(offset_elt_num)
    for i in range(offset_elt_num):
        offset[i] = i * 32
    offset = offset.astype(np.uint32)

    print("data_size:", data_size)
    golden = np.zeros(dst_elt_num).astype(dtype)
    output = np.zeros(dst_elt_num).astype(dtype)
    count = 0
    for i in range(offset_elt_num):
        for j in range(int(32/data_size)):
            golden[count] = src[int(offset[i] / data_size + j)]
            count += 1
    golden.reshape((dst_elt_num)).astype(np.uint32)

    src.tofile(str("x.bin"))
    offset.tofile(str("offset.bin"))
    golden.tofile(str("golden.bin"))

    srcAddr = 0x0
    return output, src, offset, srcAddr, golden

class tgatherbParams:
    def __init__(self, dtype, dstS1, dstS0, offsetS1, offsetS0, srcS1, srcS0):
        self.dtype = dtype
        self.dstS1 = dstS1
        self.dstS0 = dstS0
        self.offsetS1 = offsetS1
        self.offsetS0 = offsetS0
        self.srcS1 = srcS1
        self.srcS0 = srcS0

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.int32: 'int32',
        np.uint32: 'uint32',
        np.int16: 'int16',
        np.uint16: 'uint16',
        np.float16: 'half',
        np.int8: 'int8',
        np.uint8: 'uint8',
    }[param.dtype]
    return f"TGATHERBTest.case_{dtype_str}_{param.dstS1}x{param.dstS0}_{param.offsetS1}x{param.offsetS0}_{param.srcS1}x{param.srcS0}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)
    
    case_params_list = [
        tgatherbParams(np.float32, 2, 128, 2, 16, 2, 128),
        tgatherbParams(np.int32, 2, 128, 2, 16, 2, 128),
        tgatherbParams(np.uint32, 2, 128, 2, 16, 2, 128),
        tgatherbParams(np.int16, 1, 32768, 1, 2048, 1, 32768),
        tgatherbParams(np.uint16, 257, 128, 257, 8, 257, 128),
        tgatherbParams(np.float16, 1, 32768, 1, 2048, 1, 32768),
        tgatherbParams(np.int8, 2, 256, 2, 8, 2, 256),
        tgatherbParams(np.int8, 2, 32768, 2, 1024, 2, 32768),
        tgatherbParams(np.uint8, 2, 32768, 2, 1024, 2, 32768),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        print(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tgatherb(case_name, param)
        os.chdir(original_dir)