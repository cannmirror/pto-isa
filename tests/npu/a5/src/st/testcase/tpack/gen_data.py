#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------


import os
import struct
import math
import numpy as np
from ml_dtypes import float8_e4m3fn, bfloat16

np.random.seed(42)


def gen_golden_data_tpack(case_name, param):
    """Generate golden test data for TPack (bit-packing from wider to narrower type).

    TPack extracts the lower N bytes of each M-byte source element (little-endian)
    and packs them contiguously into the destination buffer.

    Strategy: generate random values in the narrow (dst) type, zero-extend each
    element to the wide (src) byte width.  The golden output is the original
    narrow-typed array.
    """
    src_dtype = param.src_dtype  # wider type  (e.g. np.uint32)
    dst_dtype = param.dst_dtype  # narrower type (e.g. np.uint16)
    valid_rows = param.valid_rows
    valid_cols = param.valid_cols

    src_itemsize = np.dtype(src_dtype).itemsize
    dst_itemsize = np.dtype(dst_dtype).itemsize

    # Generate random bytes and interpret as the narrow (dst) type
    total_elements = valid_rows * valid_cols
    raw_bytes = np.random.randint(0, 256, size=(total_elements, dst_itemsize), dtype=np.uint8)
    golden = raw_bytes.reshape(-1).view(dst_dtype).reshape(valid_rows, valid_cols)

    # Build the wide (src) array by zero-padding upper bytes of each element
    pad_width = src_itemsize - dst_itemsize
    padding = np.zeros((total_elements, pad_width), dtype=np.uint8)
    src_raw = np.concatenate([raw_bytes, padding], axis=1)  # little-endian order
    src = src_raw.reshape(-1).view(src_dtype).reshape(valid_rows, valid_cols)

    src.tofile("input.bin")
    golden.tofile("golden.bin")


class TPackParams:
    # Mapping from string name to numpy dtype
    SRC_DTYPE_MAP = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": bfloat16,
        "s32": np.int32,
        "s16": np.int16,
        "u32": np.uint32,
        "u16": np.uint16,
    }
    DST_DTYPE_MAP = {
        "fp16": np.float16,
        "bf16": bfloat16,
        "fp8": float8_e4m3fn,
        "s16": np.int16,
        "s8": np.int8,
        "u16": np.uint16,
        "u8": np.uint8,
    }

    def __init__(self, src_dtype_str, dst_dtype_str, valid_rows, valid_cols):
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.src_dtype_str = src_dtype_str
        self.dst_dtype_str = dst_dtype_str
        self.src_dtype = self.SRC_DTYPE_MAP[src_dtype_str]
        self.dst_dtype = self.DST_DTYPE_MAP[dst_dtype_str]


def generate_case_name(param):
    return f"TPACKTEST.case_{param.src_dtype_str}_{param.dst_dtype_str}_{param.valid_rows}x{param.valid_cols}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    ## default fp8 is e4m3 format
    case_params_list = [
        TPackParams("fp32", "fp16", 128, 128),
        TPackParams("fp32", "bf16", 128, 128),
        TPackParams("s32", "s16", 128, 128),
        TPackParams("u32", "u16", 128, 128),
        TPackParams("fp32", "fp8", 128, 128),
        TPackParams("s32", "s8", 128, 128),
        TPackParams("u32", "u8", 128, 128),
        TPackParams("fp16", "fp8", 128, 128),
        TPackParams("s16", "s8", 128, 128),
        TPackParams("u16", "u8", 128, 128),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpack(case_name, param)
        os.chdir(original_dir)
