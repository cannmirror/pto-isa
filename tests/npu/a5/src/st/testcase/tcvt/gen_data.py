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
import ml_dtypes

bfloat16 = np.float16  # Using float16 to simulate bfloat16 for data generation
fp8_e5m2 = ml_dtypes.float8_e5m2  # Using int8 to simulate fp8_e5m2 for data generation
fp8_e4m3 = ml_dtypes.float8_e4m3fn  # Using int8 to simulate fp8_e4m3 for data generation
hifloat8 = ml_dtypes.float8_e5m2  # Using int8 to simulate hifloat8 for data generation
np.random.seed(19)

def gen_golden(case_name, param):
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n

    # Generate input data with reasonable ranges
    if srctype == np.float32 or srctype == np.float16 or srctype == bfloat16:
        # Floating point: range [-100, 100]
        x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.int8 or srctype == fp8_e5m2 or srctype == fp8_e4m3 or srctype == hifloat8:
        # int8/fp8/hifloat8: full range [-128, 127]
        x1_gm = np.random.randint(-128, 128, [m, n]).astype(srctype)
    elif srctype == np.uint8:
        # uint8: full range [0, 255]
        x1_gm = np.random.randint(0, 256, [m, n]).astype(srctype)
    elif srctype == np.int16:
        # int16: reasonable range [-1000, 1000]
        x1_gm = np.random.randint(-1000, 1000, [m, n]).astype(srctype)
    elif srctype == np.uint16:
        # uint16: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int32:
        # int32: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    elif srctype == np.uint32:
        # uint32: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int64:
        # int64: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    else:
        # Default: signed int range
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)

    # Apply rounding mode for conversions
    mode = param.mode
    
    # Perform conversion first
    if np.issubdtype(srctype, np.floating):
        if np.issubdtype(dsttype, np.integer):
            # Floating point to integer conversion
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        elif srctype == np.float32 and dsttype == np.float32:
            # FP32 to FP32 conversion - apply rounding to integer values but keep as float
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        else:
            # Other float to float conversions - no rounding applied
            converted_golden = x1_gm
    else:
        # Integer to any type conversion
        converted_golden = x1_gm

    # Clamp the result to the destination type's representable range
    if np.issubdtype(dsttype, np.integer):
        info = np.iinfo(dsttype)
        golden = np.clip(converted_golden, info.min, info.max).astype(dsttype)
    elif np.issubdtype(dsttype, np.floating):
        info = np.finfo(dsttype)
        golden = np.clip(converted_golden, info.min, info.max).astype(dsttype)
    else:
        golden = converted_golden.astype(dsttype)
            
    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")
                
class tcvtParams:
    def __init__(self, srctype, dsttype, m, n, mode):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode

if __name__ == "__main__":
    # Type conversion pairs: (name_suffix, source_type, destination_type)
    type_pairs = [
        # FP32 Source
        ("fp32_fp32", np.float32, np.float32),
        ("fp32_fp16", np.float32, np.float16),
        ("fp32_bf16", np.float32, bfloat16),
        ("fp32_int32", np.float32, np.int32),
        ("fp32_int16", np.float32, np.int16),
        ("fp32_int64", np.float32, np.int64),
        ("fp32_fp8_e4m3", np.float32, fp8_e4m3),
        ("fp32_fp8_e5m2", np.float32, fp8_e5m2),
        ("fp32_h8", np.float32, hifloat8),
        
        # FP16 Source
        ("fp16_fp32", np.float16, np.float32),
        ("fp16_int32", np.float16, np.int32),
        ("fp16_int16", np.float16, np.int16),
        ("fp16_int8", np.float16, np.int8),
        ("fp16_uint8", np.float16, np.uint8),
        ("fp16_fp8_e5m2", np.float16, fp8_e5m2),
        ("fp16_fp8_e4m3", np.float16, fp8_e4m3),
        ("fp16_h8", np.float16, hifloat8),

        # BF16 Source
        ("bf16_fp32", bfloat16, np.float32),
        ("bf16_int32", bfloat16, np.int32),
        ("bf16_fp16", bfloat16, np.float16),
        ("bf16_fp8_e5m2", bfloat16, fp8_e5m2),
        ("bf16_fp8_e4m3", bfloat16, fp8_e4m3),

        # INT32 Source
        ("int32_fp32", np.int32, np.float32),
        ("int32_int16", np.int32, np.int16),
        # ("int32_uint16", np.int32, np.uint16),
        ("int32_int64", np.int32, np.int64),
        ("int32_uint8", np.int32, np.uint8),

        # UINT32 Source
        ("uint32_uint8", np.uint32, np.uint8),
        ("uint32_uint16", np.uint32, np.uint16),
        ("uint32_int16", np.uint32, np.int16),

        # INT16 Source
        ("int16_fp16", np.int16, np.float16),
        ("int16_fp32", np.int16, np.float32),
        ("int16_uint32", np.int16, np.uint32),
        ("int16_int32", np.int16, np.int32),
        ("int16_uint8", np.int16, np.uint8),

        # INT8 Source
        ("int8_fp16", np.int8, np.float16),
        ("int8_int16", np.int8, np.int16),
        ("int8_int32", np.int8, np.int32),

        # UINT8 Source
        ("uint8_fp16", np.uint8, np.float16),
        # ("uint8_uint16", np.uint8, np.uint16),

        # INT64 Source
        ("int64_fp32", np.int64, np.float32),
        ("int64_int32", np.int64, np.int32),

        # FP8 Source
        ("fp8_e4m3_fp32", fp8_e4m3, np.float32),
        ("fp8_e5m2_fp32", fp8_e5m2, np.float32),
        ("h8_fp32", hifloat8, np.float32),
    ]

    # Different shape configurations (m, n)
    shapes = [
        (2, 128),
        (2, 32),
        (1, 64),
        (4, 64),
    ]

    case_name_list = []
    case_params_list = []

    # Generate test cases for each type pair and shape combination
    for type_name, src, dst in type_pairs:
        for m, n in shapes:
            case_name = f"case_{type_name}_{m}x{n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)
