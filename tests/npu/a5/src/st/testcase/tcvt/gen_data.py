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
import en_dtypes

print("Warning: PyTorch not available, using NumPy for saturation tests")
HAS_TORCH = False

bfloat16 = np.float16  # Using float16 to simulate bfloat16 for data generation
fp8_e5m2 = ml_dtypes.float8_e5m2
fp8_e4m3 = ml_dtypes.float8_e4m3fn
hifloat8 = en_dtypes.hifloat8
np.random.seed(19)

# Flag to control PyTorch behavior for infinity handling
# GPU behavior (USE_PYTORCH_GPU_BEHAVIOR = True):
#   - Signed integers (int8, int16, int32): +inf → -1, -inf → 0
#   - Unsigned integers (uint8): +inf → max_value (255), -inf → 0
# CPU behavior (USE_PYTORCH_GPU_BEHAVIOR = False):
#   - All integer types: +inf → 0, -inf → 0
USE_PYTORCH_GPU_BEHAVIOR = True  # Set to False to use CPU behavior


def gen_golden(case_name, param):
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n
    valid_m, valid_n = param.valid_m, param.valid_n

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
        # Convert to float64 first to avoid overflow during clipping
        converted_float = converted_golden.astype(np.float64)
        clipped = np.clip(converted_float, info.min, info.max)
        golden = clipped.astype(dsttype)
    elif np.issubdtype(dsttype, np.floating):
        info = np.finfo(dsttype)
        golden = np.clip(converted_golden, info.min, info.max).astype(dsttype)
    else:
        golden = converted_golden.astype(dsttype)

    # Apply valid region constraints (zero out data outside valid region)
    if valid_m < m or valid_n < n:
        output = np.zeros([m, n]).astype(dsttype)
        output[:valid_m, :valid_n] = golden[:valid_m, :valid_n]
        golden = output

    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")


def gen_saturation_golden(case_name, param):
    """Generate test data for saturation mode testing with special values (inf, nan, overflow)"""
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n

    # Generate input with special values: inf, -inf, nan, and overflow values
    # Pattern matches A2/A3: special values first, then padding with zeros
    if srctype == np.float32 or srctype == np.float16:
        if dsttype == np.int8:
            # Special values: -inf, inf, nan, and 2 overflow values
            special_values = [
                -np.inf,  # -infinity
                np.inf,  # +infinity
                np.nan,  # NaN
                -200.0,  # Overflow below min (-128)
                200.0,  # Overflow above max (127)
            ]
            # Pad with zeros to reach m*n elements
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        elif dsttype == np.uint8:
            special_values = [
                -np.inf,  # -infinity
                np.inf,  # +infinity
                np.nan,  # NaN
                -100.0,  # Overflow below min (0)
                300.0,  # Overflow above max (255)
            ]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        elif dsttype == np.int16:
            special_values = [
                -np.inf,  # -infinity
                np.inf,  # +infinity
                np.nan,  # NaN
                -40000.0,  # Overflow below min (-32768)
                40000.0,  # Overflow above max (32767)
            ]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        elif dsttype == np.int32:
            special_values = [
                -np.inf,  # -infinity
                np.inf,  # +infinity
                np.nan,  # NaN
                -3e9,  # Overflow below min
                3e9,  # Overflow above max
            ]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        else:
            x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.int64:
        # int64 to int32 saturation test - only overflow values (no inf/nan for integers)
        if dsttype == np.int32:
            special_values = [
                -3000000000,  # Overflow below min
                3000000000,  # Overflow above max
                -2147483648,  # At min boundary
                2147483647,  # At max boundary
                0,  # Zero
            ]
            x1_gm = np.array(special_values + [0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        else:
            x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    elif srctype == np.int32:
        # int32 to int16 saturation test - only overflow values
        if dsttype == np.int16:
            special_values = [
                -40000,  # Overflow below min
                40000,  # Overflow above max
                -32768,  # At min boundary
                32767,  # At max boundary
                32769,  # Zero
            ]
            x1_gm = np.array(special_values + [0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        else:
            x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    else:
        # For other types, use normal range
        if srctype == np.float32 or srctype == np.float16:
            x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
        else:
            x1_gm = np.random.randint(-100, 100, [m, n]).astype(srctype)

    # Generate golden data using PyTorch's truncation mode (TRUNC)
    # Convert to PyTorch tensor
    if HAS_TORCH:
        if srctype == np.float16:
            x_torch = torch.from_numpy(x1_gm.astype(np.float32)).half()
        else:
            x_torch = torch.from_numpy(x1_gm)

        # Map numpy dtypes to torch dtypes for conversion
        dtype_map = {
            np.int8: torch.int8,
            np.uint8: torch.uint8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
        }

        torch_dtype = dtype_map.get(dsttype, torch.float32)

        # PyTorch uses truncation mode by default for float->int conversions
        golden_torch = x_torch.to(torch_dtype)
        golden_truncated = golden_torch.cpu().numpy().astype(dsttype)

        # Handle GPU vs CPU behavior for infinity
        # For signed integers: GPU: +inf → -1, -inf → 0 | CPU: +inf → 0, -inf → 0
        # For unsigned integers: GPU: +inf → max, -inf → 0 | CPU: +inf → 0, -inf → 0
        if USE_PYTORCH_GPU_BEHAVIOR and np.issubdtype(srctype, np.floating):
            if np.issubdtype(dsttype, np.signedinteger):
                # Apply GPU behavior: +inf becomes -1 for signed integers
                is_pos_inf = np.isinf(x1_gm) & (x1_gm > 0)
                golden_truncated[is_pos_inf] = -1
            elif np.issubdtype(dsttype, np.unsignedinteger):
                # Apply GPU behavior: +inf becomes max value for unsigned integers
                is_pos_inf = np.isinf(x1_gm) & (x1_gm > 0)
                info = np.iinfo(dsttype)
                golden_truncated[is_pos_inf] = info.max

        behavior = "GPU" if USE_PYTORCH_GPU_BEHAVIOR else "CPU"
        print(
            f"Generated truncated golden data using PyTorch ({behavior} behavior) for {srctype.__name__} → {dsttype.__name__}"
        )
    else:
        # Fallback to NumPy when PyTorch not available
        # Simulate truncation mode: trunc then handle special values
        if np.issubdtype(srctype, np.floating) and np.issubdtype(dsttype, np.integer):
            # Handle special values (inf, nan) with truncation behavior
            truncated_list = []
            info = np.iinfo(dsttype)

            for val in x1_gm.flat:
                if np.isnan(val) or np.isinf(val):
                    # Handle infinity based on GPU/CPU behavior flag
                    if USE_PYTORCH_GPU_BEHAVIOR and np.isinf(val) and val > 0:
                        if np.issubdtype(dsttype, np.signedinteger):
                            # GPU behavior: +inf → -1 for signed integers
                            int_val = -1
                        elif np.issubdtype(dsttype, np.unsignedinteger):
                            # GPU behavior: +inf → max value for unsigned integers
                            int_val = info.max
                        else:
                            int_val = 0
                    else:
                        # CPU behavior: all special values → 0
                        int_val = 0
                else:
                    # Truncate normal values
                    int_val = int(np.trunc(val))

                # Use wrapping behavior (like PyTorch) instead of clamping
                # This matches torch's behavior: e.g., 300 -> uint8 = 44 (not 255)
                truncated_list.append(int_val)

            # Use astype() for wrapping behavior with overflow values
            golden_truncated = np.array(truncated_list, dtype=np.int64).astype(dsttype).reshape(x1_gm.shape)
        else:
            # For non-floating to integer conversions
            converted = x1_gm
            # Use wrapping behavior (like PyTorch) instead of clamping
            # This matches torch's behavior for overflow values
            golden_truncated = converted.astype(dsttype)

        behavior = "GPU" if USE_PYTORCH_GPU_BEHAVIOR else "CPU"
        print(
            f"Generated truncated golden data using NumPy fallback ({behavior} behavior) for {srctype.__name__} → {dsttype.__name__}"
        )

    x1_gm.tofile("./x1_gm.bin")
    golden_truncated.tofile("./golden_truncated.bin")


class tcvtParams:
    def __init__(self, srctype, dsttype, m, n, mode, valid_m=None, valid_n=None):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode
        self.valid_m = valid_m if valid_m is not None else m
        self.valid_n = valid_n if valid_n is not None else n


if __name__ == "__main__":
    # Type conversion pairs: (name_suffix, source_type, destination_type)
    # Order matches TCvt.hpp organization by source type
    type_pairs = [
        # FP32 Source → fp16, bf16, int16, int32, int64, fp8 variants
        ("fp32_fp16", np.float32, np.float16),
        ("fp32_bf16", np.float32, bfloat16),
        ("fp32_int16", np.float32, np.int16),
        ("fp32_int32", np.float32, np.int32),
        ("fp32_int64", np.float32, np.int64),
        ("fp32_fp8_e4m3", np.float32, fp8_e4m3),
        ("fp32_fp8_e5m2", np.float32, fp8_e5m2),
        ("fp32_h8", np.float32, hifloat8),
        ("fp32_fp32", np.float32, np.float32),  # Same-type rounding
        # FP16 Source → fp32, int32, int16, int8, uint8, h8
        ("fp16_fp32", np.float16, np.float32),
        ("fp16_int32", np.float16, np.int32),
        ("fp16_int16", np.float16, np.int16),
        ("fp16_int8", np.float16, np.int8),
        ("fp16_uint8", np.float16, np.uint8),
        ("fp16_h8", np.float16, hifloat8),
        # BF16 Source → fp32, int32, half
        ("bf16_fp32", bfloat16, np.float32),
        ("bf16_int32", bfloat16, np.int32),
        ("bf16_fp16", bfloat16, np.float16),
        # U8 Source → half, uint16
        ("uint8_fp16", np.uint8, np.float16),
        # ("uint8_uint16", np.uint8, np.uint16),
        # I8 Source → half, int16, int32
        ("int8_fp16", np.int8, np.float16),
        ("int8_int16", np.int8, np.int16),
        ("int8_int32", np.int8, np.int32),
        # I16 Source → uint8, half, float, uint32, int32
        ("int16_uint8", np.int16, np.uint8),
        ("int16_fp16", np.int16, np.float16),
        ("int16_fp32", np.int16, np.float32),
        ("int16_uint32", np.int16, np.uint32),
        ("int16_int32", np.int16, np.int32),
        # I32 Source → float, int16, uint16, int64, uint8
        ("int32_fp32", np.int32, np.float32),
        ("int32_int16", np.int32, np.int16),
        # ("int32_uint16", np.int32, np.uint16),
        ("int32_int64", np.int32, np.int64),
        ("int32_uint8", np.int32, np.uint8),
        # U32 Source → uint8, uint16, int16
        ("uint32_uint8", np.uint32, np.uint8),
        # ("uint32_uint16", np.uint32, np.uint16),
        ("uint32_int16", np.uint32, np.int16),
        # I64 Source → float, int32
        ("int64_fp32", np.int64, np.float32),
        ("int64_int32", np.int64, np.int32),
        # FP8 Source → float
        ("fp8_e4m3_fp32", fp8_e4m3, np.float32),
        ("fp8_e5m2_fp32", fp8_e5m2, np.float32),
        ("h8_fp32", hifloat8, np.float32),
    ]

    # Different shape configurations (m, n)
    # Note: Tiles must be 32-byte aligned, so Cols * sizeof(T) must be >= 32 bytes
    # - For 32-bit types (float, int32): need Cols >= 8
    # - For 16-bit types (half, int16): need Cols >= 16
    # - For 8-bit types (int8, fp8): need Cols >= 32
    # Using shapes that work for all types (Cols >= 32)
    shapes = [
        # Single-row shapes (triggers 1D: Rows == 1)
        (1, 128),  # Single row - tests 1D path with Rows == 1
        # Multi-row contiguous shapes (triggers 1D: ValidCol == Cols)
        (2, 64),  # Small multi-row contiguous
        (4, 32),  # Multiple rows, minimal columns
        (2, 128),  # Larger multi-row contiguous
    ]

    # Partial tile configurations (m, n, valid_m, valid_n)
    # These shapes trigger 2D path: ValidCol != Cols (non-contiguous)
    # Keep ValidRows == Rows to focus on column non-contiguity
    partial_shapes = [
        (4, 128, 4, 65),  # 4 rows, half columns - basic 2D path test
        (4, 256, 4, 200),  # 4 rows, partial columns - larger 2D test
        (1, 256, 1, 129),  # Single row, partial columns - tests 2D path for single row case
    ]

    case_name_list = []
    case_params_list = []

    # Generate test cases for each type pair and shape combination
    for type_name, src, dst in type_pairs:
        # Regular full tile shapes
        for m, n in shapes:
            case_name = f"case_{type_name}_{m}x{n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))

        # Partial tile shapes
        for m, n, valid_m, valid_n in partial_shapes:
            case_name = f"case_{type_name}_{m}x{n}_{valid_m}x{valid_n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT", valid_m, valid_n))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)

    # ============================================================================
    # Saturation Mode Test Cases
    # ============================================================================
    # Generate test data for saturation mode tests (matching the test cases in main.cpp)
    # These tests use 1x32 shape and focus on conversions where saturation matters

    saturation_test_cases = [
        ("TCVTTest.saturation_fp16_int8_1x32", tcvtParams(np.float16, np.int8, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_fp32_int16_1x32", tcvtParams(np.float32, np.int16, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_fp16_int16_1x32", tcvtParams(np.float16, np.int16, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_fp16_uint8_1x32", tcvtParams(np.float16, np.uint8, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_int64_int32_1x32", tcvtParams(np.int64, np.int32, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_int32_int16_1x32", tcvtParams(np.int32, np.int16, 1, 32, "RoundMode::CAST_RINT")),
    ]

    for case_name, param in saturation_test_cases:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_saturation_golden(case_name, param)

        os.chdir(original_dir)
