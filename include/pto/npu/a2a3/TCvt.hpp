/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU A2/A3 Architecture
 *
 * FILE ORGANIZATION (for easy navigation):
 * =======================================
 *
 * SUPPORTED CONVERSIONS (quick lookup):
 * ====================================
 * FP32:  -> FP16, FP32 (rounding only), BF16, I16, I32, I64
 * FP16:  -> I32, I16, I8, U8
 * BF16:  -> I32
 * I16:   -> FP16, FP32
 * I32:   -> FP32, I16, I64, FP16 (deq)
 * I64:   -> FP32, I32
 * U8:    -> FP16
 * I8:    -> FP16
 *
 * 1. GenCastCall* helpers (lines ~20-360)
 *    - fp32 -> fp16/fp32/int64/int32/int16/bf16
 *    - fp16 -> int32/int16/int8/uint8
 *    - bf16 -> int32
 *    - int16/int32/int64 -> fp16/fp32
 *
 * 2. GenCastCallSpecialCases (lines ~360-450)
 *    - half<->fp32, bf16->fp32, int8/uint8->half
 *    - int64<->int32, int32->int16, int32->half (deq)
 *
 * 3. GenCastCall Dispatcher (lines ~450-530)
 *    - Compile-time type routing to the correct GenCastCall* helper
 *
 * 4. TCvtHead (lines ~540-600)
 *    - Processes aligned repeat blocks for main data region
 *
 * 5. TCvt Kernel (lines ~610-700)
 *    - Handles aligned region and remainder with vector masking
 *
 * 6. TCVT_IMPL (lines ~710-end)
 *    - High-level entry point computing repeat configuration
 *
 * QUICK FIND: Search for the conversion function name (e.g., "GenCastCallFp32ToFp16")
 * or the dispatcher "GenCastCall" to locate the relevant section.
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include "common.hpp"

namespace pto {
// ============================================================================
// Type Conversion Functions
// ============================================================================
// These functions handle specialized data type conversions with different
// rounding modes. Each function supports multiple rounding modes and dispatches
// to the appropriate vector conversion instruction (vconv_*) based on the
// selected rounding method.
//
// Rounding modes: RINT (round-to-nearest-int), ROUND (arithmetic round),
//                FLOOR, CEIL, TRUNC (truncate), ODD (odd rounding), NONE
// ============================================================================
inline namespace TCvtInternel {
// CTRL[59] controls saturation mode for floating point to integer conversions:
// - CTRL[59] = 0 (Saturation ON): Values are saturated to the datatype's range.
//   Out-of-range values clamp to the maximum/minimum representable value.
//   Example: converting 300.0f to int8 yields 127 (max for int8)
//
// - CTRL[59] = 1 (Saturation OFF): Values are truncated using bit masking.
//   Floating point is first converted to signed 64-bit integer, then the least
//   significant N bits are extracted (N=8 for int8, N=16 for int16, N=32 for int32).
//   Example: converting 300.0f to int8 yields 44 (300 & 0xFF = 0x2C = 44)
constexpr const int SAT_MODE_BIT = 59;

// Buffer size for temporary storage in non-saturation conversion path
// Sized to handle REPEAT_MAX repeats with maximum data width
constexpr const size_t FP16_INT8_TEMP_BUFFER_SIZE = REPEAT_MAX * 256;
} // namespace TCvtInternel

// EDGE_CASE_ALIGN_ENABLE controls PyTorch alignment for edge case values
// - When enabled (1): TCVT output matches PyTorch when handling edge values
//   like inf, -inf, nan, and overflow values. Uses NonSatTorch implementations.
// - When disabled (0): Uses standard TCVT conversion (higher performance)
// Trade-off: Enabling provides PyTorch compatibility but reduces performance
#define EDGE_CASE_ALIGN_ENABLE 0

// Converts float32 (fp32) to float16 (fp16) with various rounding modes
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToFp16(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    // fp32 to fp16 - Dispatch based on rounding mode
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322f16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322f16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322f16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322f16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ODD:
            vconv_f322f16o(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float32 to Float32 with rounding mode - Used for normalization/conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToFp32(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    // fp32 to fp32 - Apply rounding mode to float32 data
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float32 to signed 64-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt64(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // fp32 to int64 - Convert floating point to 64-bit integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s64r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s64a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s64f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s64c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s64z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322s64z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float32 to signed 32-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // fp32 to int32 - Convert floating point to 32-bit integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float32 to signed 16-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt16(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // fp32 to int16 - Convert floating point to 16-bit integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float32 to signed 16-bit integer conversion for non-saturation mode
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp32 -> int32
// 2. int32 -> int16
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToInt16_NonSatTorch(__ubuf__ typename TileDataD::DType *dst,
                                                     __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum,
                                                     RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                     uint16_t dstRepeatStride, uint16_t srcRepeatStride,
                                                     __ubuf__ int32_t *tempInt32Buf)
{
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322s32r(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s32a(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s32f(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s32c(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s32z(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        default:
            vconv_f322s32z(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
    }

    pipe_barrier(PIPE_V);

    // Step 2: Convert int32 to int16
    vconv_s322s16(dst, tempInt32Buf, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

// Float32 to bfloat16 conversion
// Bfloat16 preserves the exponent range of float32 in a 16-bit format
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp32ToBf16(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    // fp32 to bf16 - Convert floating point to bfloat16 format
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f322bf16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322bf16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322bf16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322bf16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322bf16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f322bf16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float16 (half) to signed 32-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // fp16 to int32 - Convert half-precision float to 32-bit integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float16 (half) to signed 16-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt16(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // fp16 to int16 - Convert half-precision float to 16-bit integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162s16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float16 (half) to signed 16-bit integer conversion for non-saturation mode
// This version matches PyTorch behavior for inf/-inf and performs a two-step conversion:
// 1. fp16 -> int32
// 2. int32 -> int16
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt16_NonSatTorch(__ubuf__ typename TileDataD::DType *dst,
                                                     __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum,
                                                     RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                     uint16_t dstRepeatStride, uint16_t srcRepeatStride,
                                                     __ubuf__ int32_t *tempInt32Buf)
{
    // Step 1: Convert fp16 to int32
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s32r(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s32a(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s32f(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s32c(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s32z(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        default:
            vconv_f162s32z(tempInt32Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
    }

    pipe_barrier(PIPE_V);

    vconv_s322s16(dst, tempInt32Buf, 2 * repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

// Float16 (half) to signed 8-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt8(__ubuf__ typename TileDataD::DType *dst,
                                        __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride)
{
    // fp16 to int8 - Convert half-precision float to 8-bit signed integer
    // Note: Saturation mode is now controlled globally by TCvt kernel
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s8r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s8a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s8f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s8c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162s8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Float16 (half) to signed 8-bit integer conversion for non-saturation mode
// This version matches PyTorch behavior for inf/-inf and performs a multi-step conversion:
// 1. fp16 -> int16 (direct conversion)
// 2. bitwise AND with 255 using int16
// 3. int16 -> fp16
// 4. fp16 -> int8
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt8_NonSatTorch(__ubuf__ typename TileDataD::DType *dst,
                                                    __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum,
                                                    RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                                                    uint16_t dstRepeatStride, uint16_t srcRepeatStride,
                                                    __ubuf__ int16_t *tempInt16Buf, __ubuf__ int16_t *tempAndBuf,
                                                    __ubuf__ half *tempFp16Buf)
{
    // Step 1: Convert fp16 to int16 (both are 2 bytes, use srcBlockStride and srcRepeatStride)
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162s16r(tempInt16Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s16a(tempInt16Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s16f(tempInt16Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s16c(tempInt16Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s16z(tempInt16Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
        default:
            vconv_f162s16z(tempInt16Buf, src, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
                           srcRepeatStride);
            break;
    }
    pipe_barrier(PIPE_V);

    // Step 2: Bitwise AND with 255 (0xFF) using int16 (both buffers are int16, use srcBlockStride and srcRepeatStride)
    vector_dup(tempAndBuf, static_cast<int16_t>(255), repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride,
               srcRepeatStride);
    pipe_barrier(PIPE_V);
    vand(tempAndBuf, tempInt16Buf, tempAndBuf, repeatNum, srcBlockStride, srcBlockStride, srcBlockStride,
         srcRepeatStride, srcRepeatStride, srcRepeatStride);

    // Step 3: Convert int16 to fp16 (both 2 bytes, use srcBlockStride and srcRepeatStride)
    vconv_s162f16(tempFp16Buf, tempAndBuf, repeatNum, srcBlockStride, srcBlockStride, srcRepeatStride, srcRepeatStride);

    pipe_barrier(PIPE_V);

    // Step 4: Convert fp16 to int8 (fp16=2 bytes uses srcBlockStride, int8=1 byte uses dstBlockStride)
    vconv_f162s8z(dst, tempFp16Buf, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

// Float16 (half) to unsigned 8-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToUint8(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // fp16 to uint8 - Convert half-precision float to 8-bit unsigned integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_f162u8r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162u8a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162u8f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162u8c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162u8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_f162u8z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Bfloat16 to signed 32-bit integer conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallBf16ToInt32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // bf16 to int32 - Convert bfloat16 to 32-bit signed integer
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_bf162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_bf162s32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_bf162s32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_bf162s32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_bf162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_bf162s32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Signed 16-bit integer to float16 (half) conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallInt16ToFp16(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // int16 to fp16 - Convert signed integer to half-precision float
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_s162f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s162f16a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s162f16f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s162f16c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s162f16z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_s162f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Signed 32-bit integer to float32 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallInt32ToFp32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // int32 to fp32 - Convert 32-bit signed integer to single-precision float
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_s322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s322f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s322f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s322f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s322f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_s322f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Signed 64-bit integer to float32 conversion
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallInt64ToFp32(__ubuf__ typename TileDataD::DType *dst,
                                         __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                         uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                         uint16_t srcRepeatStride)
{
    // int64 to fp32 - Convert 64-bit signed integer to single-precision float
    switch (static_cast<RoundMode>(mode)) {
        case RoundMode::CAST_RINT:
            vconv_s642f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s642f32a(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s642f32f(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s642f32c(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s642f32z(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
        default:
            vconv_s642f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            break;
    }
}

// Special case conversions using compile-time type checking
// Handles conversions like: half<->fp32, bf16<->fp32, int/uint 8<->half,
//                           int32<->int64, int32<->int16, int32->half (with deq)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallSpecialCases(__ubuf__ typename TileDataD::DType *dst,
                                          __ubuf__ typename TileDataS::DType *src, uint8_t repeatNum, RoundMode mode,
                                          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                                          uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                  std::is_same<typename TileDataS::DType, half>::value) { // half to fp32
        vconv_f162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, bfloat16_t>::value) { // bfloat16 to float
        vconv_bf162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, uint8_t>::value) { // uint8 to half
        vconv_u82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int8_t>::value) { // int8 to half
        vconv_s82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, int16_t>::value) { // int16 to float32
        vconv_s162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, int64_t>::value) { // int64 to int32
        vconv_s642s32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int64_t>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to int64
        vconv_s322s64(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to int16
        vconv_s322s16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to half
        set_deqscale(static_cast<half>(1.0));
        pipe_barrier(PIPE_V);
        vconv_deq(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

// ============================================================================
// Type Conversion Dispatcher
// ============================================================================
// GenCastCall dispatches to specific conversion functions based on source/dest
// data types using compile-time type checking (constexpr if).
template <typename TileDataD, typename TileDataS>
AICORE void GenCastCall(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
                        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride,
                        uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    // Dispatch to appropriate function based on compile-time type analysis
    if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                  std::is_same<typename TileDataS::DType, float>::value) {
        GenCastCallFp32ToFp16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to fp32
        GenCastCallFp32ToFp32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int64_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int64
        GenCastCallFp32ToInt64<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int32
        GenCastCallFp32ToInt32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to int16
        // Select implementation based on current saturation mode (CTRL[59]) and edge case alignment
        bool isSatOn = (get_ctrl() & (1ULL << SAT_MODE_BIT)) == 0;
#if EDGE_CASE_ALIGN_ENABLE
        if (!isSatOn) {
            // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
            __ubuf__ int32_t *tempInt32Buf = (__ubuf__ int32_t *)(TMP_UB_OFFSET);
            GenCastCallFp32ToInt16_NonSatTorch<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride,
                                                                     srcBlockStride, dstRepeatStride, srcRepeatStride,
                                                                     tempInt32Buf);
        } else {
            GenCastCallFp32ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                         dstRepeatStride, srcRepeatStride);
        }
#else
        // Use default implementation when edge case alignment is disabled
        GenCastCallFp32ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
#endif
    } else if constexpr (std::is_same<typename TileDataD::DType, bfloat16_t>::value &&
                         std::is_same<typename TileDataS::DType, float>::value) { // fp32 to bf16
        GenCastCallFp32ToBf16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int32
        GenCastCallFp16ToInt32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int16
        // Select implementation based on current saturation mode (CTRL[59]) and edge case alignment
        bool isSatOn = (get_ctrl() & (1ULL << SAT_MODE_BIT)) == 0;
#if EDGE_CASE_ALIGN_ENABLE
        if (!isSatOn) {
            // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
            __ubuf__ int32_t *tempInt32Buf = (__ubuf__ int32_t *)(TMP_UB_OFFSET);
            GenCastCallFp16ToInt16_NonSatTorch<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride,
                                                                     srcBlockStride, dstRepeatStride, srcRepeatStride,
                                                                     tempInt32Buf);
        } else {
            GenCastCallFp16ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                         dstRepeatStride, srcRepeatStride);
        }
#else
        // Use default implementation when edge case alignment is disabled
        GenCastCallFp16ToInt16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
#endif
    } else if constexpr (std::is_same<typename TileDataD::DType, int8_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to int8
        // Select implementation based on current saturation mode (CTRL[59]) and edge case alignment
        bool isSatOn = (get_ctrl() & (1ULL << SAT_MODE_BIT)) == 0;
#if EDGE_CASE_ALIGN_ENABLE
        if (!isSatOn) {
            // Use PyTorch-aligned implementation when saturation is OFF and edge case alignment is enabled
            __ubuf__ int16_t *tempInt16Buf = (__ubuf__ int16_t *)(TMP_UB_OFFSET);
            __ubuf__ int16_t *tempAndBuf = (__ubuf__ int16_t *)(TMP_UB_OFFSET + 2048);
            __ubuf__ half *tempFp16Buf = (__ubuf__ half *)(TMP_UB_OFFSET + 4096);
            GenCastCallFp16ToInt8_NonSatTorch<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride,
                                                                    srcBlockStride, dstRepeatStride, srcRepeatStride,
                                                                    tempInt16Buf, tempAndBuf, tempFp16Buf);
        } else {
            GenCastCallFp16ToInt8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                        dstRepeatStride, srcRepeatStride);
        }
#else
        // Use default implementation when edge case alignment is disabled
        GenCastCallFp16ToInt8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                    dstRepeatStride, srcRepeatStride);
#endif
    } else if constexpr (std::is_same<typename TileDataD::DType, uint8_t>::value &&
                         std::is_same<typename TileDataS::DType, half>::value) { // half to uint8
        GenCastCallFp16ToUint8<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                         std::is_same<typename TileDataS::DType, bfloat16_t>::value) { // bfloat16 to int32
        GenCastCallBf16ToInt32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                         std::is_same<typename TileDataS::DType, int16_t>::value) { // int16 to half
        GenCastCallInt16ToFp16<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, int32_t>::value) { // int32 to float
        GenCastCallInt32ToFp32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                         std::is_same<typename TileDataS::DType, int64_t>::value) { // int64 to float
        GenCastCallInt64ToFp32<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                     dstRepeatStride, srcRepeatStride);
    } else {
        GenCastCallSpecialCases<TileDataD, TileDataS>(dst, src, repeatNum, mode, dstBlockStride, srcBlockStride,
                                                      dstRepeatStride, srcRepeatStride);
    }
}

// ============================================================================
// Tile Conversion Helper: Process Main Data Block
// ============================================================================
// TCvtHead processes the primary aligned portion of data in complete repeat units.
// This handles data that fits evenly into repeat boundaries.
//
// @param dstPtr: Destination buffer pointer
// @param srcPtr: Source buffer pointer
// @param mode: Rounding mode for type conversions
// @param numRepeatPerLine: Number of complete repeats per line
// @param validRow: Number of valid rows to process
// @param elementsPerRepeat: Number of elements per repeat unit
// @param dstRepeatStride: Stride between repeats in destination
// @param srcRepeatStride: Stride between repeats in source
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
PTO_INST void TCvtHead(__ubuf__ typename TileDataD::DType *dstPtr, __ubuf__ typename TileDataS::DType *srcPtr,
                       RoundMode mode, unsigned numRepeatPerLine, unsigned validRow, unsigned elementsPerRepeat,
                       unsigned dstRepeatStride, unsigned srcRepeatStride)
{
    unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
    unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
    for (uint32_t i = 0; i < validRow; i++) {
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + j * elementsPerRepeat * REPEAT_MAX,
                                                  srcPtr + i * SS + j * elementsPerRepeat * REPEAT_MAX,
                                                  (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)dstRepeatStride,
                                                  (uint16_t)srcRepeatStride);
            }
        }
        if (remainAfterLoop > 0) {
            GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                              srcPtr + i * SS + numLoop * elementsPerRepeat * REPEAT_MAX,
                                              (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)dstRepeatStride,
                                              (uint16_t)srcRepeatStride);
        }
    }
}

// ============================================================================
// Core Tile Conversion Kernel
// ============================================================================
// TCvt orchestrates the complete tile conversion process by handling both:
//   1. Aligned region: Complete repeat units processed via TCvtHead
//   2. Remainder region: Partial repeats processed with vector masking
//
// Template parameters:
//   SS: Source row stride
//   DS: Destination row stride
//
// @param dst: Destination tile (output) - contains data after conversion
// @param src: Source tile (input) - contains original data to be converted
// @param mode: Rounding mode (RINT/ROUND/FLOOR/CEIL/TRUNC/NONE/ODD)
// @param satMode: Saturation mode for float-to-int conversions:
//                 ON  = Clamp to datatype range [min, max]
//                 OFF = Convert to int64, extract least significant N bits
// @param numRepeatPerLine: Number of complete repeats per line
// @param numRemainPerLine: Remaining elements per line (not aligned to repeat)
// @param validRow: Number of rows containing valid data
// @param elementsPerRepeat: Number of elements per repeat operation
// @param dstRepeatStride: Stride between repeats in destination buffer
// @param srcRepeatStride: Stride between repeats in source buffer
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
__tf__ AICORE void TCvt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src,
                        RoundMode mode, SaturationMode satMode, unsigned numRepeatPerLine, unsigned numRemainPerLine,
                        unsigned validRow, unsigned elementsPerRepeat, unsigned dstRepeatStride,
                        unsigned srcRepeatStride)
{
    // Save the original saturation mode state
    uint64_t originalCtrl = get_ctrl();
    bool originalSatMode = (originalCtrl & (1ULL << SAT_MODE_BIT)) == 0; // true if originally ON

    // Apply requested saturation mode
    if (satMode == SaturationMode::OFF) {
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT)); // Turn off saturation
    } else {
        set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT)); // Turn on saturation (default)
    }

    // Get tile buffer pointers and calculate elements per memory block
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *srcPtr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);
    constexpr unsigned dstNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
    constexpr unsigned srcNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType);

    // Process main aligned region with complete repeat units
    if (numRepeatPerLine > 0) {
        TCvtHead<TileDataD, TileDataS, SS, DS>(dstPtr, srcPtr, mode, numRepeatPerLine, validRow, elementsPerRepeat,
                                               dstRepeatStride, srcRepeatStride);
    }
    // Advance pointers to unaligned remainder region
    dstPtr += numRepeatPerLine * elementsPerRepeat;
    srcPtr += numRepeatPerLine * elementsPerRepeat;

    // Process remainder region with partial repeats (requires vector masking)
    if (numRemainPerLine > 0) {
        unsigned numLoop = validRow / REPEAT_MAX;
        unsigned remainAfterLoop = validRow % REPEAT_MAX;
        SetContinuousMask(numRemainPerLine);
        if (numLoop > 0) {
            for (uint32_t j = 0; j < numLoop; j++) {
                GenCastCall<TileDataD, TileDataS>(dstPtr + j * DS * REPEAT_MAX, srcPtr + j * SS * REPEAT_MAX,
                                                  (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)DS / dstNElemPerBlock,
                                                  (uint16_t)SS / srcNElemPerBlock);
            }
        }
        if (remainAfterLoop > 0) {
            GenCastCall<TileDataD, TileDataS>(dstPtr + numLoop * DS * REPEAT_MAX, srcPtr + numLoop * SS * REPEAT_MAX,
                                              (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)DS / dstNElemPerBlock,
                                              (uint16_t)SS / srcNElemPerBlock);
        }
        set_vector_mask(-1, -1);
    }

    // Restore original saturation mode to avoid affecting subsequent instructions
    if (originalSatMode) {
        set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT)); // Restore to ON
    } else {
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT)); // Restore to OFF
    }
}

// ============================================================================
// High-Level Tile Conversion Interface
// ============================================================================
// TCVT_IMPL is the main entry point for tile data type conversion.
// Calculates optimal repeat configuration and delegates to TCvt kernel.
//
// This is the main implementation with explicit satMode parameter.
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode)
{
    // Determine repeat width as max of source/destination element sizes
    uint64_t repeatWidth =
        static_cast<uint64_t>(max(sizeof(typename TileDataD::DType), sizeof(typename TileDataS::DType)));

    // Calculate destination repeat stride
    unsigned dstRepeatStride =
        repeatWidth == sizeof(typename TileDataD::DType) ?
            BLOCK_MAX_PER_REPEAT :
            (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataS::DType) * sizeof(typename TileDataD::DType));

    // Calculate source repeat stride
    unsigned srcRepeatStride =
        repeatWidth == sizeof(typename TileDataS::DType) ?
            BLOCK_MAX_PER_REPEAT :
            (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataD::DType) * sizeof(typename TileDataS::DType));

    // Calculate elements per repeat and split data into aligned/remainder regions
    unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
    unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat; // Complete repeats
    unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat; // Remainder elements

    // Get row strides (compile-time constants)
    constexpr unsigned SS = TileDataS::RowStride;
    constexpr unsigned DS = TileDataD::RowStride;
    unsigned validRow = dst.GetValidRow();

    // Invoke TCvt kernel with calculated parameters
    TCvt<TileDataD, TileDataS, SS, DS>(dst.data(), src.data(), mode, satMode, numRepeatPerLine, numRemainPerLine,
                                       validRow, elementsPerRepeat, dstRepeatStride, srcRepeatStride);
}

// ============================================================================
// TCVT_IMPL Overload with Type-Specific Defaults
// ============================================================================
// This overload provides conversion-specific default saturation modes:
// - FP16→UINT8, FP16→INT8: defaults to OFF (PyTorch-compatible truncation)
// - INT64→INT32, INT32→INT16: defaults to OFF (truncation behavior)
// - All others: defaults to ON (native TCVT saturation)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
{
    // Conversions that default to OFF for PyTorch compatibility or truncation behavior
    if constexpr (
        // FP16→UINT8
        (std::is_same<typename TileDataD::DType, uint8_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        // FP16→INT8
        (std::is_same<typename TileDataD::DType, int8_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        // FP32→INT16
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, float>::value) ||
        // FP16→INT16
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, half>::value) ||
        // INT64→INT32
        (std::is_same<typename TileDataD::DType, int32_t>::value &&
         std::is_same<typename TileDataS::DType, int64_t>::value) ||
        // INT32→INT16
        (std::is_same<typename TileDataD::DType, int16_t>::value &&
         std::is_same<typename TileDataS::DType, int32_t>::value)) {
        TCVT_IMPL(dst, src, mode, SaturationMode::OFF);
    } else {
        // All other conversions: default to ON (native TCVT saturation)
        TCVT_IMPL(dst, src, mode, SaturationMode::ON);
    }
}
} // namespace pto
#endif
