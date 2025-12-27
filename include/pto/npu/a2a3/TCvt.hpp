/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCVT_HPP
#define TCVT_HPP

#include <pto/common/constants.hpp>

namespace pto {
    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_1(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // fp32 to fp16
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
            case RoundMode::CAST_NONE:
                vconv_f322f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f322f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_2(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
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

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_3(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
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
                vconv_f322s64r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_4(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
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
                vconv_f322s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_5(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
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
                vconv_f322s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    } 

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_6(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {

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
        
    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_7(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
            // half to int32
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
                vconv_f162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_8(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // half to int16
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
                vconv_f162s16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }
    
    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_9(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // half to int8
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
            case RoundMode::CAST_NONE:
                vconv_f162s8(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f162s8r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_10(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // half to uint8
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
            case RoundMode::CAST_NONE:
                vconv_f162u8(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_f162u8r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_11(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // bfloat16 to int32
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
            case RoundMode::CAST_NONE:
                vconv_bf162s32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_bf162s32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }
    
    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_12(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // int16 to half
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
            case RoundMode::CAST_NONE:
                vconv_s162f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_s162f16r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }
        
    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_13(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // int16 to half
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
            case RoundMode::CAST_NONE:
                vconv_s322f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
            default:
                vconv_s322f32r(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
                break;
        }
    }
    
    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_14(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // int64 to float
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

    template <typename TileDataD, typename TileDataS>
    PTO_INTERNAL void GenCastCall_15(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                             std::is_same<typename TileDataS::DType, half>::value) {  // half to fp32
            vconv_f162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                           std::is_same<typename TileDataS::DType, bfloat16_t>::value) {  // bfloat16 to float
            vconv_bf162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                           std::is_same<typename TileDataS::DType, uint8_t>::value) {  // uint8 to half
            vconv_u82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                             std::is_same<typename TileDataS::DType, int8_t>::value) {  // int8 to half
            vconv_s82f16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                             std::is_same<typename TileDataS::DType, int16_t>::value) {  // int16 to float32
            vconv_s162f32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);        
        } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                             std::is_same<typename TileDataS::DType, int64_t>::value) {  // int64 to int32
            vconv_s642s32(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int64_t>::value &&
                             std::is_same<typename TileDataS::DType, int32_t>::value) {  // int32 to int64
            vconv_s322s64(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                             std::is_same<typename TileDataS::DType, int32_t>::value) {  // int32 to int16
            vconv_s322s16(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                             std::is_same<typename TileDataS::DType, int32_t>::value) {  // int32 to half
            set_deqscale(static_cast<half>(1.0));
            pipe_barrier(PIPE_V);
            vconv_deq(dst, src, repeatNum, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        }
    }

    template <typename TileDataD, typename TileDataS>
    AICORE void GenCastCall(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS::DType *src,
        uint8_t repeatNum, RoundMode mode, uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
        uint16_t srcRepeatStride) {
        // fp32 to fp16
        if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                      std::is_same<typename TileDataS::DType, float>::value) {
            GenCastCall_1<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                             std::is_same<typename TileDataS::DType, float>::value) {  // fp32 to fp32
            GenCastCall_2<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int64_t>::value &&
                             std::is_same<typename TileDataS::DType, float>::value) {  // fp32 to int64
            GenCastCall_3<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                             std::is_same<typename TileDataS::DType, float>::value) {  // fp32 to int32
            GenCastCall_4<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                             std::is_same<typename TileDataS::DType, float>::value) {  // fp32 to int16
            GenCastCall_5<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, bfloat16_t>::value &&
                             std::is_same<typename TileDataS::DType, float>::value) {  // fp32 to bf16
            GenCastCall_6<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                             std::is_same<typename TileDataS::DType, half>::value) {  // half to int32
            GenCastCall_7<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int16_t>::value &&
                             std::is_same<typename TileDataS::DType, half>::value) {  // half to int16
            GenCastCall_8<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int8_t>::value &&
                             std::is_same<typename TileDataS::DType, half>::value) {  // half to int8
            GenCastCall_9<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, uint8_t>::value &&
                             std::is_same<typename TileDataS::DType, half>::value) {  // half to uint8
            GenCastCall_10<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, int32_t>::value &&
                             std::is_same<typename TileDataS::DType, bfloat16_t>::value) {  // bfloat16 to int32
            GenCastCall_11<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);        
        } else if constexpr (std::is_same<typename TileDataD::DType, half>::value &&
                             std::is_same<typename TileDataS::DType, int16_t>::value) {  // int16 to half
            GenCastCall_12<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                             std::is_same<typename TileDataS::DType, int32_t>::value) {  // int32 to float
            GenCastCall_13<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else if constexpr (std::is_same<typename TileDataD::DType, float>::value &&
                             std::is_same<typename TileDataS::DType, int64_t>::value) {  // int64 to float
            GenCastCall_14<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        } else {
            GenCastCall_15<TileDataD, TileDataS>(dst,src,repeatNum, mode, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        }              
    }

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
                        (uint8_t)REPEAT_MAX,
                        mode,
                        1,
                        1,
                        (uint16_t)dstRepeatStride,
                        (uint16_t)srcRepeatStride);
                }
            }
            if (remainAfterLoop > 0) {
                GenCastCall<TileDataD, TileDataS>(dstPtr + i * DS + numLoop * elementsPerRepeat * REPEAT_MAX,
                    srcPtr + i * SS + numLoop * elementsPerRepeat * REPEAT_MAX,
                    (uint8_t)remainAfterLoop,
                    mode,
                    1,
                    1,
                    (uint16_t)dstRepeatStride,
                    (uint16_t)srcRepeatStride);
            }   
        }
    }
   
    template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
    __tf__ AICORE void TCvt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src,
        RoundMode mode, unsigned numRepeatPerLine, unsigned numRemainPerLine, unsigned validRow, unsigned elementsPerRepeat,
        unsigned dstRepeatStride, unsigned srcRepeatStride)
    {
        __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataS::DType *srcPtr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);
        constexpr unsigned dstNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
        constexpr unsigned srcNElemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType);
        if (numRepeatPerLine > 0) {
            TCvtHead<TileDataD, TileDataS, SS, DS>(dstPtr, srcPtr, mode, numRepeatPerLine, validRow, elementsPerRepeat, dstRepeatStride, srcRepeatStride);
        }
        dstPtr += numRepeatPerLine * elementsPerRepeat;
        srcPtr += numRepeatPerLine * elementsPerRepeat;
        if (numRemainPerLine > 0) {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            SetContinuousMask(numRemainPerLine);
            if (numLoop > 0) {
                for (uint32_t j = 0; j < numLoop; j++) {
                    GenCastCall<TileDataD, TileDataS>(dstPtr + j * DS * REPEAT_MAX,
                        srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX, mode,
                        1, 1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
                }
            }
            if (remainAfterLoop > 0) {
                GenCastCall<TileDataD, TileDataS>(dstPtr + numLoop * DS * REPEAT_MAX,
                    srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop,
                    mode, 1, 1, (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileDataD, typename TileDataS>
    AICORE void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
    {
        uint64_t repeatWidth = 
            static_cast<uint64_t>(max(sizeof(typename TileDataD::DType), sizeof(typename TileDataS::DType)));
        
        unsigned dstRepeatStride = 
            repeatWidth == sizeof(typename TileDataD::DType)
            ? BLOCK_MAX_PER_REPEAT
            : (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataS::DType) * sizeof(typename TileDataD::DType));
        unsigned srcRepeatStride = 
            repeatWidth == sizeof(typename TileDataS::DType)
            ? BLOCK_MAX_PER_REPEAT
            : (BLOCK_MAX_PER_REPEAT / sizeof(typename TileDataD::DType) * sizeof(typename TileDataS::DType));
        unsigned elementsPerRepeat = REPEAT_BYTE / repeatWidth;
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned SS = TileDataS::RowStride;
        constexpr unsigned DS = TileDataD::RowStride;
        unsigned validRow = dst.GetValidRow();
        TCvt<TileDataD, TileDataS, SS, DS>(dst.data(),
            src.data(),
            mode,
            numRepeatPerLine,
            numRemainPerLine,
            validRow,
            elementsPerRepeat,
            dstRepeatStride,
            srcRepeatStride);
    }
}  // namespace pto
#endif