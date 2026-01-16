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
#include <pto/common/utils.hpp>
#include <array>
#include "common.hpp"
#include "utils.hpp"


namespace pto {

// Import rounding type definitions from __cce_simd namespace
using __cce_simd::RoundRType;
using __cce_simd::RoundAType;
using __cce_simd::RoundFType;
using __cce_simd::RoundCType;
using __cce_simd::RoundZType;
using __cce_simd::RoundOType;

/**
 * Unified enum for all type conversion modes
 * Describes the vcvt intrinsic parameter pattern used for conversion
 */
enum class CastMode {
    EXPAND,          // vcvt(..., PART_EVEN) - Type expansion only, no conversion
    ROUND,           // vcvt(..., R()) - Conversion with rounding only
    ROUND_SAT,       // vcvt(..., R(), RS_ENABLE) - Conversion with rounding and saturation
    ROUND_PART,      // vcvt(..., R(), PART_EVEN) - Conversion with rounding and part operation
    ROUND_SAT_PART,  // vcvt(..., R(), RS_ENABLE, PART_EVEN) - Rounding, saturation, and part
    SAT_PART,        // vcvt(..., RS_ENABLE, PART_EVEN) - Saturation and part (no rounding)
    SAT_ROUND        // vcvt(..., RS_ENABLE, R()) - Saturation then rounding (reversed order)
};

#define FOR_ELEMENTS(elNum) constexpr uint16_t elementsNum = (elNum);\
    uint16_t repeatTimes = CeilDivision(len, elementsNum);\
    for(uint16_t idx = 0; idx < repeatTimes; idx++) {


#define END_FOR_ELEMENTS srcOffset += elementsNum;dstOffset += elementsNum;}

//--- Common templates --------------------------------------------------------
/**
 * Cast 64-bit integer to 32-bit (signed/float)
 * Handles: s64 -> s32 #sat #part, s64 -> f32 #rnd #part
 * Intrinsics:
 *   vcvt(output, input, preg, RS_ENABLE, PART_EVEN)  // s64 -> s32 with saturation
 *   vcvt(output, input, preg, R(), PART_EVEN)        // s64 -> f32 with rounding
 */
template <typename R, typename DST, typename SRC>
inline AICORE void castS64to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    vector_s64 v_input_0;

    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t len64 = len * 2; // As we operate with 64bit blocks using 32bit operations
    MaskReg preg_b64 = CreatePredicate<float>(len64);

    FOR_ELEMENTS(ELE_CNT_B64)
        RegTensor<DST> v_output;
        uint32_t len_even = len * 2; // As only the even part is taken
        MaskReg preg_b32 = CreatePredicate<float>(len_even);
        
        vlds(v_input_0, src, srcOffset, NORM);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output, v_input_0, preg_b64, RS_ENABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b64, R(), PART_EVEN);
        }
        vsts(v_output, dst, dstOffset, PK_B64, preg_b32);
   END_FOR_ELEMENTS
}

/**
 * Cast 32-bit to 16-bit types
 * Handles: f32 -> f16 #rnd #sat #part, f32 -> bf16 #rnd #sat #part, f32 -> s16 #rnd #sat #part
 * Intrinsics:
 *   vcvt(out_odd, in_1, preg, RS_ENABLE, PART_ODD/EVEN)       // No rounding mode (saturation only)
 *   vcvt(out_odd, in_1, preg, R(), RS_ENABLE, PART_ODD/EVEN)  // With rounding mode
 */
template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    
    FOR_ELEMENTS(ELE_CNT_B16)
        RegTensor<SRC> v_input_0, v_input_1;
        RegTensor<DST> v_output_odd, v_output_even, v_output;
        MaskReg preg_b16 = CreatePredicate<half>(len);

        vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B32);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output_odd, v_input_1, preg_b32, RS_ENABLE, PART_ODD);
            vcvt(v_output_even, v_input_0, preg_b32, RS_ENABLE, PART_EVEN);    
        } else {
            vcvt(v_output_odd, v_input_1, preg_b32, R(), RS_ENABLE, PART_ODD);
            vcvt(v_output_even, v_input_0, preg_b32, R(), RS_ENABLE, PART_EVEN);
        }
        vor(v_output, v_output_even, v_output_odd, preg_b16);
        vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
   END_FOR_ELEMENTS
}

/**
 * Cast between 32-bit types (float <-> int)
 * Modes:
 *   ROUND_SAT: f32 -> s32 #rnd #sat → vcvt(output, input, preg, R(), RS_ENABLE)
 *   ROUND:     s32 -> f32 #rnd     → vcvt(output, input, preg, R())
 */
template <typename R, CastMode MODE, typename DST, typename SRC>
inline AICORE void cast32to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {

    FOR_ELEMENTS(ELE_CNT_B32)
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b32 = CreatePredicate<float>(len);
        
        vlds(v_input_0, src, srcOffset, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT) {
            vcvt(v_output, v_input_0, preg_b32, R(), RS_ENABLE);
        } else {
            vcvt(v_output, v_input_0, preg_b32, R());
        }
        vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
}

/**
 * Cast 32-bit to 64-bit signed integer
 * Handles: s32 -> s64 #part, f32 -> s64 #rnd #sat #part
 * Intrinsics:
 *   vcvt(output, input, preg, PART_EVEN)                    // s32 -> s64 (type expansion)
 *   vcvt(output, input, preg, R(), RS_ENABLE, PART_EVEN)    // f32 -> s64 (with rounding and saturation)
 */
template <typename R, typename SRC>
inline AICORE void cast32toS64(__ubuf__ int64_t *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {

    const uint32_t ELE_CNT_B64 = ELE_CNT_B32 / 2;
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);

    FOR_ELEMENTS(ELE_CNT_B64)
        RegTensor<SRC> v_input_0;
        vector_s64 v_output;
        uint32_t len64 = len * 2; // As we operate with 64bit blocks using 32bit operations
        MaskReg preg_b64 = CreatePredicate<float>(len64);
        
        vlds(v_input_0, src, srcOffset, UNPK_B32);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output, v_input_0, preg_b32, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b32, R(), RS_ENABLE, PART_EVEN);
        }
        vsts(v_output, dst, dstOffset, NORM_B32, preg_b64);
    END_FOR_ELEMENTS
}

/**
 * Cast between 16-bit types
 * Modes:
 *   ROUND_SAT:  f16 -> s16 #rnd #sat → vcvt(output, input, preg, R(), RS_ENABLE)
 *   SAT_ROUND:  bf16 -> f16 #sat #rnd → vcvt(output, input, preg, RS_ENABLE, R()) [reversed order]
 *   ROUND:      s16 -> f16 #rnd      → vcvt(output, input, preg, R())
 */
template <typename R, CastMode MODE, typename DST, typename SRC >
inline AICORE void cast16to16(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {

    FOR_ELEMENTS(ELE_CNT_B16)
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b16 = CreatePredicate<half>(len);
        
        vlds(v_input_0, src, srcOffset, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT) {
            vcvt(v_output, v_input_0, preg_b16, R(), RS_ENABLE);
        } else if constexpr (MODE == CastMode::SAT_ROUND) {
            vcvt(v_output, v_input_0, preg_b16, RS_ENABLE, R());
        } else {
            vcvt(v_output, v_input_0, preg_b16, R());
        }
        vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
}

/**
 * Cast 16-bit to 32-bit types
 * Modes:
 *   EXPAND:          Type expansion (f16/bf16/s16 -> f32/u32/s32 #part) → vcvt(output, input, preg, PART_EVEN)
 *   ROUND_PART:      f16 -> s32 #rnd #part                             → vcvt(output, input, preg, R(), PART_EVEN)
 *   ROUND_SAT_PART:  bf16 -> s32 #rnd #sat #part                       → vcvt(output, input, preg, R(), RS_ENABLE, PART_EVEN)
 */
template <typename R, CastMode MODE, typename DST, typename SRC >
inline AICORE void cast16to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {

    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ELEMENTS(ELE_CNT_B32)
        RegTensor<SRC> v_input_0;
        RegTensor<DST> v_output;
        MaskReg preg_b32 = CreatePredicate<float>(len);
        
        vlds(v_input_0, src, srcOffset, UNPK_B16);
        if constexpr (MODE == CastMode::EXPAND) {
            vcvt(v_output, v_input_0, preg_b16, PART_EVEN);
        } else if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            vcvt(v_output, v_input_0, preg_b16, R(), RS_ENABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b16, R(), PART_EVEN);
        }
        vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
}

/**
 * Cast 16-bit to 8-bit types
 * Modes:
 *   ROUND_SAT_PART: f16 -> s8/u8 #rnd #sat #part → vcvt(..., R(), RS_ENABLE, PART_*)
 *   SAT_PART:       s16 -> u8 #sat #part         → vcvt(..., RS_ENABLE, PART_*)
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B16))) SRC_VEC;
   
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ELEMENTS(ELE_CNT_B8)
        SRC_VEC v_input_0, v_input_1;
        DST_VEC v_output_odd, v_output_even, v_output;
        MaskReg preg_b8 = CreatePredicate<uint8_t>(len);

        vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
        if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            vcvt(v_output_odd, v_input_1, preg_b16, R(), RS_ENABLE, PART_ODD);
            vcvt(v_output_even, v_input_0, preg_b16, R(), RS_ENABLE, PART_EVEN);
        } else {
            // SAT_PART mode: s16 -> u8 without rounding
            vcvt(v_output_odd, v_input_1, preg_b16, RS_ENABLE, PART_ODD);
            vcvt(v_output_even, v_input_0, preg_b16, RS_ENABLE, PART_EVEN);
        }
        vor(v_output, v_output_even, v_output_odd, preg_b8);
        vsts(v_output, dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
}

/**
 * Cast 8-bit to 16-bit types
 * Handles: u8/s8 -> f16/u16/s16 #part (type expansion)
 * Intrinsic: vcvt(output, input, preg, PART_EVEN)
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to16(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B16))) DST_VEC;
   
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);

    FOR_ELEMENTS(ELE_CNT_B16)
        SRC_VEC v_input_0;
        DST_VEC v_output;
        MaskReg preg_b16 = CreatePredicate<half>(len);

        vlds(v_input_0, src, srcOffset, UNPK_B8);
        vcvt(v_output, v_input_0, preg_b8, PART_EVEN);
        vsts(v_output, dst, dstOffset, NORM_B16, preg_b16);
    END_FOR_ELEMENTS
}

/**
 * Cast 8-bit to 32-bit types (FP8 to FP32 type expansion)
 * Handles: e4m3/e5m2/h8 -> f32 #pp
 * Intrinsic: vcvt(output, input, preg, PART_*)
 */
template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B32))) DST_VEC;
   
    uint32_t len8 = ELE_CNT_B8;
    MaskReg preg_b8 = CreatePredicate<uint8_t>(len8);
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);
    MaskReg pg = pset_b8(PAT_ALL);
    SRC_VEC v_zero;
    vdup((RegTensor<uint8_t> &) v_zero, 0, pg, MODE_ZEROING);  
    uint32_t next_len = (len > 64) ? len - 64 : 0;

    FOR_ELEMENTS(ELE_CNT_B16)
        SRC_VEC v_input_0, v_input_1, v_input_2;
        DST_VEC v_output_0, v_output_1;
        MaskReg preg_b16 = CreatePredicate<half>(len);
        MaskReg preg_b16_next = CreatePredicate<half>(next_len);
        MaskReg preg_b32;
        MaskReg preg_b32_next;
        punpack(preg_b32, preg_b16, LOWER);
        punpack(preg_b32_next, preg_b16_next, LOWER);

        vlds((RegTensor<uint8_t> &) v_input_0, (__ubuf__ uint8_t *) src, srcOffset, UNPK_B8);
        vintlv((RegTensor<uint8_t> &) v_input_1, (RegTensor<uint8_t> &) v_input_2, (RegTensor<uint8_t> &) v_input_0, (RegTensor<uint8_t> &) v_zero); // interleave with zero
        vcvt(v_output_0, v_input_1, preg_b8, PART_P0);
        vcvt(v_output_1, v_input_2, preg_b8, PART_P0);
        vsts(v_output_0, dst, dstOffset + ELE_CNT_B32 * (idx * 2), NORM_B32, preg_b32);
        vsts(v_output_1, dst, dstOffset + ELE_CNT_B32 * (idx * 2 + 1), NORM_B32, preg_b32_next);
    END_FOR_ELEMENTS
}

/**
 * Cast 32-bit to 8-bit types (both floating point and integer)
 * Handles: 
 *   - f32 -> e4m3/e5m2/h8 #rnd #sat #pp (ROUND_SAT_PART mode)
 *   - u32/s32 -> u8/s8 #sat #pp (SAT_PART mode)
 * Intrinsics:
 *   vcvt(..., R(), RS_ENABLE, PART_*) for floating point with rounding
 *   vcvt(..., RS_ENABLE, PART_*) for integer without rounding
 */
template <typename R, CastMode MODE, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast32to8(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B32))) SRC_VEC;

    constexpr int INPUT_VL_LEN = 64; // Max vector length for 8-bit output
    uint32_t preg_len_head = INPUT_VL_LEN;
    uint32_t preg_len_tail = (len % INPUT_VL_LEN == 0) ? INPUT_VL_LEN : (len % INPUT_VL_LEN);
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    MaskReg preg_idx = pset_b8(PAT_ALL);
    DST_VEC v_idx;
    vci((RegTensor<int8_t> &) v_idx, (int8_t) 0 , INC_ORDER);
    vmuls((RegTensor<int16_t> &) v_idx, (RegTensor<int16_t> &) v_idx, (int16_t) 4, preg_idx); // multiply by 4 for byte addressing

    FOR_ELEMENTS(ELE_CNT_B32)
        SRC_VEC v_input_0;
        DST_VEC v_output_0, v_output;
        uint32_t preg_len = (idx == repeatTimes - 1) ? preg_len_tail : preg_len_head;
        MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_len);

        vlds(v_input_0, src, srcOffset, NORM);
        if constexpr (MODE == CastMode::ROUND_SAT_PART) {
            // Floating point conversion with rounding
            vcvt(v_output_0, v_input_0, preg_b32, ROUND_R, RS_ENABLE, PART_P0);
            vselr((RegTensor<uint8_t> &) v_output, (RegTensor<uint8_t> &) v_output_0, (RegTensor<uint8_t> &) v_idx);
        } else {
            // Integer conversion without rounding (SAT_PART mode)
            vcvt(v_output_0, v_input_0, preg_b32, RS_ENABLE, PART_P0);
            vselr((RegTensor<uint8_t> &) v_output, (RegTensor<uint8_t> &) v_output_0, (RegTensor<uint8_t> &) v_idx);
        }
        vsts((RegTensor<uint8_t> &) v_output, (__ubuf__ uint8_t *) dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
}


//--- Src:: FP32 ----------------------------------------------------------------------
/**
 * FP32 to FP32 - Applies rounding mode without type conversion
 * Intrinsic: vtrc(output, input, R(), preg)
 */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    FOR_ELEMENTS(ELE_CNT_B32)
        vector_f32 v_input_0, v_output;
        MaskReg preg_b32 = CreatePredicate<float>(len);
        
        vlds(v_input_0, src, srcOffset, NORM);
        vtrc(v_output, v_input_0, R(), preg_b32);
        vsts(v_output, dst, dstOffset, NORM_B32, preg_b32);
    END_FOR_ELEMENTS
}

/**
 * FP32 to FP16
 * Conversion: f32 -> f16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ float16_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<R>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to BF16
 * Conversion: f32 -> bf16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<R>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to I16
 * Conversion: f32 -> s16 #rnd #sat #part
 * Uses cast32to16 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<R>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to I32
 * Conversion: f32 -> s32 #rnd #sat
 * Intrinsic: vcvt(output, input, preg, R(), RS_ENABLE)
 */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to32<R, CastMode::ROUND_SAT>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to I64
 * Conversion: f32 -> s64 #rnd #sat #part
 * Uses cast32toS64 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ int64_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32toS64<R>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to FP8_E4M3
 * Conversion: f32 -> e4m3 #rnd #sat #pp
 * Uses cast32to8 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e4m3_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to FP8_E5M2
 * Conversion: f32 -> e5m2 #rnd #sat #pp
 * Uses cast32to8 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e5m2_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, dstOffset, srcOffset, len);
}

/**
 * FP32 to H8
 * Conversion: f32 -> h8 #rnd #sat #pp
 * Uses cast32to8 helper
 */
template <typename R>
inline AICORE void castData(__ubuf__ hifloat8_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    constexpr int INPUT_VL_LEN = 64; // Max vector length for 8-bit output
    uint32_t preg_len_head = INPUT_VL_LEN;
    uint32_t preg_len_tail = (len % INPUT_VL_LEN == 0) ? INPUT_VL_LEN : (len % INPUT_VL_LEN);
    uint32_t len32 = ELE_CNT_B32;
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    MaskReg preg_idx = pset_b8(PAT_ALL);
    vector_u8 v_idx;
    vci((RegTensor<int8_t> &) v_idx, (int8_t) 0 , INC_ORDER);
    vmuls((RegTensor<int16_t> &) v_idx, (RegTensor<int16_t> &) v_idx, (int16_t) 4, preg_idx); // multiply by 4 for byte addressing
    
    FOR_ELEMENTS(ELE_CNT_B32)
        vector_f32 v_input_0;
        vector_hif8 v_output_0, v_output;
        uint32_t preg_len = (idx == repeatTimes - 1) ? preg_len_tail : preg_len_head;
        MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_len);
        vlds(v_input_0, src, srcOffset, NORM);
        vcvt(v_output_0, v_input_0, preg_b32, ROUND_A, RS_ENABLE, PART_P0);
        vselr((RegTensor<uint8_t> &) v_output, (RegTensor<uint8_t> &) v_output_0, (RegTensor<uint8_t> &) v_idx);
        vsts((RegTensor<uint8_t> &) v_output, (__ubuf__ uint8_t *) dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
}

//--- Src:: FP16 ----------------------------------------------------------------------
/** FP16 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<void, CastMode::EXPAND>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> I32 #rnd #part → vcvt(output, input, preg, R(), PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<R, CastMode::ROUND_PART>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> I16 #rnd #sat → vcvt(output, input, preg, R(), RS_ENABLE) */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to16<R, CastMode::ROUND_SAT>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> I8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ int8_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_s8>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> U8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_u8>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> FP8_E5M2 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e5m2_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> FP8_E4M3 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e4m3_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, dstOffset, srcOffset, len);
}

/** FP16 -> H8 #rnd #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ hifloat8_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16->H8 conversion only supports ROUND_A or ROUND_H modes
    // static_assert(std::is_same<R, RoundAType>::value || std::is_same<R, RoundCType>::value,
    //               "Fix: FP16 to HIFLOAT8 conversion only supports ROUND_A (CAST_ROUND) or ROUND_H (CAST_CEIL) rounding modes");
    uint32_t len16 = ELE_CNT_B16;
    MaskReg preg_b16 = CreatePredicate<half>(len16);

    FOR_ELEMENTS(ELE_CNT_B8)
        vector_f16 v_input_0, v_input_1;
        vector_hif8 v_output_odd, v_output_even, v_output;
        MaskReg preg_b8 = CreatePredicate<uint8_t>(len);

        vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
        vcvt(v_output_odd, v_input_1, preg_b16, ROUND_A, RS_ENABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_b16, ROUND_A, RS_ENABLE, PART_EVEN);
        vor((RegTensor<uint8_t> &) v_output, (RegTensor<uint8_t> &) v_output_even, (RegTensor<uint8_t> &) v_output_odd, preg_b8);
        vsts((RegTensor<uint8_t> &) v_output, (__ubuf__ uint8_t *) dst, dstOffset, NORM_B8, preg_b8);
    END_FOR_ELEMENTS
}

//--- Src:: BF16 ----------------------------------------------------------------------
/** BF16 -> FP32 #part (type expansion) → vcvt(output, input, preg, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<void, CastMode::EXPAND>(dst, src, dstOffset, srcOffset, len);
}

/** BF16 -> I32 #rnd #sat #part → vcvt(output, input, preg, R(), RS_ENABLE, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<R, CastMode::ROUND_SAT_PART>(dst, src, dstOffset, srcOffset, len);
}

/** BF16 -> F16 #sat #rnd → vcvt(output, input, preg, RS_ENABLE, R()) [reversed order] */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to16<R, CastMode::SAT_ROUND>(dst, src, dstOffset, srcOffset, len);
}

/** BF16 -> FP8_E5M2 #rnd #sat #part → vcvt(..., R(), RS_ENABLE, PART_*) */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e5m2_t *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_f8e5m2>(dst, src, dstOffset, srcOffset, len);
}

/** BF16 -> FP8_E4M3 #rnd #sat #part → vcvt(..., R(), RS_ENABLE, PART_*) */
template <typename R>
inline AICORE void castData(__ubuf__ float8_e4m3_t *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<R, CastMode::ROUND_SAT_PART, vector_f8e4m3>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: U8,I8 ----------------------------------------------------------------------
/** U8 -> FP16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ uint8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to16<vector_u8>(dst, src, dstOffset, srcOffset, len);
}

/** U8 -> U16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to16<vector_u8>(dst, src, dstOffset, srcOffset, len);
}

/** I8 -> FP16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ int8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to16<vector_s8>(dst, src, dstOffset, srcOffset, len);
}

/** I8 -> I16 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to16<vector_s8>(dst, src, dstOffset, srcOffset, len);
}

/** I8 -> I32 #pp (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to32<vector_s8>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: I16 ----------------------------------------------------------------------
/** I16 -> U8 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to8<void, CastMode::SAT_PART, vector_u8>(dst, src, dstOffset, srcOffset, len);
}

/** I16 -> FP16 #rnd → vcvt(output, input, preg, R()) */
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to16<R, CastMode::ROUND>(dst, src, dstOffset, srcOffset, len);
}

/** I16 -> FP32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<void, CastMode::EXPAND>(dst, src, dstOffset, srcOffset, len);
}

/** I16 -> U32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<void, CastMode::EXPAND>(dst, src, dstOffset, srcOffset, len);
}

/** I16 -> I32 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast16to32<void, CastMode::EXPAND>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: I32 ----------------------------------------------------------------------
/** I32 -> FP32 #rnd → vcvt(output, input, preg, R()) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to32<R, CastMode::ROUND>(dst, src, dstOffset, srcOffset, len);
}

/** I32 -> I16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<void>(dst, src, dstOffset, srcOffset, len);
}

/** I32 -> U16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<void>(dst, src, dstOffset, srcOffset, len);
}

/** I32 -> I64 #part (type expansion) */
template <typename R>
inline AICORE void castData(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32toS64<void>(dst, src, dstOffset, srcOffset, len);
}

/** I32 -> U8 #sat #pp */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: U32 ----------------------------------------------------------------------
/** U32 -> U8 #sat #pp */
template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ uint32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to8<void, CastMode::SAT_PART, vector_u8>(dst, src, dstOffset, srcOffset, len);
}

/** U32 -> U16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<void>(dst, src, dstOffset, srcOffset, len);
}

/** U32 -> I16 #sat #part */
template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ uint32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast32to16<void>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: I64 ----------------------------------------------------------------------
/** I64 -> FP32 #rnd #part → vcvt(output, input, preg, R(), PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int64_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    castS64to32<R>(dst, src, dstOffset, srcOffset, len);
}

/** I64 -> I32 #sat #part → vcvt(output, input, preg, RS_ENABLE, PART_EVEN) */
template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    castS64to32<void>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: FP8 ----------------------------------------------------------------------
/** E4M3 -> FP32 #pp (type expansion) → vcvt(output, input, preg, PART_EVEN/ODD) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float8_e4m3_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to32<vector_f8e4m3>(dst, src, dstOffset, srcOffset, len);
}

/** E5M2 -> FP32 #pp (type expansion) → vcvt(output, input, preg, PART_EVEN/ODD) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float8_e5m2_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to32<vector_f8e5m2>(dst, src, dstOffset, srcOffset, len);
}

/** H8 -> FP32 #pp (type expansion) → vcvt(output, input, preg, PART_EVEN/ODD) */
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ hifloat8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    cast8to32<vector_hif8>(dst, src, dstOffset, srcOffset, len);
}

/**
 * Main TCVT implementation function
 * Converts tile data from source type to destination type using specified rounding mode
 * Iterates over rows and calls appropriate castData specialization
 */
template <typename TileDataD, typename TileDataS, typename R>
__tf__ PTO_INTERNAL OP_NAME(TCVT) OP_TYPE(element_wise)
void implTCVT(typename TileDataD::TileDType __out__ dst, 
              typename TileDataS::TileDType __in__ src, 
    unsigned validRows, unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T1 = typename TileDataD::DType;
    using T2 = typename TileDataS::DType;
    __ubuf__ T1 *dstPtr = (__ubuf__ T1 *)__cce_get_tile_ptr(dst);
    __ubuf__ T2 *srcPtr = (__ubuf__ T2 *)__cce_get_tile_ptr(src);
    __VEC_SCOPE__ {
        uint16_t rows = (uint16_t) validRows;
        uint16_t cols = (uint16_t) validCols;
        for (uint16_t row = 0; row < rows; row++) {
            int32_t dstOffset = row * TileDataD::Cols;
            int32_t srcOffset = row * TileDataS::Cols;
            castData<R>(dstPtr, srcPtr, dstOffset, srcOffset, cols);
        }
    }
}

template <typename TileDataD, typename TileDataS>
AICORE void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
{
    switch (mode) {
        case RoundMode::CAST_RINT:
            implTCVT<TileDataD,TileDataS,RoundRType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            break;
        case RoundMode::CAST_ROUND:
            implTCVT<TileDataD,TileDataS,RoundAType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            break;
        case RoundMode::CAST_FLOOR:
            implTCVT<TileDataD,TileDataS,RoundFType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            break;
        case RoundMode::CAST_CEIL:
            implTCVT<TileDataD,TileDataS,RoundCType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            break;
        case RoundMode::CAST_TRUNC:
            implTCVT<TileDataD,TileDataS,RoundZType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            break;
        case RoundMode::CAST_ODD:
            if constexpr (std::is_same<typename TileDataD::DType, half>::value && 
                std::is_same<typename TileDataS::DType, float>::value) {
                implTCVT<TileDataD,TileDataS,RoundOType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            } 
            break;
        default:
            implTCVT<TileDataD,TileDataS,RoundRType>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            break;
    }
}

}  // namespace pto
#endif