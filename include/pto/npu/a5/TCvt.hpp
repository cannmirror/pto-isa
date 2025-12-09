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

using namespace pto;

namespace pto {

#define FOR_ELEMENTS(elNum) constexpr uint16_t elementsNum = (elNum);\
    uint16_t count = (len + elementsNum-1) / elementsNum;\
    for(uint16_t idx = count; idx>0; idx--) {

#define FOR_ELEMENTS_64(elNum) constexpr uint16_t elementsNum = (elNum);\
    uint16_t count = (len + elementsNum-1) / elementsNum;\
    len=len*2; /* As we operate with 64bit blocks using 32bit operations */\
    for(uint16_t idx = count; idx>0; idx--) {

#define END_FOR_ELEMENTS srcOffset += elementsNum;dstOffset += elementsNum;}

//--- Common templates --------------------------------------------------------
template <typename R, typename DST, typename SRC>
inline AICORE void castS64to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B32))) DST_VEC;
    vector_s64 v_input_0;

    FOR_ELEMENTS_64(ELE_CNT_B32/2)
        vector_bool preg_b32 = plt_b32(len, POST_UPDATE); // len should be subtructed here by ElementsNum
        DST_VEC v_output;

        vlds(v_input_0, src, srcOffset, NORM);
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output, v_input_0, preg_b32, RS_ENABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_b32, R(), PART_EVEN);
        }
        vsts(v_output, dst, dstOffset, PK_B64, preg_b32);
   END_FOR_ELEMENTS
}

template <typename R, typename DST, typename SRC>
inline AICORE void cast32to16(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B32))) SRC_VEC;
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B16))) DST_VEC;
    SRC_VEC v_input_0, v_input_1;
    uint32_t len32 = ELE_CNT_B32;
    uint32_t new_len = len * 2;
    vector_bool preg_f32 = plt_b32(len32, POST_UPDATE);
    
    FOR_ELEMENTS(ELE_CNT_B32*2)
        vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B32);

        vector_bool preg_f16 = plt_b16(new_len, POST_UPDATE); // len should be subtructed here by ElementsNum

        DST_VEC v_output_odd, v_output_even, v_output;
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output_odd, v_input_1, preg_f32, RS_ENABLE, PART_ODD);
            vcvt(v_output_even, v_input_0, preg_f32, RS_ENABLE, PART_EVEN);    
        } else {
            vcvt(v_output_odd, v_input_1, preg_f32, R(), RS_ENABLE, PART_ODD);
            vcvt(v_output_even, v_input_0, preg_f32, R(), RS_ENABLE, PART_EVEN);
        }

        vor(v_output, v_output_even, v_output_odd, preg_f16);

        vsts(v_output, dst, dstOffset, NORM_B16, preg_f16);
   END_FOR_ELEMENTS
}

template <typename R, bool EN_RS, typename DST, typename SRC>
inline AICORE void cast32to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B32))) SRC_VEC;
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B32))) DST_VEC;

    FOR_ELEMENTS(ELE_CNT_B32)
        SRC_VEC v_input_0;
        DST_VEC v_output;
        vector_bool preg_f32 = plt_b32(len, POST_UPDATE);
        
        vlds(v_input_0, src, srcOffset, NORM);
        if constexpr (EN_RS){
            vcvt(v_output, v_input_0, preg_f32, R(), RS_ENABLE);
        } else {
            vcvt(v_output, v_input_0, preg_f32, R());
        }            
        vsts(v_output, dst, dstOffset, NORM_B32, preg_f32);
    END_FOR_ELEMENTS
}

template <typename R, typename SRC>
inline AICORE void cast32toS64(__ubuf__ int64_t *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B32))) SRC_VEC;

    FOR_ELEMENTS_64(ELE_CNT_B32/2)
        SRC_VEC v_input_0;
        vector_s64 v_output;
        
        vector_bool preg_f32 = plt_b32(len, POST_UPDATE);
        
        vlds(v_input_0, src, srcOffset, UNPK_B32);
        if constexpr (std::is_same<R, void>::value){
            vcvt(v_output, v_input_0, preg_f32, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_f32, R(), RS_ENABLE, PART_EVEN);
        }
            
        vsts(v_output, dst, dstOffset, NORM_B32, preg_f32);
    END_FOR_ELEMENTS
}

template <typename R, bool EN_RS, bool EN_FIRST, typename DST, typename SRC >
inline AICORE void cast16to16(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B16))) SRC_VEC;
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B16))) DST_VEC;

    FOR_ELEMENTS(ELE_CNT_B16)
        SRC_VEC v_input_0;
        DST_VEC v_output;
        vlds(v_input_0, src, srcOffset, NORM);
        
        vector_bool preg_f16 = plt_b16(len, POST_UPDATE); // len is subtructed here by ELE_CNT_B32
        if constexpr (EN_FIRST) {
            //FP16 -> I16
            vcvt(v_output, v_input_0, preg_f16, R(), RS_ENABLE);
        } else if constexpr (EN_RS){
            //BF16 -> FP16
            vcvt(v_output, v_input_0, preg_f16, RS_ENABLE, R());
        } else {
            //I16 -> FP16
            vcvt(v_output, v_input_0, preg_f16, R());
        }
        vsts(v_output, dst, dstOffset, NORM_B16, preg_f16);
    END_FOR_ELEMENTS
}

template <typename R, bool EN_RS, typename DST, typename SRC >
inline AICORE void cast16to32(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B16))) SRC_VEC;
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B32))) DST_VEC;

    uint32_t reg_16 = ELE_CNT_B16;
    vector_bool preg_f16 = plt_b16(reg_16, POST_UPDATE);

    FOR_ELEMENTS(ELE_CNT_B32)
        SRC_VEC v_input_0;
        DST_VEC v_output;
        vector_bool preg_f32 = plt_b32(len, POST_UPDATE); // len is subtructed here by ELE_CNT_B32
        vlds(v_input_0, src, srcOffset, UNPK_B16);
        // US_B16 reads half the data and upsamples it by factor of 2 (e.g 0xa234 -> 0xa234a234)
        if constexpr (std::is_same<R, void>::value) {
            vcvt(v_output, v_input_0, preg_f16, PART_EVEN);
        } else if constexpr (EN_RS){
            vcvt(v_output, v_input_0, preg_f16, R(), RS_ENABLE, PART_EVEN);
        } else {
            vcvt(v_output, v_input_0, preg_f16, R(), PART_EVEN);
        }

        vsts(v_output, dst, dstOffset, NORM_B32, preg_f32);
    END_FOR_ELEMENTS
}

template <typename R, typename DST_VEC, typename DST, typename SRC>
inline AICORE void cast16to8(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef SRC __attribute__((ext_vector_type(ELE_CNT_B16))) SRC_VEC;
   
    uint32_t len16 = ELE_CNT_B16;
    vector_bool preg_f16 = plt_b16(len16, POST_UPDATE);
    uint32_t new_len = len * 4;

    FOR_ELEMENTS(ELE_CNT_B8)
        SRC_VEC v_input_0, v_input_1;
        DST_VEC v_output_odd, v_output_even, v_output;

        vector_bool preg_b8 = plt_b8(new_len, POST_UPDATE);

        vlds(v_input_0, v_input_1, src, srcOffset, DINTLV_B16);
        vcvt(v_output_odd, v_input_1, preg_f16, R(), RS_ENABLE, PART_ODD);
        vcvt(v_output_even, v_input_0, preg_f16, R(), RS_ENABLE, PART_EVEN);
        vor(v_output, v_output_even, v_output_odd, preg_b8);
        vsts(v_output, dst, dstOffset, NORM_B16, preg_b8);
    END_FOR_ELEMENTS
}


template <typename SRC_VEC, typename DST, typename SRC>
inline AICORE void cast8to16(__ubuf__ DST *dst, __ubuf__ SRC *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    typedef DST __attribute__((ext_vector_type(ELE_CNT_B16))) DST_VEC;
   
    uint32_t len8 = ELE_CNT_B8;
    vector_bool preg_b8 = plt_b8(len8, POST_UPDATE);    

    FOR_ELEMENTS(ELE_CNT_B16)
        SRC_VEC v_input_0;
        DST_VEC v_output;

        vlds(v_input_0, src, srcOffset, UNPK_B8);
        vector_bool preg_f16 = plt_b16(len, POST_UPDATE);

        vcvt(v_output, v_input_0, preg_b8, PART_EVEN);
        vsts(v_output, dst, dstOffset, NORM_B16, preg_f16);
    END_FOR_ELEMENTS
}


//--- Src:: FP32 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP32 to FP32
    FOR_ELEMENTS(ELE_CNT_B32)
        vector_f32 v_input_0, v_output;
        vlds(v_input_0, src, srcOffset, NORM);
        
        vector_bool preg_f32 = plt_b32(len, POST_UPDATE); // len is subtructed here by ELE_CNT_B32
        vtrc(v_output, v_input_0, R(), preg_f32);
        vsts(v_output, dst, dstOffset, NORM_B32, preg_f32);
    END_FOR_ELEMENTS
}

template <typename R>
inline AICORE void castData(__ubuf__ float16_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP32 to FP16
    cast32to16<R>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP32 to BF16
    cast32to16<R>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP32 to I16
    cast32to16<R>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP32 to I32
    cast32to32<R,true>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int64_t *dst, __ubuf__ float *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP32 to I64
    cast32toS64<R>(dst, src, dstOffset, srcOffset, len);
}


//--- Src:: FP16 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16 to FP32
    cast16to32<void, false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16 to I32
    cast16to32<R , false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16 to FP16
    FOR_ELEMENTS(ELE_CNT_B16)
        vector_f16 v_input_0, v_output;
        vlds(v_input_0, src, srcOffset, NORM);
        
        vector_bool preg_f16 = plt_b16(len, POST_UPDATE); // len is subtructed here by ELE_CNT_B32
        vtrc(v_output, v_input_0, R(), preg_f16);
        vsts(v_output, dst, dstOffset, NORM_B16, preg_f16);
    END_FOR_ELEMENTS
}

template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16 to I16
    cast16to16<R, true, false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int8_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16 to I8
    cast16to8<R, vector_s8>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ uint8_t *dst, __ubuf__ half *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // FP16 to U8
    cast16to8<R, vector_u8>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: BF16 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // BF16 to FP32
    cast16to32<void, false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // BF16 to I32
    cast16to32<R, true>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ bfloat16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // BF16 to F16
    cast16to16<R,false, true>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: U8,I8 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ uint8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // U8 to FP16
    cast8to16<vector_u8>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // U8 to U16
    cast8to16<vector_u8>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ int8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I8 to FP16
    cast8to16<vector_s8>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I8 to I16
    cast8to16<vector_s8>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: I16 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ half *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I16 to FP16
    cast16to16<R,false, false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I16 to FP32 
    cast16to32<void, false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I16 to U32 
    cast16to32<void, false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I16 to I32 
    cast16to32<void, false>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: I32 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I32 to FP32
    cast32to32<R,false>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I32 to I16
    cast32to16<void>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I32 to U16
    cast32to16<void>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I32 to I64
    cast32toS64<void>(dst, src, dstOffset, srcOffset, len);
}

//--- Src:: I64 ----------------------------------------------------------------------
template <typename R>
inline AICORE void castData(__ubuf__ float *dst, __ubuf__ int64_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I64 to FP32
    castS64to32<R>(dst, src, dstOffset, srcOffset, len);
}

template <typename R>
inline AICORE void castData(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, int32_t& dstOffset, int32_t& srcOffset, uint32_t len) {
    // I64 to I32
    castS64to32<void>(dst, src, dstOffset, srcOffset, len);
}


template <typename TileDataD, typename TileDataS, typename R>
inline AICORE void implTCVT(TileDataD &dst, TileDataS &src)
{
    uint16_t rows = src.GetValidRow();
    uint16_t cols = src.GetValidCol();
    __VEC_SCOPE__ {
        for(uint16_t row = 0; row < rows; row++){
            int32_t dstOffset=row*TileDataD::Cols;
            int32_t srcOffset=row*TileDataS::Cols;

            castData<R>(dst.data(), src.data(), dstOffset, srcOffset, cols);
        }
    }
}

template <typename TileDataD, typename TileDataS>
AICORE void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
{
    switch (mode) {
        case RoundMode::CAST_RINT:
            implTCVT<TileDataD,TileDataS,RoundRType>(dst,src);
            break;
        case RoundMode::CAST_ROUND:
            implTCVT<TileDataD,TileDataS,RoundAType>(dst,src);
            break;
        case RoundMode::CAST_FLOOR:
            implTCVT<TileDataD,TileDataS,RoundFType>(dst,src);
            break;
        case RoundMode::CAST_CEIL:
            implTCVT<TileDataD,TileDataS,RoundCType>(dst,src);
            break;
        case RoundMode::CAST_TRUNC:
            implTCVT<TileDataD,TileDataS,RoundZType>(dst,src);
            break;
        case RoundMode::CAST_ODD:
            if constexpr (std::is_same<typename TileDataD::DType, half>::value && 
                std::is_same<typename TileDataS::DType, float>::value) {
                implTCVT<TileDataD,TileDataS,RoundOType>(dst,src);
            } 
            break;
        default:
            implTCVT<TileDataD,TileDataS,RoundRType>(dst,src);
            break;
    }
}

}  // namespace pto
#endif