/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRIL_HPP
#define TTRIL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto 
{
template <typename TileData, unsigned rowStride, int diagonal>
__tf__ PTO_INTERNAL void  TTriu(
    typename TileData::TileDType __out__ dst,
    unsigned validRows,
    unsigned validCols
) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);  
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
    constexpr uint32_t start_row = (diagonal > 0) ? 0  : (1-diagonal);
    constexpr uint32_t start_num = diagonal;
    __VEC_SCOPE__
    {
        RegTensor<T> v_ones, v_zeros;
        vbr(v_ones,  (T)1);
        vbr(v_zeros, (T)0);
        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        //store ones
        for (uint16_t i = 0; i < (uint16_t) validRows; ++i) {
            uint32_t num_ones = validCols;
            for (uint16_t j = 0; j < (uint16_t) numRepeatPerRow; ++j){
                vector_bool preg_ones = CreatePredicate<T>(num_ones);
                vsts(v_ones, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_ones);
            }
        }
        
        //store zeros
        for (uint16_t i = start_row; i < (uint16_t) validRows; ++i) {
            uint32_t num_zeros = i + start_num;
            for (uint16_t j = 0; j < (uint16_t) numRepeatPerRow; ++j){
                vector_bool preg_zeros  = CreatePredicate<T>(num_zeros);
                vsts(v_zeros, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_zeros);
            }
        }
    }
}

template <typename TileData, unsigned rowStride, int diagonal>
__tf__ PTO_INTERNAL void  TTril(
    typename TileData::TileDType __out__ dst,
    unsigned validRows,
    unsigned validCols
) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);  
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
    constexpr uint32_t start_row = (diagonal < 0) ? (-diagonal)  : (0);
    constexpr uint32_t start_num = diagonal+1;
    __VEC_SCOPE__
    {
        RegTensor<T> v_ones, v_zeros;
        vbr(v_ones,  (T)1);
        vbr(v_zeros, (T)0);
        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        //store zeros
        for (uint16_t i = 0; i < (uint16_t) validRows; ++i) {
            uint32_t num_zeros = validCols;
            for (uint16_t j = 0; j < (uint16_t) numRepeatPerRow; ++j){
                vector_bool preg_zeros = CreatePredicate<T>(num_zeros);
                vsts(v_zeros, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_zeros);
            }
        }
        
        //store ones
        for (uint16_t i = start_row; i < (uint16_t) validRows; ++i) {
            uint32_t num_ones = i + start_num;
            for (uint16_t j = 0; j < (uint16_t) numRepeatPerRow; ++j){
                vector_bool preg_ones  = CreatePredicate<T>(num_ones);
                vsts(v_ones, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_ones);
            }
        }
    }
}

template <typename TileData, int upperOrLower, int diagonal>
PTO_INTERNAL void TTRIL_IMPL(TileData &dst)
{
    using T = typename TileData::DType;
    static_assert(std::is_same<T, int32_t>::value   ||
                    std::is_same<T, int16_t>::value   ||
                    std::is_same<T, int8_t>::value    ||
                    std::is_same<T, uint32_t>::value  ||
                    std::is_same<T, uint16_t>::value  ||
                    std::is_same<T, uint8_t>::value   ||
                    std::is_same<T, half>::value      ||
                    std::is_same<T, float16_t>::value ||
                    std::is_same<T, float32_t>::value ||
                    std::is_same<T, bfloat16_t>::value,
                    "TTRIL: Invalid data type.");
    
    if constexpr (upperOrLower == 0)
        TTril<TileData, TileData::RowStride, diagonal>(dst.data(), dst.GetValidRow(), dst.GetValidCol());
    else
        TTriu<TileData, TileData::RowStride, diagonal>(dst.data(), dst.GetValidRow(), dst.GetValidCol());
}
}  // namespace pto

#endif // TTRIL_HPP