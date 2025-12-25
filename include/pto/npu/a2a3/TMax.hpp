/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMAX_HPP
#define TMAX_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/TBinOp.hpp"
#include "pto/npu/a2a3/TBinPlusOp.hpp"

namespace pto {

template <typename T> struct MaxOp {
    PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vmax(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        vmax(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL void TMax(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, unsigned validRow, unsigned validCol)
{    
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<MaxOp<T>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
}

template <typename TileData>
PTO_INTERNAL void TMAX_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                  std::is_same<typename TileData::DType, int>::value ||
                  std::is_same<typename TileData::DType, int16_t>::value ||
                  std::is_same<typename TileData::DType, half>::value ||
                  std::is_same<typename TileData::DType, float16_t>::value ||
                  std::is_same<typename TileData::DType, float>::value ||
                  std::is_same<typename TileData::DType, float32_t>::value,
                  "TMAX: Invalid data type.");
    static_assert(TileData::isRowMajor, "TMAX: not supported Layout type.");
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TMax<TileData, elementsPerRepeat, blockSizeElem, rowStride>
        (dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ PTO_INTERNAL void TMax(typename TileDataDst::TileDType __out__ dstData,
    typename TileDataSrc0::TileDType __in__ src0Data, typename TileDataSrc1::TileDType __in__ src1Data,
    unsigned validRow, unsigned validCol) {
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src0 = (__ubuf__ T *)__cce_get_tile_ptr(src0Data);
    __ubuf__ T *src1 = (__ubuf__ T *)__cce_get_tile_ptr(src1Data);
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc0> && std::is_same_v<TileDataDst, TileDataSrc1>) {
        constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
        constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
        constexpr unsigned rowStride = TileDataDst::RowStride;
        BinaryInstr<MaxOp<T>, T, TileDataDst, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, validRow, validCol);
    } else {
        BinaryPlusInstr<MaxOp<T>, T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMAX_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, typename TileDataSrc0::DType>::value ||
                  std::is_same<T, typename TileDataSrc1::DType>::value,
                  "The data type of dst must be consistent with of src0 and src1.");

    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int>::value ||
                  std::is_same<T, int16_t>::value || std::is_same<T, half>::value ||
                  std::is_same<T, float16_t>::value || std::is_same<T, float>::value ||
                  std::is_same<T, float32_t>::value, "TMAX: Invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "TMAX: not supported Layout type.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    if ((validRow == src0.GetValidRow() && validCol == src0.GetValidCol()) &&
        (validRow == src1.GetValidRow() && validCol == src1.GetValidCol())) {
        TMax<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst.data(), src0.data(), src1.data(), validRow, validCol);
    } else {
        TPARTMAX_IMPL(dst, src0, src1);
    }
}
}  // namespace pto
#endif