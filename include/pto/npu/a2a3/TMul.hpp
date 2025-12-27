/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMUL_HPP
#define TMUL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/TBinOp.hpp"
#include "pto/npu/a2a3/TBinPlusOp.hpp"

namespace pto {

template <typename T> struct MulOp {
    PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vmul(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    PTO_INTERNAL static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
        uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        vmul(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL void TMul(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, unsigned validRow, unsigned validCol)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<MulOp<T>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
        dstPtr, src0Ptr, src1Ptr, validRow, validCol);
}

template <typename TileData>
AICORE void TMUL_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                std::is_same<typename TileData::DType, int>::value ||
                std::is_same<typename TileData::DType, int16_t>::value ||
                std::is_same<typename TileData::DType, half>::value ||
                std::is_same<typename TileData::DType, float16_t>::value ||
                std::is_same<typename TileData::DType, float>::value ||
                std::is_same<typename TileData::DType, float32_t>::value,
                "TMUL: Invalid data type.");
    
    static_assert(TileData::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileData::ValidCol <= TileData::Cols, "Number of valid columns must not be greater than number of tile columns.");
    static_assert(TileData::ValidRow <= TileData::Rows, "Number of valid rows must not be greater than number of tile rows.");
    static_assert(TileData::isRowMajor, "TMul: not supported Layout type.");

    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned stride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TMul<TileData, elementsPerRepeat, blockSizeElem, stride>
        (dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ PTO_INTERNAL void TMul(typename TileDataDst::TileDType __out__ dstData,
    typename TileDataSrc0::TileDType __in__ src0Data, typename TileDataSrc1::TileDType __in__ src1Data,
    unsigned validRow, unsigned validCol) {
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src0 = (__ubuf__ T *)__cce_get_tile_ptr(src0Data);
    __ubuf__ T *src1 = (__ubuf__ T *)__cce_get_tile_ptr(src1Data);
    BinaryPlusInstr<MulOp<T>, T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, validRow, validCol);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMUL_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, typename TileDataSrc0::DType>::value ||
                  std::is_same<T, typename TileDataSrc1::DType>::value,
                  "TMUL: The data type of dst must be consistent with of src0 and src1.");

    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int>::value ||
                  std::is_same<T, int16_t>::value || std::is_same<T, half>::value ||
                  std::is_same<T, float16_t>::value || std::is_same<T, float>::value ||
                  std::is_same<T, float32_t>::value, "TMUL: Invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "TMUL: not supported Layout type.");

    static_assert((TileDataDst::Loc == TileType::Vec) &&
                  (TileDataSrc0::Loc == TileType::Vec) &&
                  (TileDataSrc1::Loc == TileType::Vec), "TileType of src and dst tiles must be Vec.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    if ((validRow == src0.GetValidRow() && validCol == src0.GetValidCol()) &&
        (validRow == src1.GetValidRow() && validCol == src1.GetValidCol())) {
        TMul<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst.data(), src0.data(), src1.data(), validRow, validCol);
    } else {
        PTO_ASSERT(false, "TMUL: dstTile validRow/validCol must be consistent with of src0 and src1.");
    }
}
}  // namespace pto
#endif