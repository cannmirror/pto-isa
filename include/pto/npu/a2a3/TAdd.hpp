/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TADD_HPP
#define TADD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/TBinOp.hpp"

namespace pto {

template <typename T> struct AddOp {
    __PTO_INSTR__ static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats)
    {
        vadd(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
    }
    __PTO_INSTR__ static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
                                uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        vadd(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ __PTO_INSTR__ void TAdd(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0, 
    typename TileData::TileDType __in__ src1, unsigned validRow, unsigned validCol)
{    
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<AddOp<T>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
}

template <typename TileData>
__PTO_INSTR__ void TADD_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                  std::is_same<typename TileData::DType, int>::value ||
                  std::is_same<typename TileData::DType, int16_t>::value ||
                  std::is_same<typename TileData::DType, half>::value ||
                  std::is_same<typename TileData::DType, float16_t>::value ||
                  std::is_same<typename TileData::DType, float>::value ||
                  std::is_same<typename TileData::DType, float32_t>::value,
                  "TADD: Invalid data type.");
    static_assert(TileData::isRowMajor, "TADD: not supported Layout type.");
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>
        (dst.data(), src0.data(), src1.data(), validRow, validCol);
}
}  // namespace pto
#endif