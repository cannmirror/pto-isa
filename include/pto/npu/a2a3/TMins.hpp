/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMINS_HPP
#define TMINS_HPP

#include <pto/common/constants.hpp>
#include "TBinSOp.hpp"
namespace pto
{
    template<typename T>
    struct MinSOp {
        __PTO_INSTR__ static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats) {
            vmins(dst, src0, src1, repeats, 1, 1, 8, 8);
        }
        __PTO_INSTR__ static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
            vmins(dst, src0, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
        }
    };
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ __PTO_INSTR__ void TMinS(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::DType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
    using T = typename TileData::DType;
	__ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
   	__ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);

    TBinSInstr<MinSOp<typename TileData::DType>, TileData, elementsPerRepeat, blockSizeElem, stride>(
            dstPtr, src0Ptr, src1, validRow, validCol);
}
    template <typename TileData>
    __PTO_INSTR__ void TMINS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TMinS<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), scalar, validRow, validCol);
    }
}

#endif
