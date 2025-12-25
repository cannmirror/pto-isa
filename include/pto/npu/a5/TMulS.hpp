/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMULS_HPP
#define TMULS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"

namespace pto {

template <typename T> struct MulSOp {
    PTO_INTERNAL static void BinSInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, T src1, MaskReg &preg)
    {
        vmuls(reg_dst, reg_src0, src1, preg, MODE_ZEROING);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL OP_NAME(TMULS) OP_TYPE(element_wise)
void TMulS(typename TileData::TileDType __out__ dst, 
           typename TileData::TileDType __in__ src0, 
           typename TileData::DType src1,
           unsigned kValidRows,
           unsigned kValidCols,
           BinSOpsImpl version = BinSOpsImpl::BinSOpsIMPL_DEFAULT) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    BinaryInstr<MulSOp<T>, TileData, T, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1, kValidRows, kValidCols, version);
}

template <typename TileData>
AICORE void TMULS_IMPL(TileData &dst, TileData &src0, typename TileData::DType src1)
{
    using T = typename TileData::DType;
    static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TMULS: Invalid data type");
    static_assert(TileData::Loc == TileType::Vec, "TileType of input and output tiles must be TileType::Vec.");
    static_assert(TileData::ValidCol <= TileData::Cols, "Number of valid columns must not be greater than number of tile columns.");
    static_assert(TileData::ValidRow <= TileData::Rows, "Number of valid rows must not be greater than number of tile rows.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned rowStride = TileData::RowStride;
    
    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of input and output must be the same.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of input and output must be the same.");

    TMulS<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), src1, validRow, validCol);
}
}  // namespace pto
#endif
