/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTMAX_HPP
#define TPARTMAX_HPP

#include "TPartBinOps.hpp"

namespace pto {

template <typename T> struct TPartMaxOp {
    static constexpr T PadVal = Padding<T>::Min;
    __PTO_INSTR__ static void BinInstr(RegTensor<T> &dst, RegTensor<T> &src0, RegTensor<T> &src1,
        MaskReg preg)
    {
        vmax(dst, src0, src1, preg, MODE_ZEROING);
    }
};

template <typename DstTileData, typename Src0TileData, typename Src1TileData> 
__aicore__ PTO_INLINE void TPARTMAX_IMPL(DstTileData &dst, Src0TileData& src0, Src1TileData& src1)
{

    using T  = typename DstTileData::DType;
    using S0 = typename Src0TileData::DType;
    using S1 = typename Src1TileData::DType;

    static_assert (std::is_same_v<T, S0> && std::is_same_v<T, S1>, "TPARTMAX: Input and output types should match" );

    static_assert (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>  || std::is_same_v<T, uint16_t> || 
                   std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                   std::is_same_v<T, half>    || std::is_same_v<T, float>   || std::is_same_v<T, bfloat16_t>,
                   "TPARTMAX: Invalid data type."
    );

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    constexpr unsigned DstRowStride  = DstTileData::RowStride;
    constexpr unsigned Src0RowStride = Src0TileData::RowStride;
    constexpr unsigned Src1RowStride = Src1TileData::RowStride;

    unsigned src0ValidRow = src0.GetValidRow(), src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow(), src1ValidCol = src1.GetValidCol();
    unsigned dstValidRow  = dst.GetValidRow(),  dstValidCol  = dst.GetValidCol();

    if (src0ValidRow <= 0 || src0ValidCol <= 0 || src1ValidRow <= 0 || src1ValidCol <= 0 || dstValidRow <= 0 || dstValidCol <= 0)
        return;
    
    bool condSrc0EqDst = (src0ValidRow == dstValidRow && src0ValidCol == dstValidCol);
    bool condSrc1EqDst = (src1ValidRow == dstValidRow && src1ValidCol == dstValidCol);

    // dst has to be larger than or equal to both sources
    bool condDstgeSrc = (src1ValidRow <= dstValidRow && src1ValidCol <= dstValidCol) &&
                        (src0ValidRow <= dstValidRow && src0ValidCol <= dstValidCol);
                   
                        
    if (condSrc0EqDst && condSrc1EqDst) { // src0 == src1 == dst
        TBinOper<TPartMaxOp<typename DstTileData::DType>, DstTileData, elementsPerRepeat>(dst.data(), src0.data(), src1.data(), dstValidRow, dstValidCol);
    } else if (condDstgeSrc){             // src0 <= dst && src1 <= dst
        TCopyPadOp<TPartMaxOp<typename DstTileData::DType>, DstTileData, elementsPerRepeat, Src0RowStride, Src1RowStride, DstRowStride>
            (dst.data(), src0.data(), src1.data(), src0ValidRow, src0ValidCol, 
             src1ValidRow, src1ValidCol, dstValidRow, dstValidCol);
    }  // other conditions not supported
}
}  // namespace pto
#endif