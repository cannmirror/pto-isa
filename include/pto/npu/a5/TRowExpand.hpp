/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPAND_HPP
#define TROWEXPAND_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

template <typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL void TRowExpand(typename TileDataOut::TileDType __out__ dst,
                                  typename TileDataIn::TileDType __in__ src,
                                  unsigned kValidRows,
                                  unsigned kValidCols,
                                  uint32_t eleCntReg,
                                  uint32_t dstCols,
                                  uint32_t srcCols){
    using T = typename TileDataOut::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    uint16_t repeatTimes = CeilDivision(kValidCols, eleCntReg);
    constexpr auto eleCntValue = CCE_VL /sizeof(T);
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        MaskReg pg0 = CreatePredicate<T>(eleCntReg);
        MaskReg preg;
        uint32_t sreg;
        for (uint16_t i = 0; i < (uint16_t)kValidRows; i++) {
            vlds(vreg0, srcPtr, i * srcCols, NORM);
            vdup(vreg1, vreg0, pg0, POS_LOWEST, MODE_ZEROING);
            sreg = (uint32_t)(kValidCols);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                preg = CreatePredicate<T>(sreg);
                vsts(vreg1, dst, (int32_t)(j * eleCntValue + i * dstCols), distValue, preg);
            }
        }
    }
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TROWEXPAND_IMPL(TileDataOut &dst, TileDataIn &src)
{   
    static_assert((sizeof(typename TileDataIn::DType) == 1) || (sizeof(typename TileDataIn::DType) == 2) ||
                    (sizeof(typename TileDataIn::DType) == 4), "Data type must be b8/b16/b32");
    static_assert(TileDataIn::Loc == pto::TileType::Vec, "Src TileType must be Vec!");
    static_assert(((TileDataOut::isRowMajor && (TileDataOut::SFractal == SLayout::NoneBox)) &&
                    (TileDataIn::isRowMajor && (TileDataIn::SFractal == SLayout::NoneBox))),
                    "Src and dst layout must be ND!");

    unsigned kValidCols = dst.GetValidCol();
    unsigned kValidRows = dst.GetValidRow();
    uint32_t eleCntReg = CCE_VL / sizeof(typename TileDataIn::DType);
    uint32_t dstCols = TileDataOut::Cols;
    uint32_t srcCols = TileDataIn::Cols;
    
    TRowExpand<TileDataOut, TileDataIn>(dst.data(), src.data(), kValidRows, kValidCols, eleCntReg, dstCols, srcCols);
}
}  // namespace pto
#endif
