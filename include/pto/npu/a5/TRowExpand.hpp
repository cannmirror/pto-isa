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
#include "pto/common.hpp"
#include "utils.hpp"


namespace pto {

template <typename TileDataOut, typename TileDataIn>
__tf__ __aicore__ PTO_INLINE void TRowExpand(typename TileDataOut::TileDType __out__ dst, 
                                  typename TileDataIn::TileDType __in__ src,
                                  unsigned kValidRows,
                                  unsigned kValidCols,
                                  uint32_t eleCntReg,
                                  uint32_t dstCols,
                                  uint32_t srcCols){
    using T = typename TileDataOut::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    uint32_t remainEleNum = kValidCols % eleCntReg ?: eleCntReg;
    uint16_t repeatTimes = CeilDivision(kValidCols, eleCntReg);


    if constexpr (sizeof(T) == 1) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;
            MaskReg pg0 = CreatePredicate<T>(eleCntReg);
            MaskReg pg1 = CreatePredicate<T>(remainEleNum);
            for (uint16_t i = 0; i < (uint16_t)kValidRows; i++) {
                vlds(vreg0, srcPtr + i * srcCols, (int32_t)0, NORM);
                vdup(vreg1, vreg0, pg0, POS_LOWEST, MODE_ZEROING);
                for (uint16_t j = 0; j < (uint16_t)(repeatTimes - 1); j++) {
                    vsts(vreg1, dst + i * dstCols, (int32_t)(j * ELE_CNT_B8), NORM_B8, pg0);
                }
                vsts(vreg1, dst + i * dstCols, (int32_t)((repeatTimes - 1) * ELE_CNT_B8), NORM_B8, pg1);
            }
        }
    } else if constexpr (sizeof(T) == 2) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;
            MaskReg pg0 = CreatePredicate<T>(eleCntReg);
            MaskReg pg1 = CreatePredicate<T>(remainEleNum);
            for (uint16_t i = 0; i < (uint16_t)kValidRows; i++) {
                vlds(vreg0, srcPtr + i * srcCols, (int32_t)0, NORM);
                vdup(vreg1, vreg0, pg0, POS_LOWEST, MODE_ZEROING);
                for (uint16_t j = 0; j < (uint16_t)(repeatTimes - 1); j++) {
                    vsts(vreg1, dst + i * dstCols, (int32_t)(j * ELE_CNT_B16), NORM_B16, pg0);
                }
                vsts(vreg1, dst + i * dstCols, (int32_t)((repeatTimes - 1) * ELE_CNT_B16), NORM_B16, pg1);
            }
        }
    } else if constexpr (sizeof(T) == 4) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;
            MaskReg pg0 = CreatePredicate<T>(eleCntReg);
            MaskReg pg1 = CreatePredicate<T>(remainEleNum);
            for (uint16_t i = 0; i < (uint16_t)kValidRows; i++) {
                vlds(vreg0, srcPtr + i * srcCols, (int32_t)0, NORM);
                vdup(vreg1, vreg0, pg0, POS_LOWEST, MODE_ZEROING);
                for (uint16_t j = 0; j < (uint16_t)(repeatTimes - 1); j++) {
                    vsts(vreg1, dst + i * dstCols, (int32_t)(j * ELE_CNT_B32), NORM_B32, pg0);
                }
                vsts(vreg1, dst + i * dstCols, (int32_t)((repeatTimes - 1) * ELE_CNT_B32), NORM_B32, pg1);
            }
        }
    }
}

template <typename TileDataOut, typename TileDataIn>
__aicore__ PTO_INLINE void TROWEXPAND_IMPL(TileDataOut &dst, TileDataIn &src)
{   
    static_assert((sizeof(typename TileDataIn::DType) == 1) || (sizeof(typename TileDataIn::DType) == 2) ||
                    (sizeof(typename TileDataIn::DType) == 4), "Data type must be b8/b16/b32");
    static_assert(TileDataIn::Loc == pto::Location::Vec, "Src location must be Vec!");
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
