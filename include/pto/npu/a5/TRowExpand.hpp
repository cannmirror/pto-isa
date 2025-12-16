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
PTO_INTERNAL void TRowExpandCheck(unsigned srcValidRow, unsigned srcValidCol,unsigned dstValidRow) {
    static_assert((sizeof(typename TileDataIn::DType) == 1) || (sizeof(typename TileDataIn::DType) == 2) ||
                  (sizeof(typename TileDataIn::DType) == 4), "Data type must be b8/b16/b32");
    static_assert(TileDataIn::Loc == pto::TileType::Vec, "Src TileType must be Vec Tile!");
    static_assert(TileDataOut::Loc == pto::TileType::Vec, "Dst TileType must be Vec Tile!");
    static_assert(TileDataIn::isRowMajor && TileDataIn::SFractal == SLayout::NoneBox,
      "This instruction only support Nd fractal Tile");
    static_assert(TileDataOut::isRowMajor && TileDataOut::SFractal == SLayout::NoneBox,
      "This instruction only support Nd fractal Tile");
    static_assert(std::is_same_v<typename TileDataOut::DType, typename TileDataIn::DType>,
      "The input data type must be consistent with the output data type.");
    PTO_ASSERT(srcValidRow == dstValidRow, "The input valid row must be consistent with the output valid row.");
    PTO_ASSERT(srcValidRow != 0 && srcValidCol != 0, "The input shape is invalid, validCol or validRow is 0.");
}

template <typename T, unsigned DstStride, unsigned SrcStride>
PTO_INTERNAL void TRowExpandInstr_NoPostUpdate(__ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned dstValidRow,
                                               unsigned dstValidCol, uint16_t repeatTimes, uint16_t eleCntValue) {
    constexpr auto distValue = 
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        MaskReg preg;
        __ubuf__ T* dstOffset;
        uint32_t sreg = eleCntValue;
        MaskReg pg0 = CreatePredicate<T>(sreg);
        for (uint16_t i = 0; i < (uint16_t)dstValidRow; i++) {
            vlds(vreg0, srcPtr, i * SrcStride, NORM);
            vdup(vreg1, vreg0, pg0, POS_LOWEST, MODE_ZEROING);
            sreg = (uint32_t)(dstValidCol);
            dstOffset = dstPtr + i * DstStride;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                preg = CreatePredicate<T>(sreg);
                vsts(vreg1, dstOffset, (int32_t)(j * eleCntValue), distValue, preg);
            }
        }
    }
}

template <typename T, unsigned DstStride, unsigned SrcStride>
PTO_INTERNAL void TRowExpandInstr_PostUpdate(__ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned dstValidRow,
                                             unsigned dstValidCol, uint16_t repeatTimes, uint16_t eleCntValue) {
    constexpr auto distValue = 
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        MaskReg preg;
        uint32_t sreg = eleCntValue;
        MaskReg pg0 = CreatePredicate<T>(sreg);
        for (uint16_t i = 0; i < (uint16_t)dstValidRow; i++) {
            vlds(vreg0, srcPtr, SrcStride, NORM, POST_UPDATE);
            vdup(vreg1, vreg0, pg0, POS_LOWEST, MODE_ZEROING);
            sreg = (uint32_t)(dstValidCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                preg = CreatePredicate<T>(sreg);
                vsts(vreg1, dstPtr, (int32_t)(eleCntValue), distValue, preg, POST_UPDATE);
            }
            dstPtr += (DstStride - repeatTimes * eleCntValue);
        }
    }
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TROWEXPAND) OP_TYPE(broadcast)
void TRowExpand(typename TileDataOut::TileDType __out__ dst,
                                  typename TileDataIn::TileDType __in__ src,
                                  unsigned dstValidRow,
                                  unsigned dstValidCol,
                                  unsigned version = VFImplKind::VFIMPL_DEFAULT) {
    using T = typename TileDataOut::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint16_t eleCntValue = elementsPerRepeat;
    uint16_t repeatTimes = CeilDivision(dstValidCol, eleCntValue);

    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            TRowExpandInstr_NoPostUpdate<T, TileDataOut::Cols, TileDataIn::Cols>(
                dstPtr, srcPtr, dstValidRow, dstValidCol, repeatTimes, eleCntValue);
            break;
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            TRowExpandInstr_PostUpdate<T, TileDataOut::Cols, TileDataIn::Cols>(
                dstPtr, srcPtr, dstValidRow, dstValidCol, repeatTimes, eleCntValue);
            break;
        default:
            TRowExpandInstr_PostUpdate<T, TileDataOut::Cols, TileDataIn::Cols>(
                dstPtr, srcPtr, dstValidRow, dstValidCol, repeatTimes, eleCntValue);
            break;
    }
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TROWEXPAND_IMPL(TileDataOut &dst, TileDataIn &src)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataIn::DType);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataIn::DType);
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    TRowExpandCheck<TileDataOut, TileDataIn>(src.GetValidRow(), src.GetValidCol(), dstValidRow);
    TRowExpand<TileDataOut, TileDataIn, elementsPerRepeat, blockSizeElem>(
        dst.data(), src.data(), dstValidRow, dstValidCol);
}
}  // namespace pto
#endif
