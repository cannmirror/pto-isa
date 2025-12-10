/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSEL_HPP
#define TSEL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto
{
    #define TILE_PTRS(dst, selmask, src0, src1) \
        using T = typename TileData::DType; \
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst); \
        __ubuf__ typename MaskTile::DType *maskPtr = (__ubuf__ typename MaskTile::DType *)__cce_get_tile_ptr(selmask);  \
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);   \
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1)

    template <typename TileData, typename MaskTile, unsigned elementsPerRepeat>
    __tf__ PTO_INTERNAL void TSel_b32(
        typename TileData::TileDType __out__ dst,
        typename MaskTile::TileDType __in__ selmask,
        typename TileData::TileDType __in__ src0,
        typename TileData::TileDType __in__ src1,
        unsigned validRow)
    {
        TILE_PTRS(dst, selmask, src0, src1);
        __VEC_SCOPE__
        {
            MaskReg preg, selMask0, selMask1, selMask2, tmpMask0;
            MaskReg tmpMask1 = pset_b16(PAT_ALL);
            RegTensor<T> vreg0, vreg1, vreg2, vreg3, dreg0, dreg1, vreg4, vreg5, dreg2;
            constexpr uint32_t unRollConstant = 2;
            uint32_t sreg = (uint32_t)(validRow * TileData::Cols);
            constexpr uint32_t selOffset = VECTOR_REG_WIDTH / 8 / sizeof(T) / 2;
            uint16_t repeatTimes = CeilDivision(validRow * TileData::Cols, elementsPerRepeat);
            uint16_t newRepeatTimes = repeatTimes / unRollConstant;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

            for (uint16_t i = 0; i < (uint16_t)newRepeatTimes; ++i)
            {
                plds(tmpMask0, (__ubuf__ uint32_t *)maskPtr + i * selOffset, 0, US);
                pintlv_b16(selMask0, selMask1, tmpMask0, tmpMask1);
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg1, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vsel(dreg0, vreg0, vreg1, selMask0);
                vsts(dreg0, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);

                vlds(vreg2, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg3, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vsel(dreg1, vreg2, vreg3, selMask1);
                vsts(dreg1, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
            }
            for (uint16_t i = 0; i < (uint16_t)repeatTimes % unRollConstant; ++i)
            {
                plds(selMask2, (__ubuf__ uint32_t *)maskPtr + newRepeatTimes * selOffset, 0, US);
                punpack(selMask2, selMask2, LOWER);
                preg = CreatePredicate<T>(sreg);
                vlds(vreg4, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg5, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vsel(dreg2, vreg4, vreg5, selMask2);
                vsts(dreg2, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
            }
        }
    }

    template <typename TileData, typename MaskTile, unsigned elementsPerRepeat>
    __tf__ PTO_INTERNAL void TSel_b16_8(
        typename TileData::TileDType __out__ dst,
        typename MaskTile::TileDType __in__ selmask,
        typename TileData::TileDType __in__ src0,
        typename TileData::TileDType __in__ src1,
        unsigned validRow)
    {
        TILE_PTRS(dst, selmask, src0, src1);
        __VEC_SCOPE__
        {
            MaskReg preg, maskreg;
            RegTensor<T> vreg0, vreg1, vreg2;
            uint32_t sreg = (uint32_t)(validRow * TileData::Cols);
            constexpr uint32_t selOffset = VECTOR_REG_WIDTH / 8 / sizeof(T) / 4;
            uint16_t repeatTimes = CeilDivision(validRow * TileData::Cols, elementsPerRepeat);
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i)
            {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg1, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                if (sizeof(T) == 2)
                {
                    plds(maskreg, (__ubuf__ uint32_t *)maskPtr + i * selOffset, 0, US);
                }
                else
                {
                    plds(maskreg, (__ubuf__ uint32_t *)maskPtr + i * selOffset, 0, NORM);
                }
                vsel(vreg2, vreg0, vreg1, maskreg);
                vsts(vreg2, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
            }
        }
    }

    template <typename TileData, typename MaskTile>
    PTO_INTERNAL void TSEL_IMPL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1)
    {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned validRow = dst.GetValidRow();
        if (sizeof(typename TileData::DType) == 4)
        {
            TSel_b32<TileData, MaskTile, elementsPerRepeat>(dst.data(), selMask.data(), src0.data(), src1.data(), validRow);
        }
        else
        {
            TSel_b16_8<TileData, MaskTile, elementsPerRepeat>(dst.data(), selMask.data(), src0.data(), src1.data(), validRow);
        }
    }
}
#endif