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

#define PTO_CEIL(x,y)        ((((x)+(y)-1)/(y)) * (y))
#define PTO_DIV_ROUNDUP(x,y) ((((x)+(y)-1)/(y)))

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
        unsigned validRow,
        unsigned validCol,
        unsigned rowStride)
    {
        TILE_PTRS(dst, selmask, src0, src1);
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        uint16_t gmStride = PTO_CEIL(PTO_DIV_ROUNDUP(validCol, 8), 32);
        constexpr uint32_t unRollConstant = 2;
        uint16_t pairedRepeatTimes = repeatTimes / unRollConstant;
        uint16_t remainRepeat = repeatTimes % unRollConstant;

        __VEC_SCOPE__
        {
            MaskReg preg, selMask0, selMask1, selMask2, tmpMask0;
            MaskReg tmpMask1 = pset_b16(PAT_ALL);
            RegTensor<T> vreg0, vreg1, vreg2, vreg3, dreg0, dreg1, vreg4, vreg5, dreg2;

            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

            for( uint16_t i = 0; i < (uint16_t)(validRow); ++i){
                for (uint16_t j = 0; j < (uint16_t)(pairedRepeatTimes); ++j){
                    uint16_t repeatIdx = j * unRollConstant;
                    uint32_t colOffset0 = repeatIdx * elementsPerRepeat;
                    uint32_t colOffset1 = colOffset0 + elementsPerRepeat;

                    vlds(vreg0, src0Ptr, (int32_t)(i * rowStride + colOffset0), NORM);
                    vlds(vreg1, src1Ptr, (int32_t)(i * rowStride + colOffset0), NORM);
                    plds(tmpMask0, (__ubuf__ uint32_t *)maskPtr,  i * gmStride + repeatIdx * 8, US);
                    pintlv_b16(selMask0, selMask1, tmpMask0, tmpMask1);

                    vsel(dreg0, vreg0, vreg1, selMask0);

                    uint32_t count0 = ((colOffset0 + elementsPerRepeat) >= validCol ? validCol - colOffset0 : elementsPerRepeat);
                    preg = CreatePredicate<T>(count0);

                    vsts(dreg0, dstPtr, (int32_t)(i * rowStride + colOffset0), distValue, preg);

                    vlds(vreg2, src0Ptr, (int32_t)(i * rowStride + colOffset1), NORM);
                    vlds(vreg3, src1Ptr, (int32_t)(i * rowStride + colOffset1), NORM);
                    vsel(dreg1, vreg2, vreg3, selMask1);
                    uint32_t count1 = ((colOffset1 + elementsPerRepeat) >= validCol ? validCol - colOffset1 : elementsPerRepeat);
                    preg = CreatePredicate<T>(count1);
                    vsts(dreg1, dstPtr, (int32_t)(i * rowStride + colOffset1), distValue, preg);

                }
            }

            if (remainRepeat != 0) {
                for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
                    uint16_t repeatIdx = pairedRepeatTimes * unRollConstant;
                    uint32_t colOffset = repeatIdx * elementsPerRepeat;
                    uint32_t count = (validCol > colOffset) ? (validCol - colOffset) : 0;
                    preg = CreatePredicate<T>(count);

                    plds(selMask2, (__ubuf__ uint32_t *)maskPtr,  repeatIdx * 8 + i * gmStride, US);
                    punpack(selMask2, selMask2, LOWER);

                    vlds(vreg4, src0Ptr, (int32_t)(colOffset + i * rowStride), NORM);
                    vlds(vreg5, src1Ptr, (int32_t)(colOffset + i * rowStride), NORM);
                    vsel(dreg2, vreg4, vreg5, selMask2);
                    vsts(dreg2, dstPtr, (int32_t)(colOffset + i * rowStride), distValue, preg);
                }
            }
        }
    }

    template <typename TileData, typename MaskTile, unsigned elementsPerRepeat>
    __tf__ PTO_INTERNAL void TSel_b16_8(
        typename TileData::TileDType __out__ dst,
        typename MaskTile::TileDType __in__ selmask,
        typename TileData::TileDType __in__ src0,
        typename TileData::TileDType __in__ src1,
        unsigned validRow,
        unsigned validCol,
        unsigned rowStride)
    {
        TILE_PTRS(dst, selmask, src0, src1);
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        uint16_t gmStride = PTO_CEIL(PTO_DIV_ROUNDUP(validCol, 8), 32);
        __VEC_SCOPE__
        {
            MaskReg preg, maskreg;
            RegTensor<T> vreg0, vreg1, vreg2;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < validRow; ++i)
            {
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j)
                {
                    vlds(vreg0, src0Ptr, i * rowStride + j * elementsPerRepeat, NORM);
                    vlds(vreg1, src1Ptr, i * rowStride + j * elementsPerRepeat, NORM);
                    if (sizeof(T) == 2)
                    {
                        plds(maskreg, (__ubuf__ uint32_t *)maskPtr, i * gmStride + j * 16, US);
                    }
                    else
                    {
                        plds(maskreg, (__ubuf__ uint32_t *)maskPtr, i * gmStride + j * 16, NORM);
                    }
                    uint32_t count = ((j + 1) * elementsPerRepeat >= validCol ? validCol - j * elementsPerRepeat : elementsPerRepeat);
                    preg = CreatePredicate<T>(count);
                    vsel(vreg2, vreg0, vreg1, maskreg);
                    vsts(vreg2, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg);
                }
            }
        }
    }

    template <typename TileData, typename MaskTile>
    PTO_INTERNAL void TSEL_IMPL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1)
    {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        constexpr unsigned rowStride = TileData::RowStride;
        if (sizeof(typename TileData::DType) == 4)
        {
            TSel_b32<TileData, MaskTile, elementsPerRepeat>(dst.data(), selMask.data(), src0.data(), src1.data(), validRow, validCol, rowStride);
        }
        else
        {
            TSel_b16_8<TileData, MaskTile, elementsPerRepeat>(dst.data(), selMask.data(), src0.data(), src1.data(), validRow, validCol, rowStride);
        }
    }
}
#endif