/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIVS_HPP
#define TDIVS_HPP

#include <pto/common/constants.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"

using namespace pto;
using namespace std;

namespace pto {

    template <typename T> struct DivSOp {
        PTO_INTERNAL static void BinSInstr(RegTensor<T> &vregdst, RegTensor<T> &vregsrc, T src1, MaskReg &preg)
        {
            float divider = static_cast<float>(src1);
            if (divider != 0.0f)
            {
                divider = 1.0f / divider;
            }
            else
            {
                divider = 1.0 / 0.0;
            }
            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
            {
                vmuls(vregdst, vregsrc, divider, preg);
            }
            else if constexpr (std::is_same<T, int32_t>::value)
            {
                RegTensor<float> tempDst;
                vcvt(tempDst, vregsrc, preg, RoundRType());
                vmuls(tempDst, tempDst, divider, preg);
                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
            }
            else if constexpr (std::is_same<T, int16_t>::value)
            {
                RegTensor<half> tempDst;
                vcvt(tempDst, vregsrc, preg, RoundRType());
                vmuls(tempDst, tempDst, divider, preg);
                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
            }
        }
    };

        template <typename T> struct DivSOpS {
        PTO_INTERNAL static void BinSInstr(RegTensor<T> &vregdst, RegTensor<T> &vregsrc, T src0, MaskReg &preg)
        {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
            {
                vdup(vregdst, src0, preg, MODE_ZEROING);
                vdiv(vregdst, vregdst, vregsrc, preg);
            }
            else if constexpr (std::is_same<T, int32_t>::value)
            {
                RegTensor<float> tempDst;
                RegTensor<float> tempSrc;
                vdup(vregdst, src0, preg, MODE_ZEROING);
                vcvt(tempDst, vregdst, preg, RoundRType());
                vcvt(tempSrc, vregsrc, preg, RoundRType());
                vdiv(tempDst, tempDst, tempSrc, preg);
                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
            }
            else if constexpr (std::is_same<T, int16_t>::value)
            {
                RegTensor<half> tempDst;
                RegTensor<half> tempSrc;
                vdup(vregdst, src0, preg, MODE_ZEROING);
                vcvt(tempDst, vregdst, preg, RoundRType());
                vcvt(tempSrc, vregsrc, preg, RoundRType());
                vdiv(tempDst, tempDst, tempSrc, preg);
                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
            }
        }
    };

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ PTO_INTERNAL OP_NAME(TDIVS) OP_TYPE(element_wise)
    void TDivS(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0, 
                                typename TileData::DType __in__ src1, 
                                unsigned validRow,
                                unsigned validCol,
                                BinSOpsImpl version = BinSOpsImpl::BinSOpsIMPL_DEFAULT) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        BinaryInstr<DivSOp<T>, TileData, T, elementsPerRepeat, blockSizeElem, rowStride>(
                    dstPtr, src0Ptr, src1, validRow, validCol, version);
    }

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ PTO_INTERNAL OP_NAME(TDIVS) OP_TYPE(element_wise)
    void TDivS(typename TileData::TileDType __out__ dst,
                                typename TileData::DType __in__ src1, 
                                typename TileData::TileDType __in__ src0, 
                                unsigned validRow,
                                unsigned validCol,
                                BinSOpsImpl version = BinSOpsImpl::BinSOpsIMPL_DEFAULT) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        BinaryInstr<DivSOpS<T>, TileData, T, elementsPerRepeat, blockSizeElem, rowStride>(
                    dstPtr, src0Ptr, src1, validRow, validCol, version);
    }

    template <typename TileData>
    AICORE void TDIVS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TDivS<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), scalar, validRow, validCol);
    }

    template <typename TileData>
    AICORE void TDIVS_IMPL(TileData &dst, typename TileData::DType scalar, TileData &src0)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TDivS<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), scalar, src0.data(), validRow, validCol);
    }
}
#endif
