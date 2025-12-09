/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINS_HPP
#define TBINS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {

enum class BinSOpsImpl : uint8_t {
  BinSOpsIMPL_DEFAULT = 0,
  BinSOpsIMPL_1D_NO_POST_UPDATE = 1,
  BinSOpsIMPL_2D_NO_POST_UPDATE = 2,
  BinSOpsIMPL_1D_POST_UPDATE = 3,
  BinSOpsIMPL_2D_POST_UPDATE = 4,
};

template <typename Op, typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TBinSOps_1D_NoPostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *src0Ptr, 
                    ScalarType src1,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidRows * kValidCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg2;
        MaskReg  preg;

        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = kValidRows * kValidCols;
        #pragma clang loop unroll(disable)
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, i*elementsPerRepeat, NORM);
            Op::BinSInstr(vreg2, vreg0, src1, preg);
            vsts(vreg2, dstPtr, i*elementsPerRepeat, distValue, preg);
        }
    }
}

template <typename Op, typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TBinSOps_1D_PostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *src0Ptr, 
                    ScalarType src1,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes_pu = CeilDivision(kValidRows * kValidCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_pu;
        RegTensor<T> vreg2_pu;
        MaskReg  preg_pu;

        constexpr auto distValue_pu = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg_pu = kValidRows * kValidCols;
        #pragma clang loop unroll(disable)
        for (uint16_t i = 0; i < (uint16_t)repeatTimes_pu; ++i) {
            preg_pu = CreatePredicate<T>(sreg_pu);
            vlds(vreg0_pu, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
            Op::BinSInstr(vreg2_pu, vreg0_pu, src1, preg_pu);
            vsts(vreg2_pu, dstPtr, elementsPerRepeat, distValue_pu, preg_pu, POST_UPDATE);
        }
    } 
}

template <typename Op, typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TBinSOps_2D_NoPostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *src0Ptr, 
                    ScalarType src1,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg2;
        MaskReg preg;
        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
            uint32_t sreg = (uint32_t)(kValidCols);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr,  i * rowStride + j * elementsPerRepeat, NORM);
                Op::BinSInstr(vreg2, vreg0, src1, preg);
                vsts(vreg2, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename Op, typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TBinSOps_2D_PostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *src0Ptr, 
                    ScalarType src1,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes_pu = CeilDivision(kValidCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_pu;
        RegTensor<T> vreg2_pu;
        MaskReg preg_pu;
        constexpr auto distValue_pu = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)repeatTimes_pu; ++j) {
                vlds(vreg0_pu, src0Ptr,  i * rowStride + j * elementsPerRepeat, NORM);
                uint32_t count_pu = ((j + 1) * elementsPerRepeat >= kValidCols ? kValidCols - j * elementsPerRepeat : elementsPerRepeat);
                preg_pu = CreatePredicate<T>(count_pu);
                Op::BinSInstr(vreg2_pu, vreg0_pu, src1, preg_pu);
                vsts(vreg2_pu, dstPtr, i * rowStride + j * elementsPerRepeat, distValue_pu, preg_pu);
            }
        }
    }
}
template <typename Op, typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__aicore__ PTO_INLINE void TBinSOps_1D_selector(typename TileData::TileDType __out__ dst, 
                            typename TileData::TileDType __in__ src0, 
                            ScalarType src1,
                            unsigned kValidRows,
                            unsigned kValidCols,
                            BinSOpsImpl version) {
        switch (version) {
        case BinSOpsImpl::BinSOpsIMPL_1D_NO_POST_UPDATE:
        case BinSOpsImpl::BinSOpsIMPL_2D_NO_POST_UPDATE:
            TBinSOps_1D_NoPostUpdate<Op, TileData, ScalarType, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, kValidRows, kValidCols);
            break;
        case BinSOpsImpl::BinSOpsIMPL_DEFAULT:
        case BinSOpsImpl::BinSOpsIMPL_1D_POST_UPDATE:
        case BinSOpsImpl::BinSOpsIMPL_2D_POST_UPDATE:
        default:
            TBinSOps_1D_PostUpdate<Op, TileData, ScalarType, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, kValidRows, kValidCols);
            break;
        }
} 
template <typename Op, typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstr(typename TileData::TileDType __out__ dst,
                            typename TileData::TileDType __in__ src0, 
                            ScalarType src1,
                            unsigned kValidRows,
                            unsigned kValidCols,
                            BinSOpsImpl version) {
    using T = typename TileData::DType;
    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint16_t> || 
        std::is_same_v<T, int16_t> || std::is_same_v<T, uint32_t> ||
        std::is_same_v<T, int32_t> || std::is_same_v<T, half> ||
        std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>,  "TBinSOps: Invalid data type."); 
        if constexpr (TileData::ValidCol == TileData::Cols) {
            TBinSOps_1D_selector<Op, TileData, ScalarType, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, kValidRows, kValidCols, version);
        } else {
            if (TileData::Cols == kValidCols) {
                TBinSOps_1D_selector<Op, TileData, ScalarType, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, kValidRows, kValidCols, version);  
            }
            else {
                switch (version) {
                case BinSOpsImpl::BinSOpsIMPL_1D_NO_POST_UPDATE:
                case BinSOpsImpl::BinSOpsIMPL_2D_NO_POST_UPDATE:
                    TBinSOps_2D_NoPostUpdate<Op, TileData, ScalarType, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, kValidRows, kValidCols);
                    break;
                case BinSOpsImpl::BinSOpsIMPL_DEFAULT:
                case BinSOpsImpl::BinSOpsIMPL_1D_POST_UPDATE:
                case BinSOpsImpl::BinSOpsIMPL_2D_POST_UPDATE:
                default:
                    TBinSOps_2D_PostUpdate<Op, TileData, ScalarType, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, kValidRows, kValidCols);
                    break;
                }  
            }
        }
    }
}  // namespace pto
#endif
