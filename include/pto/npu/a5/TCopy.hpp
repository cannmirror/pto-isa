/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOPY_HPP
#define TCOPY_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/common.hpp"
#include "utils.hpp"

namespace pto {

    template <typename TileDataDst, typename TileDataSrc, TCopyMode copyMode, 
             unsigned blockSizeElem, unsigned srcStride, unsigned dstStride>
    __tf__ __aicore__ PTO_INLINE void TCopy(typename TileDataDst::TileDType __out__ dst,
                                            typename TileDataSrc::TileDType __in__ src,
                                            uint64_t validRow,
                                            uint64_t validCol){
        if (validRow == 0 || validCol == 0) {
            return;
        }                                    
        using T = typename TileDataSrc::DType;
        using U = typename TileDataDst::DType;
        __ubuf__ T * srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ U * dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataSrc::DType); 
        if constexpr(copyMode == pto::TCopyMode::SHALLOW_COPY){ // shallow copy
            (__ubuf__ U*)dstPtr = (__ubuf__ T*)srcPtr;
        }else{ // deep copy
            static_assert(sizeof(T) == sizeof(U), "TCOPY: src and dst data type is different!");
            __VEC_SCOPE__
            {
                RegTensor<T> vreg0;
                MaskReg preg;
                uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
                constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {                
                    uint32_t sreg = (uint32_t)(validCol);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vreg0, srcPtr + i * srcStride, j * elementsPerRepeat, NORM);
                        vsts(vreg0, dstPtr + i * dstStride, j * elementsPerRepeat, distValue, preg);
                    }
                }
            } // end VF
        } // end of deep copy
    } // end of tf

    template <typename TileDataDst, typename TileDataSrc, TCopyMode copyMode>
    __aicore__ PTO_INLINE void TCOPY_IMPL(TileDataDst &dst, TileDataSrc &src) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);
        constexpr unsigned srcStride = TileDataSrc::RowStride;
        constexpr unsigned dstStride = TileDataDst::RowStride;
        uint64_t validSrcRow = src.GetValidRow();
        uint64_t validSrcCol = src.GetValidCol();
        uint64_t validDstRow = dst.GetValidRow();
        uint64_t validDstCol = dst.GetValidCol();

        uint64_t validRow = (validSrcRow <= validDstRow) ? validSrcRow : validDstRow;
        uint64_t validCol = (validSrcCol <= validDstCol) ? validSrcCol : validDstCol;

        TCopy<TileDataDst, TileDataSrc, copyMode, blockSizeElem, srcStride, dstStride>
             (dst.data(), src.data(), validRow, validCol);
    }
}
#endif
