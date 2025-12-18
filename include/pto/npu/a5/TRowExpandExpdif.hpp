/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPANDEXPDIF_HPP
#define TROWEXPANDEXPDIF_HPP


#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TRowExpandBinOp.hpp"

using namespace pto;
using namespace std;


namespace pto {
    
    template <typename T> struct RowExpandExpdifOp {
        PTO_INTERNAL static void RowExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1, MaskReg &preg)
        {
            if constexpr (std::is_same_v<T, float>) {
                vexpdif(reg_dst, reg_src0, reg_src1, preg, PART_ODD);
            }
            else {
                vsub(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
                vexp(reg_dst, reg_dst, preg, MODE_ZEROING);
            }
        }
    };

    template <typename TileDataDst, typename TileDataSrc1, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ AICORE void TRowExpandExpdif(typename TileDataDst::TileDType __out__ dst, 
                                typename TileDataDst::TileDType __in__ src0,
                                typename TileDataSrc1::TileDType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
        using T = typename TileDataDst::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

        RowExpandBinaryInstr<RowExpandExpdifOp<T>, TileDataDst, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(
                                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }

    template <typename TileDataDst, typename TileDataSrc1>
    PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1) {
        static_assert(std::is_same_v<typename TileDataDst::DType, typename TileDataSrc1::DType>,
                  "TROWEXPANDEXPDIF: src and dst data type is different!");
        static_assert(std::is_same<typename TileDataDst::DType, float>::value ||
                      std::is_same<typename TileDataDst::DType, half>::value,
                      "TROWEXPANDEXPDIF: Invalid data type.");
        static_assert(TileDataDst::isRowMajor && !TileDataSrc1::isRowMajor && TileDataSrc1::Cols == 1,
                  "TROWEXPANDEXPDIF: Invalid tile shape.");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType); 
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType); 
        constexpr unsigned rowStride = TileDataDst::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();

        TRowExpandExpdif<TileDataDst, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), src1.data(), validRow, validCol);
    }
}
#endif