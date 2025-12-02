/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTIALMIN_HPP
#define TPARTIALMIN_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
    __aicore__ PTO_INLINE void TPARTMIN_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
    {
        static_assert(std::is_same<typename TileDataDst::DType, typename TileDataSrc0::DType>::value &&
                      std::is_same<typename TileDataDst::DType, typename TileDataSrc1::DType>::value,
                      "TPARTMIN: src and dst data type is different!");
        static_assert((std::is_same<typename TileDataDst::DType, int32_t>::value) ||
                      (std::is_same<typename TileDataDst::DType, int>::value) ||
                      (std::is_same<typename TileDataDst::DType, int16_t>::value) ||
                      (std::is_same<typename TileDataDst::DType, half>::value) ||
                      (std::is_same<typename TileDataDst::DType, float16_t>::value) ||
                      (std::is_same<typename TileDataDst::DType, float>::value) ||
                      (std::is_same<typename TileDataDst::DType, float32_t>::value),
                      "TPARTMIN: Invalid data type.");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        unsigned src0ValidRow = src0.GetValidRow();
        unsigned src0ValidCol = src0.GetValidCol();
        unsigned src1ValidRow = src1.GetValidRow();
        unsigned src1ValidCol = src1.GetValidCol();
        unsigned dstValidRow = dst.GetValidRow();
        unsigned dstValidCol = dst.GetValidCol();
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
        constexpr unsigned src1RowStride = TileDataSrc1::RowStride;

        TPartMaxMin<false, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride,
                 src1RowStride>(dst.data(),
            src0.data(),
            src1.data(),
            src0ValidRow,
            src0ValidCol,
            src1ValidRow,
            src1ValidCol,
            dstValidRow,
            dstValidCol);
    }
}

#endif
