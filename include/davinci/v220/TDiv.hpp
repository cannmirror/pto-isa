/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIV_HPP
#define TDIV_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ __aicore__ void TDiv(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::TileDType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *src0Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileData::DType *src1Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src1);
        
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;
        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (uint32_t i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (uint32_t j = 0; j < numLoop; j++) {
                        vdiv(dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src0Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src1Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                    }
                }
                if (remainAfterLoop) {
                    vdiv(dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         src0Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         src1Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         remainAfterLoop, 1, 1, 1, 8, 8, 8);
                }   
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        src0Ptr += numRepeatPerLine * elementsPerRepeat;
        src1Ptr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine) {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop) {
                for (uint32_t i = 0; i < numLoop; i++) {
                    if (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            vdiv(dstPtr + i * REPEAT_MAX * stride + j * stride,
                                 src0Ptr + i * REPEAT_MAX * stride + j * stride,
                                 src1Ptr + i * REPEAT_MAX * stride + j * stride,
                                 1, 1, 1, 1, 1, 1, 1);
                        }
                    } else {
                        vdiv(dstPtr + i * REPEAT_MAX * stride,
                             src0Ptr + i * REPEAT_MAX * stride,
                             src1Ptr + i * REPEAT_MAX * stride,
                             REPEAT_MAX, 1, 1, 1, stride / blockSizeElem, stride / blockSizeElem, stride / blockSizeElem);
                    }
                }
            }
            if (remainAfterLoop) {
                if (strideOverFlag) {
                    for (uint64_t j = 0; j < remainAfterLoop; j++) {
                        vdiv(dstPtr + numLoop * REPEAT_MAX * stride + j * stride,
                             src0Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                             src1Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                             1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    vdiv(dstPtr + numLoop * REPEAT_MAX * stride,
                         src0Ptr + numLoop * REPEAT_MAX * stride,
                         src1Ptr + numLoop * REPEAT_MAX * stride,
                         remainAfterLoop, 1, 1, 1, stride / blockSizeElem, stride / blockSizeElem, stride / blockSizeElem);
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileData>
    __aicore__ void TDIV_IMPL(TileData &dst, TileData &src0, TileData &src1) {
        static_assert(std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TDIV: Invalid data type");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();

        TDiv<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), src1.data(),
                                                                 validRow, validCol);
    }
}
#endif