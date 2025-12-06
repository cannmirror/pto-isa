/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSUB_HPP
#define TSUB_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ __aicore__ void TSub(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::TileDType __in__ src1,
                                unsigned numRepeatPerLine,
                                unsigned numRemainPerLine,
                                unsigned validRow)
    {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *src0Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileData::DType *src1Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src1);

        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vsub(dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src0Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src1Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                    }
                }
                if (remainAfterLoop) {
                    vsub(dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
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
                for (int i = 0; i < numLoop; i++) {
                    if (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            vsub(dstPtr + i * REPEAT_MAX * stride + j * stride,
                                 src0Ptr + i * REPEAT_MAX * stride + j * stride,
                                 src1Ptr + i * REPEAT_MAX * stride + j * stride,
                                 1, 1, 1, 1, 1, 1, 1);
                        }
                    } else {
                        vsub(dstPtr + i * REPEAT_MAX * stride,
                             src0Ptr + i * REPEAT_MAX * stride,
                             src1Ptr + i * REPEAT_MAX * stride,
                             REPEAT_MAX, 1, 1, 1, stride / blockSizeElem, stride / blockSizeElem, stride / blockSizeElem);
                    }
                }
            }
            if (remainAfterLoop) {
                if (strideOverFlag) {
                    for (unsigned j = 0; j < remainAfterLoop; j++) {
                        vsub(dstPtr + numLoop * REPEAT_MAX * stride + j * stride,
                             src0Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                             src1Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                             1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    vsub(dstPtr + numLoop * REPEAT_MAX * stride,
                         src0Ptr + numLoop * REPEAT_MAX * stride,
                         src1Ptr + numLoop * REPEAT_MAX * stride,
                         remainAfterLoop, 1, 1, 1, stride / blockSizeElem, stride / blockSizeElem, stride / blockSizeElem);
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileData>
    __aicore__ void TSUB_IMPL(TileData &dst, TileData &src0, TileData &src1) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();

        TSub<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), src1.data(),
                                                                numRepeatPerLine, numRemainPerLine, validRow);
    }
}
#endif