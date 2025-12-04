/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLSUM_HPP
#define TCOLSUM_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template <typename T, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
              unsigned srcstride, bool IsBinary>
    __tf__ __aicore__ PTO_INLINE void TColSum(typename TileDataDst::TileDType __out__ dst,
                                              typename TileDataSrc::TileDType __in__ src,
                                              typename TileDataTmp::TileDType __in__ tmp,
                                              unsigned validRow, unsigned validCol) {
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

        constexpr unsigned DTypeSize = sizeof(T);
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / DTypeSize;
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / DTypeSize;
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;
        unsigned numBlockPerLine = (srcstride * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        unsigned dupSrcStride = numBlockPerLine * blockSizeElem;
        unsigned elementsPerLine = numRepeatPerLine * elementsPerRepeat;
        unsigned lenBurst = (validCol * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

        if (validRow == 1) {
            copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, lenBurst, 0, 0);
            pipe_barrier(PIPE_V);
            return;
        }

        if (IsBinary) {
            if (numRepeatPerLine > 0) {
                set_mask_count();
                set_vector_mask(0, elementsPerLine);
                for (uint32_t i = 0; i < validRow / 2; i++) {
                    vadd(tmpPtr + i * dupSrcStride, srcPtr + 2 * i * dupSrcStride,
                         srcPtr + (2 * i + 1) * dupSrcStride, 0, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }

                if (validRow % 2 == 1) {
                    vadd(tmpPtr, tmpPtr, srcPtr + (validRow - 1) * dupSrcStride,
                         0, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                set_mask_norm();
                set_vector_mask(-1, -1);
            }

            if (numRemainPerLine) {
                SetContinuousMask(numRemainPerLine);
                for (int i = 0; i < validRow / 2; i++) {
                    vadd(tmpPtr + elementsPerLine + i * dupSrcStride, srcPtr + elementsPerLine + 2 * i * dupSrcStride,
                         srcPtr + elementsPerLine + (2 * i + 1) * dupSrcStride, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }

                if (validRow % 2 == 1) {
                    vadd(tmpPtr + elementsPerLine, tmpPtr + elementsPerLine,
                         srcPtr + elementsPerLine + (validRow - 1) * dupSrcStride, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                set_vector_mask(-1, -1);
            }

            unsigned cnt = validRow / 2;
            while (cnt > 1)
            {
                if (numRepeatPerLine > 0) {
                    set_mask_count();
                    set_vector_mask(0, elementsPerLine);
                    for (uint32_t i = 0; i < cnt / 2; i++) {
                        vadd(tmpPtr + i * dupSrcStride, tmpPtr + 2 * i * dupSrcStride,
                             tmpPtr + (2 * i + 1) * dupSrcStride, 0, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    }

                    if(cnt % 2 == 1) {
                        vadd(tmpPtr , tmpPtr, tmpPtr + (cnt - 1) * dupSrcStride,
                             0, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    }
                    set_mask_norm();
                    set_vector_mask(-1, -1);
                }
                if (numRemainPerLine) {
                    SetContinuousMask(numRemainPerLine);
                    for (int i = 0; i < cnt / 2; i++) {
                        vadd(tmpPtr + elementsPerLine + i * dupSrcStride, tmpPtr + elementsPerLine + 2 * i * dupSrcStride,
                             tmpPtr + elementsPerLine + (2 * i + 1) * dupSrcStride, 1, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    }
                    if (cnt % 2 == 1) {
                        vadd(tmpPtr + elementsPerLine, tmpPtr + elementsPerLine, tmpPtr + elementsPerLine + (cnt - 1) * dupSrcStride,
                             1, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    }
                    set_vector_mask(-1, -1);
                }
                cnt /= 2;
            }
            copy_ubuf_to_ubuf(dstPtr, tmpPtr, 0, 1, lenBurst, 0, 0);
            pipe_barrier(PIPE_V);
        } else {
            copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, lenBurst, 0, 0);
            pipe_barrier(PIPE_V);

            if (numRepeatPerLine > 0) {
                set_mask_count();
                set_vector_mask(0, elementsPerLine);
                for (uint32_t i = 1; i < validRow; i++) {
                    vadd(dstPtr, dstPtr, srcPtr + i * dupSrcStride, 0, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                set_mask_norm();
                set_vector_mask(-1, -1);
            }

            dstPtr += elementsPerLine;
            srcPtr += elementsPerLine;

            if (numRemainPerLine) {
                SetContinuousMask(numRemainPerLine);
                for (uint32_t i = 1; i < validRow; i++) {
                    vadd(dstPtr, dstPtr, srcPtr + i * dupSrcStride, 1, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                set_vector_mask(-1, -1);
            }
        }
    }

    template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
    __aicore__ PTO_INLINE void TCOLSUM_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, bool IsBinary) {
        static_assert((sizeof(typename TileDataSrc::DType) == 2) || (sizeof(typename TileDataSrc::DType) == 4),
                      "Data type must be 16/32");
        static_assert(TileDataSrc::Loc == pto::Location::Vec, "Src location must be Vec!");
        static_assert((TileDataSrc::isRowMajor && (TileDataSrc::SFractal == SLayout::NoneBox)) &&
                      (TileDataDst::isRowMajor && (TileDataDst::SFractal == SLayout::NoneBox)),
                      "Src and dst layout must be ND!");
        if ((IsBinary && tmp.data() == nullptr) || src.GetValidRow() == 0 || src.GetValidCol() == 0) {
            return;
        }
        constexpr unsigned srcstride = TileDataSrc::RowStride;
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        if (IsBinary) {
            TColSum<typename TileDataSrc::DType, TileDataDst, TileDataSrc, TileDataTmp, srcstride,
                true>(dst.data(), src.data(), tmp.data(), validRow, validCol);
        } else {
            TColSum<typename TileDataSrc::DType, TileDataDst, TileDataSrc, TileDataTmp, srcstride,
                false>(dst.data(), src.data(), tmp.data(), validRow, validCol);
        }
    }
}
#endif