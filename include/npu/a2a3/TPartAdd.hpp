/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPATIALADD_HPP
#define TPATIALADD_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
template <typename T, typename TileDataDst, typename TileDataSrc, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
__aicore__
PTO_INLINE
void TPartCopyInstr(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr,
    uint64_t validRow, uint64_t validCol, uint64_t startRow)
{
    validRow -= startRow;
    srcPtr += startRow * TileDataDst::RowStride;
    dstPtr += startRow * TileDataSrc::RowStride;

    set_mask_count();  // counter mode
    if constexpr (sizeof(T) == 4) {
        uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
        uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
        unsigned numLoop = validRow / REPEAT_MAX;  // REPEAT_MAX = 255
        unsigned remainAfterLoop = validRow % REPEAT_MAX;
        set_vector_mask(0, validCol);
        for (unsigned i = 0; i < numLoop; i++) {
            vcopy((__ubuf__ int32_t *)(dstPtr + i * validCol * REPEAT_MAX),
                (__ubuf__ int32_t *)(srcPtr + i * validCol * REPEAT_MAX),
                REPEAT_MAX, 1, 1, dstRepeatSize, srcRepeatSize);
        }
        if (remainAfterLoop) {
            vcopy((__ubuf__ int32_t *)(dstPtr + numLoop * validCol * REPEAT_MAX),
                (__ubuf__ int32_t *)(srcPtr + numLoop * validCol * REPEAT_MAX),
                remainAfterLoop, 1, 1, dstRepeatSize, srcRepeatSize);
        }
    } else if constexpr (sizeof(T) == 2) {
        uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
        uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
        unsigned numLoop = validRow / REPEAT_MAX;  // REPEAT_MAX = 255
        unsigned remainAfterLoop = validRow % REPEAT_MAX;
        set_vector_mask(0, validCol);
        for (unsigned i = 0; i < numLoop; i++) {
            vcopy((__ubuf__ int16_t *)(dstPtr + i * validCol * REPEAT_MAX),
                (__ubuf__ int16_t *)(srcPtr + i * validCol * REPEAT_MAX),
                REPEAT_MAX, 1, 1, dstRepeatSize, srcRepeatSize);
        }
        if (remainAfterLoop) {
            vcopy((__ubuf__ int16_t *)(dstPtr + numLoop * validCol * REPEAT_MAX),
                (__ubuf__ int16_t *)(srcPtr + numLoop * validCol * REPEAT_MAX),
                remainAfterLoop, 1, 1, dstRepeatSize, srcRepeatSize);
        }
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2, "TPARTADD Invalid data type.");
    }
    set_mask_norm();  // restore to norm mode
    set_vector_mask(-1, -1);
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstStride, unsigned src0Stride, unsigned src1Stride>
__aicore__
PTO_INLINE
void TPartAddInstr(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
    unsigned validRow, unsigned validCol) {
    unsigned numRepeatPerLine = validCol / elementsPerRepeat;
    unsigned numRemainPerLine = validCol % elementsPerRepeat;

    if (numRepeatPerLine) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    vadd(dstPtr + i * dstStride + j * REPEAT_MAX * elementsPerRepeat,
                         src0Ptr + i * src0Stride + j * REPEAT_MAX * elementsPerRepeat,
                         src1Ptr + i * src1Stride + j * REPEAT_MAX * elementsPerRepeat,
                         REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                }
            }
            if (remainAfterLoop) {
                vadd(dstPtr + i * dstStride + numLoop * REPEAT_MAX * elementsPerRepeat,
                     src0Ptr + i * src0Stride + numLoop * REPEAT_MAX * elementsPerRepeat,
                     src1Ptr + i * src1Stride + numLoop * REPEAT_MAX * elementsPerRepeat,
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
        bool constexpr strideOverFlag = ((src0Stride / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                          (src1Stride / blockSizeElem > REPEAT_STRIDE_MAX) ||
                                          (dstStride / blockSizeElem > REPEAT_STRIDE_MAX));
        SetContinuousMask(numRemainPerLine);
        for (unsigned i = 0; i < numLoop; i++) { 
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    vadd(dstPtr + i * REPEAT_MAX * dstStride + j * dstStride,
                         src0Ptr + i * REPEAT_MAX * src0Stride + j * src0Stride,
                         src1Ptr + i * REPEAT_MAX * src1Stride + j * src1Stride,
                         1, 1, 1, 1, 1, 1, 1);
                }
            } else {
                vadd(dstPtr + i * REPEAT_MAX * dstStride,
                     src0Ptr + i * REPEAT_MAX * src0Stride,
                     src1Ptr + i * REPEAT_MAX * src1Stride,
                     REPEAT_MAX, 1, 1, 1,
                     dstStride / blockSizeElem, src0Stride / blockSizeElem, src1Stride / blockSizeElem);
            }
        }
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (unsigned j = 0; j < remainAfterLoop; j++) {
                    vadd(dstPtr + numLoop * REPEAT_MAX * dstStride + j * dstStride,
                         src0Ptr + numLoop * REPEAT_MAX * src0Stride + j * src0Stride,
                         src1Ptr + numLoop * REPEAT_MAX * src1Stride + j * src1Stride,
                         1, 1, 1, 1, 1, 1, 1);
                }
            } else {
                vadd(dstPtr + numLoop * REPEAT_MAX * dstStride,
                     src0Ptr + numLoop * REPEAT_MAX * src0Stride,
                     src1Ptr + numLoop * REPEAT_MAX * src1Stride,
                     remainAfterLoop, 1, 1, 1,
                     dstStride / blockSizeElem, src0Stride / blockSizeElem, src1Stride / blockSizeElem);
            }
        }
        set_vector_mask(-1, -1);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
__tf__
__aicore__
PTO_INLINE
void TPartAdd(typename TileDataDst::TileDType __out__ dst,
    typename TileDataSrc0::TileDType __in__ src0, typename TileDataSrc1::TileDType __in__ src1, unsigned src0ValidRow,
    unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    if (dstValidRow == 0 || dstValidCol == 0) {
        return;
    }
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    bool condSrc0EqDst = (src0ValidRow == dstValidRow && src0ValidCol == dstValidCol);
    bool condSrc0RowLtDst = (src0ValidRow < dstValidRow && src0ValidCol == dstValidCol);
    bool condSrc0ColLtDst = (src0ValidRow == dstValidRow && src0ValidCol < dstValidCol);
    bool condSrc1EqDst = (src1ValidRow == dstValidRow && src1ValidCol == dstValidCol);
    bool condSrc1RowLtDst = (src1ValidRow < dstValidRow && src1ValidCol == dstValidCol);
    bool condSrc1ColLtDst = (src1ValidRow == dstValidRow && src1ValidCol < dstValidCol);

    if (condSrc0EqDst && condSrc1EqDst) {  // src0 == src1 == dst
        TPartAddInstr<T, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem,
            dstRowStride, src0RowStride, src1RowStride>(
            dstPtr, src0Ptr, src1Ptr, dstValidRow, dstValidCol);
    } else if (condSrc0ColLtDst && condSrc1EqDst) {  // src0Col < dstCol
        TPartCopyInstr<T, TileDataDst, TileDataSrc1, blockSizeElem, dstRowStride, src1RowStride>(
            dstPtr, src1Ptr, src1ValidRow, dstValidCol, 0);
        if (src0ValidCol != 0) {
            pipe_barrier(PIPE_V);
            TPartAddInstr<T, TileDataDst, TileDataSrc0, TileDataDst, elementsPerRepeat, blockSizeElem,
                dstRowStride, src0RowStride, dstRowStride>(
                dstPtr, src0Ptr, dstPtr, src0ValidRow, src0ValidCol);
        }
    } else if (condSrc0RowLtDst && condSrc1EqDst) {  // src0Row < dstRow
        if (src0ValidRow != 0) {
            TPartAddInstr<T, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem,
                dstRowStride, src0RowStride, src1RowStride>(
                dstPtr, src0Ptr, src1Ptr, src0ValidRow, src0ValidCol);
            pipe_barrier(PIPE_V);
        }
        TPartCopyInstr<T, TileDataDst, TileDataSrc1, blockSizeElem, dstRowStride, src1RowStride>(
            dstPtr, src1Ptr, src1ValidRow, dstValidCol, src0ValidRow);    
    } else if (condSrc1ColLtDst && condSrc0EqDst) {  // src1Col < dstCol
        TPartCopyInstr<T, TileDataDst, TileDataSrc0, blockSizeElem, dstRowStride, src0RowStride>(
            dstPtr, src0Ptr, src0ValidRow, dstValidCol, 0);
        if (src1ValidCol != 0) {
            pipe_barrier(PIPE_V);
            TPartAddInstr<T, TileDataDst, TileDataSrc1, TileDataDst, elementsPerRepeat, blockSizeElem,
                dstRowStride, src1RowStride, dstRowStride>(
                dstPtr, src1Ptr, dstPtr, src1ValidRow, src1ValidCol);
        }
    } else if (condSrc1RowLtDst && condSrc0EqDst) {  // src1Row < dstRow
        if (src1ValidRow != 0) {
            TPartAddInstr<T, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem,
                dstRowStride, src0RowStride, src1RowStride>(
                dstPtr, src0Ptr, src1Ptr, src1ValidRow, src1ValidCol);
            pipe_barrier(PIPE_V);
        }
        TPartCopyInstr<T, TileDataDst, TileDataSrc0, blockSizeElem, dstRowStride, src0RowStride>(
            dstPtr, src0Ptr, src0ValidRow, dstValidCol, src1ValidRow);    
    }  // unsupport other conditions
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__aicore__
PTO_INLINE
void TPARTADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    static_assert(std::is_same<typename TileDataDst::DType, typename TileDataSrc0::DType>::value &&
                  std::is_same<typename TileDataDst::DType, typename TileDataSrc1::DType>::value,
                  "TPARTADD: src and dst data type is different!");
    static_assert((std::is_same<typename TileDataDst::DType, int32_t>::value) ||
                  (std::is_same<typename TileDataDst::DType, int>::value) ||
                  (std::is_same<typename TileDataDst::DType, int16_t>::value) ||
                  (std::is_same<typename TileDataDst::DType, half>::value) ||
                  (std::is_same<typename TileDataDst::DType, float16_t>::value) ||
                  (std::is_same<typename TileDataDst::DType, float>::value) ||
                  (std::is_same<typename TileDataDst::DType, float32_t>::value),
                  "TPARTADD: Invalid data type.");
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

    TPartAdd<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride,
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
}  // namespace pto
#endif