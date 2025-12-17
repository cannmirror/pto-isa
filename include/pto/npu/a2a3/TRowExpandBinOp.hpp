/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPANDBIN_HPP
#define TROWEXPANDBIN_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename Op, typename T, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TRowExpandBinaryCountMode(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
    unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::RowExpandBinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + i * blockSizeElem, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TRowExpandBinaryNormMode(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
    unsigned validRow, unsigned validCol)
{
    constexpr uint8_t repeatStride = (uint8_t)(rowStride / blockSizeElem); // rowStride / blockSizeElem>256不会进到norm mode
    if constexpr (rowStride <= elementsPerRepeat) {
        SetContMaskByDType<T>(validCol);
        Op::RowExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride);
        SetFullVecMaskByDType<T>();
    } else {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned numRemainAfterLoop = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            Op::RowExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride);
            dstPtr += elementsPerRepeat;
            src0Ptr += elementsPerRepeat;
        }
        if (numRemainAfterLoop) {
            SetContMaskByDType<T>(numRemainAfterLoop);
            Op::RowExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride);
            SetFullVecMaskByDType<T>();
        }
    }
}

template <typename Op, typename T, typename U, int row, unsigned rowStride>
PTO_INTERNAL
void TRowExpandBinaryInstr(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ U *src1Ptr, __ubuf__ T *tmpPtr,
    __ubuf__ U *tmpPtr_, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    unsigned repeatTimes = CeilDivision(validRow, 8);
    constexpr bool repeatStrideOverflow = rowStride / blockSizeElem > 255;
    bool useCountMode = repeatStrideOverflow || rowStride / blockSizeElem > validRow;
    constexpr unsigned repeatMax = 30; // tmpbuf只能存放vbrcb 32个repeat的数据,32个repeat256行大于REPEAT_MAX，不好处理，所以取30repeat240行
    constexpr unsigned MAX_ROW = 240;
    if constexpr (row < 256) {
        vbrcb(tmpPtr_, src1Ptr, 1, 8, repeatTimes);
        pipe_barrier(PIPE_V);
        if (useCountMode) {
            TRowExpandBinaryCountMode<Op, T, blockSizeElem, rowStride>(dstPtr, src0Ptr, tmpPtr, validRow, validCol);
        } else {
            TRowExpandBinaryNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, src0Ptr, tmpPtr, validRow, validCol);
        }
    } else {
        if (validRow < 256) {
            vbrcb(tmpPtr_, src1Ptr, 1, 8, repeatTimes);
            pipe_barrier(PIPE_V);
            if (useCountMode) {
                TRowExpandBinaryCountMode<Op, T, blockSizeElem, rowStride>(dstPtr, src0Ptr, tmpPtr, validRow, validCol);
            } else {
                TRowExpandBinaryNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, src0Ptr, tmpPtr, validRow, validCol);
            }
        } else {
            unsigned numLoop = repeatTimes / repeatMax;
            unsigned numRemainAfterLoop = repeatTimes % repeatMax;
            unsigned offset = MAX_ROW * rowStride;
            for ( unsigned i = 0; i < numLoop; i++) {
                vbrcb(tmpPtr_, src1Ptr, 1, 8, repeatMax);
                pipe_barrier(PIPE_V);
                TRowExpandBinaryNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, src0Ptr, tmpPtr, MAX_ROW, validCol); // 大于256行时repeatstride不会越界，可以用norm mode
                pipe_barrier(PIPE_V);
                dstPtr += offset;
                src0Ptr += offset;
                src1Ptr += MAX_ROW;
            }
            if (numRemainAfterLoop) {
                vbrcb(tmpPtr_, src1Ptr, 1, 8, numRemainAfterLoop);
                pipe_barrier(PIPE_V);
                TRowExpandBinaryNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, src0Ptr, tmpPtr, validRow % 240, validCol);
            }
        }
    }
}
}  // namespace pto
#endif