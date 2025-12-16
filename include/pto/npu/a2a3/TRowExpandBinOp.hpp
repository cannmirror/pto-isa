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
template <typename Op, typename T, unsigned rowStride>
PTO_INTERNAL
void TRowExpandBinaryInstr(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
    unsigned validRow, unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T); 
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::RowExpandBinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + i * blockSizeElem, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, typename U, unsigned rowStride>
PTO_INTERNAL
void TRowExpandBinaryInstr(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ U *src1Ptr, __ubuf__ T *tmpPtr,
    __ubuf__ U *tmpPtr_, unsigned validRow, unsigned validCol)
{
    constexpr int repeatMax = 32; // tmpbuf只能存放vbrcb 32个repeat的数据
    int repeatTimes = CeilDivision(validRow, 8);
    int numLoop = repeatTimes / repeatMax;
    int numRemainAfterLoop = repeatTimes % repeatMax;
    unsigned offset = 256 * rowStride;
    for ( unsigned i = 0; i < numLoop; i++) {
        vbrcb(tmpPtr_, src1Ptr, 1, 8, repeatMax);
        pipe_barrier(PIPE_V);
        TRowExpandBinaryInstr<Op, T, rowStride>(dstPtr, src0Ptr, tmpPtr, 256, validCol);
        pipe_barrier(PIPE_V);
        dstPtr += offset;
        src0Ptr += offset;
        src1Ptr += 256;
    }
    if (numRemainAfterLoop) {
        vbrcb(tmpPtr_, src1Ptr, 1, 8, numRemainAfterLoop);
        pipe_barrier(PIPE_V);
        TRowExpandBinaryInstr<Op, T, rowStride>(dstPtr, src0Ptr, tmpPtr, validRow % 256, validCol);
    }
}
}  // namespace pto
#endif