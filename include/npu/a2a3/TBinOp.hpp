/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBIN_HPP
#define TBIN_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {

#define SMALL_RPT (4)

template <typename Op, typename T>
__PTO_INSTR__ void Bin1LCountMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validRow * validCol);
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, 0);
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned rowStride>
__PTO_INSTR__ void Bin2LCountMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat>
__PTO_INSTR__ void Bin1LNormMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;  
    unsigned headRepeats = numElements / elementsPerRepeat; 
    unsigned tailElements = numElements % elementsPerRepeat; 
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, headRepeats); // headRepeats can be zero
    if (tailElements) {
        unsigned offset = headRepeats * elementsPerRepeat;
        SetContMaskByDType<T>(tailElements);
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1);
        SetFullVecMaskByDType<T>();
    }
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
__PTO_INSTR__ void Bin2LNormModeColVLAlign(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, headRepeats);
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
__PTO_INSTR__ void Bin2LNormModeHead(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop) {
                for (int j = 0; j < numLoop; j++) {
                    unsigned offset = i * stride + j * elementsPerRepeat * REPEAT_MAX;
                    Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                unsigned offset = i * stride + numLoop * elementsPerRepeat * REPEAT_MAX;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, remainAfterLoop);
            }
        }
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
__PTO_INSTR__ void Bin2LNormModeTail(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned numRemainPerLine)
{
    unsigned numLoop = 0;
    unsigned remainAfterLoop = validRow;
    constexpr bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(numRemainPerLine);
    if constexpr (Rows > pto::REPEAT_MAX) {
        numLoop = validRow / REPEAT_MAX;
        for (int i = 0; i < numLoop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    unsigned offset = i * REPEAT_MAX * stride + j * stride;
                    Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1, 1, 1, 1);
                }
            } else {
                unsigned offset = i * REPEAT_MAX * stride;
                uint8_t repeatStride = stride / blockSizeElem;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset,
                            REPEAT_MAX, repeatStride, repeatStride, repeatStride);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }
    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                unsigned offset = numLoop * REPEAT_MAX * stride + j * stride;
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1, 1, 1, 1);
            }
        } else {
            unsigned offset = numLoop * REPEAT_MAX * stride;
            uint8_t repeatStride = stride / blockSizeElem;
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset,
                    remainAfterLoop, repeatStride, repeatStride, repeatStride);
        }
    }
    SetFullVecMaskByDType<T>();
}


template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__PTO_INSTR__ void Bin2LNormModeRowRpt(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) { 
        unsigned numLoop = validCol / elementsPerRepeat; 
        unsigned tailElements = validCol % elementsPerRepeat; 
        for (unsigned i = 0; i < numLoop; i++) {
            unsigned offset = i * elementsPerRepeat;
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset,
                        validRow, repeatStride, repeatStride, repeatStride);
        }

        if (tailElements) {
            unsigned offset = numLoop * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset,
                        validRow, repeatStride, repeatStride, repeatStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Bin2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>
                (dstPtr, src0Ptr, src1Ptr, validRow, numRepeatPerLine);
            unsigned offset = numRepeatPerLine * elementsPerRepeat;
            dstPtr += offset; src0Ptr += offset; src1Ptr += offset;
        }
        if (numRemainPerLine){
            Bin2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>
                (dstPtr, src0Ptr, src1Ptr, validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__PTO_INSTR__ void BinaryInstr(__ubuf__ typename TileData::DType *dstPtr,
    __ubuf__ typename TileData::DType *src0Ptr, __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow,
    unsigned validCol)
{
    using T = typename TileData::DType;
    // continuous check in compile time
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
        if constexpr (totalRepeats > pto::REPEAT_MAX) {  // comments: [1, 96] (non-VL aligned can go into this branch)
            Bin1LCountMode<Op, T>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        } else {
            Bin1LNormMode<Op, T, elementsPerRepeat>(dstPtr, src0Ptr, src1Ptr, validRow, TileData::Cols);
        }
    } else {
        // continuous check in runtime(merge axis)
        if ((TileData::Cols == validCol) || (validRow == 1)) {
            unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
            if (totalRepeats > pto::REPEAT_MAX) {
                Bin1LCountMode<Op, T>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            } else {
                Bin1LNormMode<Op, T, elementsPerRepeat>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            }
        } else {  // not - continuous
            constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
            if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT)) {
                Bin2LCountMode<Op, T, rowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            } else if constexpr (TileData::Rows < (normColRepeat + 1)) { 
                unsigned tailElements = validCol % elementsPerRepeat;
                if (tailElements) {
                    Bin2LCountMode<Op, T, rowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
                } else {
                    Bin2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(
                        dstPtr, src0Ptr, src1Ptr, validRow, validCol);
                }
            } else {
                Bin2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                    dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            }
        }
    }
}

}  // namespace pto
#endif