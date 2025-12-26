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

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
constexpr unsigned SMALL_RPT_BINOP = 4;

template <typename Op, typename T>
PTO_INTERNAL void Bin1LCountMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol) {
    set_mask_count();
    SetVectorCount(validRow * validCol);
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, 0);
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned rowStride>
PTO_INTERNAL void Bin2LCountMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol) {
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned tileCols, uint8_t repeatStride>
PTO_INTERNAL void Bin1LNormModeSmall(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol) {
    SetContMaskByDType<T>(validCol);
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, validRow, repeatStride, repeatStride, repeatStride);
    SetFullVecMaskByDType<T>();
    return;
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride,
    unsigned tileCols>
PTO_INTERNAL void Bin1LNormMode(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol) {
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinInstr(dstPtr, src0Ptr, src1Ptr, headRepeats); // headRepeats can be zero
    if (tailElements) [[unlikely]] {
        unsigned offset = headRepeats * elementsPerRepeat;
        SetContMaskByDType<T>(tailElements);
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, 1);
        SetFullVecMaskByDType<T>();
    }
}

template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeColVLAlign(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol) {
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        unsigned offset = i * rowStride;
        Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, headRepeats);
    }
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeHead(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned numRepeatPerLine) {
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop) [[unlikely]] {
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
PTO_INTERNAL void Bin2LNormModeTail(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned numRemainPerLine) {
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
                Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, REPEAT_MAX, repeatStride,
                    repeatStride, repeatStride);
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
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, remainAfterLoop, repeatStride,
                repeatStride, repeatStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem,
    unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeRowRpt(
    __ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, unsigned validRow, unsigned validCol) {
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            unsigned offset = i * elementsPerRepeat;
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, validRow, repeatStride, repeatStride,
                repeatStride);
        }

        if (tailElements) {
            unsigned offset = numLoop * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::BinInstr(dstPtr + offset, src0Ptr + offset, src1Ptr + offset, validRow, repeatStride, repeatStride,
                repeatStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Bin2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, numRepeatPerLine);
            unsigned offset = numRepeatPerLine * elementsPerRepeat;
            dstPtr += offset;
            src0Ptr += offset;
            src1Ptr += offset;
        }
        if (numRemainPerLine) {
            Bin2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstrFastPath(__ubuf__ typename TileData::DType *dstPtr,
    __ubuf__ typename TileData::DType *src0Ptr, __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow,
    unsigned validCol) {
    using T = typename TileData::DType;
    constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
    if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
        Bin1LCountMode<Op, T>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        Bin1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstrGeneralPath(__ubuf__ typename TileData::DType *dstPtr,
    __ubuf__ typename TileData::DType *src0Ptr, __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow,
    unsigned validCol) {
    using T = typename TileData::DType;
    // Continuous check in runtime(merge axis)
    if ((TileData::Cols == validCol) || (validRow == 1)) [[likely]] {
        unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
        bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
        if (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) [[unlikely]] {
            Bin1LCountMode<Op, T>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        } else {
            Bin1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        }
    } else { // Non continuous
        constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
        if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_BINOP)) {
            Bin2LCountMode<Op, T, rowStride>(dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
            if ((validCol % elementsPerRepeat) > 0) {
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

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstr(__ubuf__ typename TileData::DType *dstPtr, __ubuf__ typename TileData::DType *src0Ptr,
    __ubuf__ typename TileData::DType *src1Ptr, unsigned validRow, unsigned validCol) {
    using T = typename TileData::DType;
    // Small shape optimization
    if constexpr ((TileData::Rows <= pto::REPEAT_MAX) && (TileData::Cols < elementsPerRepeat)) {
        constexpr uint8_t repeatStride = rowStride / blockSizeElem;
        Bin1LNormModeSmall<Op, T, elementsPerRepeat, TileData::Cols, repeatStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        return;
    }
    // Continuous check in compile time
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        BinaryInstrFastPath<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        BinaryInstrGeneralPath<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    }
}
} // namespace pto
#endif