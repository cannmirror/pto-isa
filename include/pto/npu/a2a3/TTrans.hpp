/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRANS_HPP
#define TTRANS_HPP

#include "pto/common/constants.hpp"
#include "pto/common/utils.hpp"

namespace pto {

template <typename T> 
struct TransOp {
    __PTO_INSTR__ static void TransB8Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        scatter_vnchwconv_b8(VA0, VA2, repeat, dstStride, srcStride, false, false);
        scatter_vnchwconv_b8(VA6, VA2, repeat, dstStride, srcStride, false, true);
        scatter_vnchwconv_b8(VA0, VA4, repeat, dstStride, srcStride, true, false);
        scatter_vnchwconv_b8(VA6, VA4, repeat, dstStride, srcStride, true, true);
    }

    __PTO_INSTR__ static void TransB16Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        scatter_vnchwconv_b16(VA0, VA2, repeat, dstStride, srcStride);
    }

    __PTO_INSTR__ static void TransB32Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        scatter_vnchwconv_b32(VA0, VA2, repeat, dstStride, srcStride);
    }

    __PTO_INSTR__ static void CopyInstr(__ubuf__ uint32_t *dstPtr, __ubuf__ uint32_t *srcPtr, uint8_t repeat, 
                                        uint16_t dstRepeatStride, uint16_t srcRepeatStride)
    {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }

    __PTO_INSTR__ static void CopyInstr(__ubuf__ uint16_t *dstPtr, __ubuf__ uint16_t *srcPtr, uint8_t repeat, 
                                        uint16_t dstRepeatStride, uint16_t srcRepeatStride)
    {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    } 
};

template <typename Op, typename T, unsigned blockElemSize, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeB32(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubtileX, unsigned numSubtileY)
{
    // numSubtileY can reach REPEAT_MAX = 255 in b32 case
    // 1 subtile = 16 * 32B = 0.5KB, UB = 192KB, so UB can contain 384 subtiles at most
    uint64_t srcUb[16] = {0};
    uint64_t dstUb[16] = {0};
    const unsigned numRepeat = (numSubtileY + REPEAT_MAX - 1) / REPEAT_MAX;
    for (unsigned repeat = 0; repeat < numRepeat;
         ++repeat, srcPtr += 16 * REPEAT_MAX * srcStride, dstPtr += 16 * REPEAT_MAX, numSubtileY -= REPEAT_MAX) {
        for (unsigned i = 0; i < numSubtileX; ++i) {
            for (unsigned j = 0; j < 16; ++j) {
                srcUb[j] = (uint64_t)(srcPtr + i * blockElemSize + j * srcStride);
                dstUb[j] = (uint64_t)(dstPtr + ((j >> 1) + i * blockElemSize) * dstStride + (j & 1) * blockElemSize);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, dstUb);
            set_va_reg_sb(VA1, &dstUb[8]);
            if (numSubtileY == 1) {
                Op::TransB32Instr(1, 0, 0);
            } else if (numSubtileY < REPEAT_MAX) {
                Op::TransB32Instr(numSubtileY, 2, 16 * srcStride / blockElemSize);
            } else {
                Op::TransB32Instr(REPEAT_MAX, 2, 16 * srcStride / blockElemSize);
            }
        }
    }
}

template <typename Op, typename T, unsigned blockElemSize, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeB16(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubtileX, unsigned numSubtileY)
{
    // numSubtileY can reach REPEAT_MAX = 255 in b16 case
    // 1 subtile = 16 * 32B = 0.5KB, UB = 192KB, so UB can contain 384 subtiles at most
    uint64_t srcUb[16] = {0};
    uint64_t dstUb[16] = {0};
    const unsigned numRepeat = (numSubtileY + REPEAT_MAX - 1) / REPEAT_MAX;
    for (unsigned repeat = 0; repeat < numRepeat;
         ++repeat, srcPtr += 16 * REPEAT_MAX * srcStride, dstPtr += 16 * REPEAT_MAX, numSubtileY -= REPEAT_MAX) {
        for (unsigned i = 0; i < numSubtileX; ++i) {
            for (unsigned j = 0; j < 16; ++j) {
                srcUb[j] = (uint64_t)(srcPtr + i * blockElemSize + j * srcStride);
                dstUb[j] = (uint64_t)(dstPtr + (j + i * blockElemSize) * dstStride);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, dstUb);
            set_va_reg_sb(VA1, &dstUb[8]);
            if (numSubtileY == 1) {
                Op::TransB16Instr(1, 0, 0);
            } else if (numSubtileY < REPEAT_MAX) {
                Op::TransB16Instr(numSubtileY, 1, 16 * srcStride / blockElemSize);
            } else {
                Op::TransB16Instr(REPEAT_MAX, 1, 16 * srcStride / blockElemSize);
            }
        }
    }
}

template <typename Op, typename T, unsigned blockElemSize, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeB8(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubtileX, unsigned numSubtileY)
{
    // numSubtileY couldn't reach REPEAT_MAX = 255 in b8 case
    // 1 subtile = 32 * 32B = 1KB, UB = 192KB, so UB can contain 192 subtiles at most
    uint64_t srcUb[16] = {0};
    uint64_t dstUb[16] = {0};
    uint64_t srcUb1[16] = {0};
    uint64_t dstUb1[16] = {0};
    for (unsigned i = 0; i < numSubtileX; ++i) {
        for (unsigned j = 0; j < 16; ++j) {
            srcUb[j] = (uint64_t)(srcPtr + i * blockElemSize + j * srcStride);
            srcUb1[j] = srcUb[j] + 16 * srcStride;
            dstUb[j] = (uint64_t)(dstPtr + (j + i * blockElemSize) * dstStride);
            dstUb1[j] = dstUb[j] + 16 * dstStride;
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, dstUb);
        set_va_reg_sb(VA1, &dstUb[8]);
        set_va_reg_sb(VA4, srcUb1);
        set_va_reg_sb(VA5, &srcUb1[8]);
        set_va_reg_sb(VA6, dstUb1);
        set_va_reg_sb(VA7, &dstUb1[8]);
        if (numSubtileY == 1) {
            Op::TransB8Instr(1, 0, 0);
        } else {
            Op::TransB8Instr(numSubtileY, 1, 32 * srcStride / blockElemSize);
        }
    }
}

template <typename Op, typename T, unsigned blockElemSize, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeFullSubTiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubtileX, unsigned numSubtileY)
{
    if (numSubtileX == 0 || numSubtileY == 0) {
        return;
    }

    if constexpr (sizeof(T) == 4) {  // b32 case
        TransposeB32<Op, T, blockElemSize, dstStride, srcStride>(dstPtr, srcPtr, numSubtileX, numSubtileY);
    } else if constexpr (sizeof(T) == 2) {
        TransposeB16<Op, T, blockElemSize, dstStride, srcStride>(dstPtr, srcPtr, numSubtileX, numSubtileY);
    } else if constexpr (sizeof(T) == 1) {
        TransposeB8<Op, T, blockElemSize, dstStride, srcStride>(dstPtr, srcPtr, numSubtileX, numSubtileY);
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TTRANS: Invalid data type.");
    }
}

template <typename Op, typename T, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void CopyRowsWithMask(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numRow, unsigned numTail)
{
    if (numRow == 0 || numTail == 0) {
        return;
    }

    constexpr uint16_t blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint16_t srcRepeatSize = srcStride / blockElemSize;
    constexpr uint16_t dstRepeatSize = dstStride / blockElemSize;

    if constexpr (sizeof(T) == 4) {
        SetContMaskByDType<T>(numTail);
        Op::CopyInstr((__ubuf__ uint32_t *)dstPtr, (__ubuf__ uint32_t *)srcPtr, numRow, dstRepeatSize, srcRepeatSize);
    } else if constexpr (sizeof(T) == 2) {
        SetContMaskByDType<T>(numTail);
        Op::CopyInstr((__ubuf__ uint16_t *)dstPtr, (__ubuf__ uint16_t *)srcPtr, numRow, dstRepeatSize, srcRepeatSize);
    } else if constexpr (sizeof(T) == 1) {
        if (numTail > 1) {
            SetContMaskByDType<T>(numTail >> 1);
            Op::CopyInstr((__ubuf__ uint16_t *)dstPtr, (__ubuf__ uint16_t *)srcPtr, numRow, dstRepeatSize, srcRepeatSize);
        }
        // vcopy(...) doesn't support b8 data type pointers. So in rare case of odd numTail we should additionally use
        // scalar copy from src to dst for the last dst column
        if (numTail % 2) {
            // The sync is necessary for scalar copy to be after all vector ops
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            for (unsigned i = 0; i < numRow; ++i) {
                dstPtr[i * dstStride + numTail - 1] = srcPtr[i * srcStride + numTail - 1];
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        }
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TTRANS: Invalid data type.");
    }
}

template <typename Op, typename T, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeXTailSubtiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numTailX, unsigned numSubtileY)
{
    if (numTailX == 0 || numSubtileY == 0) {
        return;
    }

    constexpr uint16_t blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint16_t tmpStride = TMP_UB_SIZE / BLOCK_BYTE_SIZE / sizeof(T);  // 8KB/32B/sizeof(T) elements
    constexpr uint16_t fullBurst = tmpStride / blockElemSize;
    constexpr uint16_t dstFullGap = (uint16_t)(dstStride / blockElemSize) - fullBurst;
    constexpr uint16_t tmpSubtilesMax = sizeof(T) == 1 ? tmpStride / 32 : tmpStride / 16;

    __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB, start from 184KB, UB:192KB=184+8KB
    const unsigned fullIterNum = numSubtileY / tmpSubtilesMax;
    for (unsigned iter = 0; iter < fullIterNum; ++iter) {
        TransposeFullSubTiles<Op, T, blockElemSize, tmpStride, srcStride>(
            tmpPtr, (srcPtr + iter * tmpStride * srcStride), 1, tmpSubtilesMax);
        pipe_barrier(PIPE_V);
        copy_ubuf_to_ubuf((dstPtr + iter * tmpStride), tmpPtr, 0, numTailX, fullBurst, 0, dstFullGap);
        pipe_barrier(PIPE_V);
    }

    uint16_t tmpSubtilesTail = numSubtileY % tmpSubtilesMax;
    if (tmpSubtilesTail > 0) {
        const uint16_t tailBurst =
            sizeof(T) == 1 ? 32 * tmpSubtilesTail / blockElemSize : 16 * tmpSubtilesTail / blockElemSize;
        const uint16_t tmpTailGap = tmpStride / blockElemSize - tailBurst;
        const uint16_t dstTailGap = (uint16_t)(dstStride / blockElemSize) - tailBurst;
        TransposeFullSubTiles<Op, T, blockElemSize, tmpStride, srcStride>(
            tmpPtr, (srcPtr + fullIterNum * tmpStride * srcStride), 1, tmpSubtilesTail);
        pipe_barrier(PIPE_V);
        copy_ubuf_to_ubuf((dstPtr + fullIterNum * tmpStride), tmpPtr, 0, numTailX, tailBurst, tmpTailGap, dstTailGap);
        pipe_barrier(PIPE_V);
    }
}

template <typename Op, typename T, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeYTailSubtiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubtileX, unsigned numTailY)
{
    if (numSubtileX == 0 || numTailY == 0) {
        return;
    }

    constexpr uint16_t blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint16_t tmpStride = sizeof(T) == 1 ? 32 : 16;
    // Here we decrement tmpSubtilesMax to make sure the copying within TMP UB buffer (8KB)
    // and tmpRowsMax within repeatTimes limit (255)
    constexpr uint16_t tmpSubtilesMax = TMP_UB_SIZE / BLOCK_BYTE_SIZE / tmpStride - 1;  // 8KB/32B/tmpStride-1
    constexpr uint16_t tmpRowsMax = tmpSubtilesMax * blockElemSize;
    static_assert(tmpRowsMax <= REPEAT_MAX);

    __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB, start from 184KB, UB:192KB=184+8KB
    const unsigned fullIterNum = numSubtileX / tmpSubtilesMax;
    for (unsigned iter = 0; iter < fullIterNum; ++iter) {
        TransposeFullSubTiles<Op, T, blockElemSize, tmpStride, srcStride>(
            tmpPtr, (srcPtr + iter * tmpRowsMax), tmpSubtilesMax, 1);
        pipe_barrier(PIPE_V);
        CopyRowsWithMask<Op, T, dstStride, tmpStride>(
            (dstPtr + iter * tmpRowsMax * dstStride), tmpPtr, tmpRowsMax, numTailY);
        pipe_barrier(PIPE_V);
    }

    int tmpSubtilesTail = numSubtileX % tmpSubtilesMax;
    if (tmpSubtilesTail > 0) {
        TransposeFullSubTiles<Op, T, blockElemSize, tmpStride, srcStride>(
            tmpPtr, (srcPtr + fullIterNum * tmpSubtilesMax * blockElemSize), tmpSubtilesTail, 1);
        pipe_barrier(PIPE_V);
        CopyRowsWithMask<Op, T, dstStride, tmpStride>(
            (dstPtr + fullIterNum * tmpRowsMax * dstStride), tmpPtr, tmpSubtilesTail * blockElemSize, numTailY);
        pipe_barrier(PIPE_V);
    }
}

template <typename Op, typename T, unsigned blockElemSize, unsigned dstStride, unsigned srcStride>
__PTO_INSTR__ void TransposeXYTailSubtile(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numTailX, unsigned numTailY)
{
    if (numTailX == 0 || numTailY == 0) {
        return;
    }

    constexpr unsigned tmpStride = sizeof(T) == 1 ? 32 : 16;

    __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB, start from 184KB, UB:192KB=184+8KB
    TransposeFullSubTiles<Op, T, blockElemSize, tmpStride, srcStride>(tmpPtr, srcPtr, 1, 1);
    pipe_barrier(PIPE_V);
    CopyRowsWithMask<Op, T, dstStride, tmpStride>(dstPtr, tmpPtr, numTailX, numTailY);
    pipe_barrier(PIPE_V);
}

template <typename TileData, unsigned dstStride, unsigned srcStride>
__tf__ __PTO_INSTR__ void TTrans(typename TileData::TileDType __out__ dst, 
    typename TileData::TileDType __in__ src, unsigned validRow, unsigned validCol)
{
    using T = typename TileData::DType;
    constexpr unsigned blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);

    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    const unsigned numFullSubtileX = validCol / blockElemSize;
    const unsigned numTailX = validCol % blockElemSize;
    const unsigned numFullSubtileY = (sizeof(T) == 1) ? validRow / 32 : validRow / 16;
    const unsigned numTailY = (sizeof(T) == 1) ? validRow % 32 : validRow % 16;

    TransposeFullSubTiles<TransOp<T>, T, blockElemSize, dstStride, srcStride>(
        dstPtr, srcPtr, numFullSubtileX, numFullSubtileY);

    TransposeXTailSubtiles<TransOp<T>, T, dstStride, srcStride>(
        dstPtr + (validCol - numTailX) * dstStride, srcPtr + (validCol - numTailX), numTailX, numFullSubtileY);

    TransposeYTailSubtiles<TransOp<T>, T, dstStride, srcStride>(
        dstPtr + (validRow - numTailY), srcPtr + (validRow - numTailY) * srcStride, numFullSubtileX, numTailY);

    TransposeXYTailSubtile<TransOp<T>, T, blockElemSize, dstStride, srcStride>(
        dstPtr + (validCol - numTailX) * dstStride + (validRow - numTailY),
        srcPtr + (validRow - numTailY) * srcStride + (validCol - numTailX),
        numTailX, numTailY);
}

template <typename TileDataDst, typename TileDataSrc>
__PTO_INSTR__ void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    using T = typename TileDataSrc::DType;
    static_assert( sizeof(T) == sizeof(typename TileDataDst::DType), "TTRANS: Inconsistent data types.");
    static_assert(TileDataSrc::isRowMajor, "TTRANS: Inconsistent source Layout type.");
    static_assert(TileDataDst::isRowMajor, "TTRANS: Inconsistent destination Layout type.");

    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    TTrans<TileDataSrc, dstStride, srcStride>(dst.data(), src.data(), validRow, validCol);
}
}  // namespace pto
#endif