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
    PTO_INTERNAL static void TransB8Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride) {
        scatter_vnchwconv_b8(VA0, VA2, repeat, dstStride, srcStride, false, false);
        scatter_vnchwconv_b8(VA6, VA2, repeat, dstStride, srcStride, false, true);
        scatter_vnchwconv_b8(VA0, VA4, repeat, dstStride, srcStride, true, false);
        scatter_vnchwconv_b8(VA6, VA4, repeat, dstStride, srcStride, true, true);
    }

    PTO_INTERNAL static void TransB16Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride) {
        scatter_vnchwconv_b16(VA0, VA2, repeat, dstStride, srcStride);
    }

    PTO_INTERNAL static void TransB32Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride) {
        scatter_vnchwconv_b32(VA0, VA2, repeat, dstStride, srcStride);
    }

    PTO_INTERNAL static void CopyInstr(__ubuf__ uint32_t *dstPtr, __ubuf__ uint32_t *srcPtr, uint8_t repeat,
        uint16_t dstRepeatStride, uint16_t srcRepeatStride) {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }

    PTO_INTERNAL static void CopyInstr(__ubuf__ uint16_t *dstPtr, __ubuf__ uint16_t *srcPtr, uint8_t repeat,
        uint16_t dstRepeatStride, uint16_t srcRepeatStride) {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};

template <typename Op, typename T, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TransB32FullSubTiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubTileX, unsigned numSubTileY) {
    if ((numSubTileY > 0) && (numSubTileX > 0)) {
        constexpr uint16_t vconvSrcStride = 16 * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileX; i++) {
            uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
            uint64_t offset = i * blockSizeElem;
            for (int j = 0; j < 16; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                tmpUb[j] = (uint64_t)(dstPtr + ((j >> 1) + offset) * dstStride + (j & 1) * blockSizeElem);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[8]);
            if (numSubTileY == 1) {
                Op::TransB32Instr(1, 0, 0);
            } else {
                Op::TransB32Instr(numSubTileY, 2, vconvSrcStride);
            }
        } // end loop num_subtile
    }
}

template <typename Op, typename T, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TransB16FullSubTiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubTileX, unsigned numSubTileY) {
    if ((numSubTileY > 0) && (numSubTileX > 0)) {
        constexpr uint16_t vconvSrcStride = 16 * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileX; i++) {
            uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
            uint64_t offset = i * blockSizeElem;
            for (int j = 0; j < 16; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                tmpUb[j] = (uint64_t)(dstPtr + (j + offset) * dstStride);
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[8]);
            if (numSubTileY == 1) {
                Op::TransB16Instr(1, 0, 0);
            } else {
                Op::TransB16Instr(numSubTileY, 1, vconvSrcStride);
            }
        } // end loop num_subtile
    }
}

template <typename Op, typename T, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TransB8FullSubTiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubTileX, unsigned numSubTileY) {
    if (numSubTileX) { // [32, 32] aligned
        uint64_t srcUb[16] = {0}, srcUb1[16] = {0}, tmpUb[16] = {0}, tmpUb1[16] = {0};
        uint16_t vconvSrcStride = 32 * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileX; i++) {
            uint64_t offset = i * blockSizeElem;
            for (int j = 0; j < 16; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                srcUb1[j] = srcUb[j] + 16 * srcStride;
                tmpUb[j] = (uint64_t)(dstPtr + (j + offset) * dstStride);
                tmpUb1[j] = tmpUb[j] + 16 * dstStride;
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[8]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[8]);

            set_va_reg_sb(VA4, srcUb1);
            set_va_reg_sb(VA5, &srcUb1[8]);
            set_va_reg_sb(VA6, tmpUb1);
            set_va_reg_sb(VA7, &tmpUb1[8]);
            if (numSubTileY == 1) { // [32, 32]
                Op::TransB8Instr(1, 0, 0);
            } else { // larger then [32, 32], e.g, [32, 64]
                Op::TransB8Instr(numSubTileY, 1, vconvSrcStride);
            }
        } // end of numSubTileX
    }
}

template <typename Op, typename T, unsigned blockSizeElem, unsigned srcStride>
PTO_INTERNAL void TransB32YTailTiles(__ubuf__ T *tmpPtr, __ubuf__ T *srcPtr, unsigned tmpStride, unsigned numSubTileX,
    unsigned numSubTileY, unsigned remain_y) {
    if (remain_y > 0) {
        uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
        uint64_t offset = numSubTileY * 16;
        for (int i = 0; i < remain_y; i++) {
            srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
        }
        for (int i = 0; i < 16; i++) {
            tmpUb[i] = (uint64_t)(tmpPtr + (i & 1) * blockSizeElem + (i >> 1) * tmpStride);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[8]);

        if (numSubTileX == 1) {
            Op::TransB32Instr(1, 0, 0);
        } else {
            uint16_t vconvSrcStride = blockSizeElem * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE;
            Op::TransB32Instr(numSubTileX, vconvSrcStride, 1);
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TransB16YTailTiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubTileX, unsigned numSubTileY, unsigned remain_y) {
    if (remain_y) {
        uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
        uint64_t offset = numSubTileY * 16;
        for (int i = 0; i < remain_y; i++) {
            srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
        }
        for (int i = 0; i < 16; i++) {
            tmpUb[i] = (uint64_t)(dstPtr + offset + i * dstStride);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[8]);
        if (numSubTileX == 1) {
            Op::TransB16Instr(1, 0, 0);
        } else {
            uint16_t vconvSrcStride = blockSizeElem * dstStride * sizeof(T) / BLOCK_BYTE_SIZE;
            Op::TransB16Instr(numSubTileX, vconvSrcStride, 1);
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TransB8YTailTiles(
    __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned numSubTileX, unsigned numSubTileY, unsigned remain_y) {
    if (remain_y) { // e.g., [8, 32]
        uint64_t srcUb[16] = {0}, tmpUb[16] = {0};
        uint64_t srcUb1[16] = {0}, tmpUb1[16] = {0};
        uint64_t offset = numSubTileY * 32;
        for (int i = 0; i < remain_y; i++) {
            if (i < 16) {
                srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
            } else {
                srcUb1[i - 16] = (uint64_t)(srcPtr + (offset + i) * srcStride);
            }
        }
        for (int i = 0; i < 16; i++) {
            tmpUb[i] = (uint64_t)(dstPtr + offset + i * dstStride);
            tmpUb1[i] = tmpUb[i] + 16 * dstStride;
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[8]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[8]);

        set_va_reg_sb(VA4, srcUb1);
        set_va_reg_sb(VA5, &srcUb1[8]);
        set_va_reg_sb(VA6, tmpUb1);
        set_va_reg_sb(VA7, &tmpUb1[8]);
        if (numSubTileX == 1) {
            Op::TransB8Instr(1, 0, 0);
        } else {
            Op::TransB8Instr(numSubTileX, 32 * dstStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem, unsigned dstStride>
PTO_INTERNAL void CopyB32Tail(__ubuf__ T *dstPtr, __ubuf__ T *tmpPtr, unsigned tmpStride, unsigned validRow,
    unsigned validCol, unsigned remain_y) {
    if (remain_y > 0) {
        if (validCol > REPEAT_MAX) {
            uint16_t lenBurst = dstStride / blockSizeElem;
            uint16_t srcGap = (tmpStride - dstStride) / blockSizeElem;
            copy_ubuf_to_ubuf(dstPtr + (validRow - remain_y), tmpPtr, 0, validCol, lenBurst, srcGap, 0);
        } else {
            const uint16_t srcRepeatStride = tmpStride / blockSizeElem;
            constexpr uint16_t dstRepeatStride = dstStride / blockSizeElem;
            SetContMaskByDType<T>(remain_y);
            pipe_barrier(PIPE_V);
            Op::CopyInstr((__ubuf__ uint32_t *)(dstPtr + (validRow - remain_y)), (__ubuf__ uint32_t *)(tmpPtr),
                validCol, dstRepeatStride, srcRepeatStride);
        }
    }
}

template <typename TileData, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
__tf__ PTO_INTERNAL void TTrans(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src,
    typename TileData::TileDType __in__ tmp, unsigned validRow, unsigned validCol) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    int numSubTileX = (validCol + blockSizeElem - 1) / blockSizeElem;
    int numSubTileY = validRow / yTileSizeElem;
    int remain_y = validRow % yTileSizeElem;

    const unsigned tmpStride = (sizeof(T) == 1) ? (remain_y + 31) / 32 * 32 : (remain_y + 15) / 16 * 16;

    if constexpr (sizeof(T) == 4) { // b32
        TransB32FullSubTiles<TransOp<T>, T, blockSizeElem, dstStride, srcStride>(
            dstPtr, srcPtr, numSubTileX, numSubTileY);
        TransB32YTailTiles<TransOp<T>, T, blockSizeElem, srcStride>(
            tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remain_y);
        CopyB32Tail<TransOp<T>, T, blockSizeElem, dstStride>(dstPtr, tmpPtr, tmpStride, validRow, validCol, remain_y);
    } else if constexpr (sizeof(T) == 2) { // b16
        TransB16FullSubTiles<TransOp<T>, T, blockSizeElem, dstStride, srcStride>(
            dstPtr, srcPtr, numSubTileX, numSubTileY);
        TransB16YTailTiles<TransOp<T>, T, blockSizeElem, dstStride, srcStride>(
            dstPtr, srcPtr, numSubTileX, numSubTileY, remain_y);
    } else if constexpr (sizeof(T) == 1) { // b8
        TransB8FullSubTiles<TransOp<T>, T, blockSizeElem, dstStride, srcStride>(
            dstPtr, srcPtr, numSubTileX, numSubTileY);
        TransB8YTailTiles<TransOp<T>, T, blockSizeElem, dstStride, srcStride>(
            dstPtr, srcPtr, numSubTileX, numSubTileY, remain_y);
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TTRANS: Invalid data type.");
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp) {
    using TS = typename TileDataSrc::DType;
    using TD = typename TileDataDst::DType;
    static_assert(sizeof(TS) == sizeof(TD), "TTRANS: Inconsistent input and output data types.");
    static_assert(TileDataSrc::isRowMajor, "TTRANS: not supported Layout type.");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(TS);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    TTrans<TileDataSrc, blockSizeElem, dstStride, srcStride>(dst.data(), src.data(), tmp.data(), validRow, validCol);
}
} // namespace pto
#endif