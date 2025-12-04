/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWSUM_HPP
#define TROWSUM_HPP

#include <common/utils.hpp>
#include <common/type.hpp>
#include <common/pto_tile.hpp>
#include <common/tile_tensor_impl.hpp>

namespace pto
{

#define B16_REPEAT_MAX 65535

template <typename T, bool cntModeEn, int cols, uint32_t dstRepeatStride, uint32_t srcRepeatStride,
    uint8_t nElemPerRepeat>
__aicore__ PTO_INLINE void VcaddByMode(__ubuf__ T *dst, __ubuf__ T *src, unsigned repeatTimes) {
    if constexpr (dstRepeatStride > B16_REPEAT_MAX) {
        for (int i = 0; i < repeatTimes; i++) {
            vcadd(dst + i * dstRepeatStride, src + i * cols, 1, 0, 1, 0, false);
        }
    } else if (cntModeEn) {
        set_mask_count();
        set_vector_mask(0, (uint32_t)repeatTimes * nElemPerRepeat);
        vcadd(dst, src, 0, dstRepeatStride, 1, srcRepeatStride, false);
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        vcadd(dst, src, repeatTimes, dstRepeatStride, 1, srcRepeatStride, false);
    }
}

template <typename T, bool cntModeEn, int dstCols, int src0Cols, int src1Cols,
    uint32_t dstRepeatStride, uint32_t src0RepeatStride, uint32_t src1RepeatStride, uint8_t nElemPerRepeat>
__aicore__ PTO_INLINE void VaddByMode(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned repeatTimes)
{
    if constexpr (dstRepeatStride > REPEAT_MAX || src0RepeatStride > REPEAT_MAX || src1RepeatStride > REPEAT_MAX) {
        for (int i = 0; i < repeatTimes; i++) {
            vadd(dst + i * dstCols, src0 + i * src0Cols, src1 + i * src1Cols, 1, 1, 1, 1, 0, 0, 0);
        }
    } else if (cntModeEn) {
        set_mask_count();
        set_vector_mask(0, repeatTimes * nElemPerRepeat);
        vadd(dst, src0, src1, 0, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        vadd(dst, src0, src1, repeatTimes, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
}

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp, uint8_t nElemPerRepeat,
    uint32_t dstRepeatStride, uint32_t srcRepeatStride, uint32_t tmpRepeatStride>
__tf__ __aicore__ PTO_INLINE void TRowSum(typename TileDataOut::TileDType __out__ dstData,
                                          typename TileDataIn::TileDType __in__ srcData,
                                          typename TileDataTmp::TileDType __in__ tmpData, int validCol, int validRow) {
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);

    int srcRepeatPerRow = validCol / nElemPerRepeat;    // src一行满足repeat个数
    int remain = validCol % nElemPerRepeat;             // src一行repeat之后剩余多少元素

    // 需要处理的行若超过uint8, 则拆分为多次进行循环
    int rowRepeatTimes = validRow / REPEAT_MAX;
    unsigned repeatTimes;
    __ubuf__ T *dstP = dst;
    __ubuf__ T *srcP = src;
    __ubuf__ T *tmpP = tmp;

    if (validCol == nElemPerRepeat) {
        VcaddByMode<T, true, TileDataIn::Cols, dstRepeatStride, srcRepeatStride, nElemPerRepeat>(dst, src, validRow);
        pipe_barrier(PIPE_V);
        return;
    }

    if (validCol < nElemPerRepeat) {
        SetContinuousMask(remain);
        do {
            repeatTimes = rowRepeatTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
            pipe_barrier(PIPE_V);
            VcaddByMode<T, false, TileDataIn::Cols, dstRepeatStride, srcRepeatStride,
                nElemPerRepeat>(dstP, srcP, repeatTimes);
            rowRepeatTimes -= 1;
            dstP += repeatTimes * TileDataOut::Cols;
            srcP += repeatTimes * TileDataIn::Cols;
            tmpP += repeatTimes * TileDataTmp::Cols;
        } while (rowRepeatTimes >= 0);

        set_vector_mask(-1, -1);
        return;
    }

    if (validCol < 2 * nElemPerRepeat) {
        // 解决 ccec 编译检查问题； 如果删除会导致copy_ubuf_to_ubuf编译错误，提醒第六、七个参数的范围必须是[0, 65535]
        if constexpr ((srcRepeatStride < BLOCK_MAX_PER_REPEAT) || (tmpRepeatStride < BLOCK_MAX_PER_REPEAT)) {
            return;
        }
        // 将满足一次repeat部分copy到dst
        copy_ubuf_to_ubuf(tmp, src, 0, validRow, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT,
            tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
        pipe_barrier(PIPE_V);
    }

    int i;
    // 二分Add, 将每行相邻的两个repeat相加存入tmp
    for (i = 0; i < srcRepeatPerRow / 2; i++) {
        VaddByMode<T, true, TileDataTmp::Cols, TileDataIn::Cols, TileDataIn::Cols,
            tmpRepeatStride, srcRepeatStride, srcRepeatStride, nElemPerRepeat>(tmp + i * nElemPerRepeat,
            src + (i * 2) * nElemPerRepeat, src + (i * 2 + 1) * nElemPerRepeat, validRow);
        pipe_barrier(PIPE_V);
    }
    // 若repeat为奇数, 则将最后的repeat加入tmp
    if (srcRepeatPerRow != 1 && srcRepeatPerRow % 2 == 1) {
        VaddByMode<T, true, TileDataTmp::Cols, TileDataTmp::Cols, TileDataIn::Cols,
            tmpRepeatStride, tmpRepeatStride, srcRepeatStride, nElemPerRepeat>(tmp, tmp,
            src + (srcRepeatPerRow - 1) * nElemPerRepeat, validRow);
        pipe_barrier(PIPE_V);
    }

    unsigned curLen;
    unsigned loopRemain;
    unsigned repeatOffset;
    do {
        repeatTimes = rowRepeatTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
        curLen = srcRepeatPerRow;

        // 若存在剩余为奇数, 则将最后的repeat加入tmp
        if (remain > 0) {
            // 将remain加入temp
            repeatOffset = curLen == 1 ? 0 : (curLen / 2 - 1);
            SetContinuousMask(remain);
            VaddByMode<T, false, TileDataTmp::Cols, TileDataTmp::Cols, TileDataIn::Cols,
                tmpRepeatStride, tmpRepeatStride, srcRepeatStride, nElemPerRepeat>(tmpP + repeatOffset * nElemPerRepeat,
                tmpP + repeatOffset * nElemPerRepeat, srcP + curLen * nElemPerRepeat, repeatTimes);
            pipe_barrier(PIPE_V);
            set_vector_mask(-1, -1);
        }
        // 二分Add后的repeat数
        curLen = curLen / 2;
        while (curLen > 1) {
            for (i = 0; i < curLen / 2; i++) {
                VaddByMode<T, false, TileDataTmp::Cols, TileDataTmp::Cols, TileDataTmp::Cols,
                    tmpRepeatStride, tmpRepeatStride, tmpRepeatStride, nElemPerRepeat>(tmpP + i * nElemPerRepeat,
                    tmpP + i * 2 * nElemPerRepeat, tmpP + (i * 2 + 1) * nElemPerRepeat, repeatTimes);
                pipe_barrier(PIPE_V);
            }

            loopRemain = curLen % 2;
            curLen = curLen / 2;
            if (loopRemain > 0) {
                VaddByMode<T, false, TileDataTmp::Cols, TileDataTmp::Cols, TileDataTmp::Cols, tmpRepeatStride,
                    tmpRepeatStride, tmpRepeatStride, nElemPerRepeat>(tmpP + (curLen - 1) * nElemPerRepeat,
                    tmpP + (curLen - 1) * nElemPerRepeat, tmpP + curLen * 2 * nElemPerRepeat, repeatTimes);
                pipe_barrier(PIPE_V);
            }
        }
        VcaddByMode<T, false, TileDataTmp::Cols, dstRepeatStride, tmpRepeatStride,
            nElemPerRepeat>(dstP, tmpP, repeatTimes);
        pipe_barrier(PIPE_V);

        rowRepeatTimes -= 1;
        dstP += repeatTimes * TileDataOut::Cols;
        srcP += repeatTimes * TileDataIn::Cols;
        tmpP += repeatTimes * TileDataTmp::Cols;
    } while (rowRepeatTimes > 0);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__aicore__ PTO_INLINE void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp) {
    using T = typename TileDataIn::DType;
    constexpr bool isTargetType = std::is_same_v<T, half> || std::is_same_v<T, float>;
    static_assert(isTargetType, "The input data type is not supported by this instruction.");

    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    if (validCol == 0 || validRow == 0) {
        return;
    }
    constexpr uint8_t nElemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint8_t nElemPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t dstRepeatStride = TileDataOut::Cols;
    constexpr uint32_t srcRepeatStride = TileDataIn::Cols / nElemPerBlock;
    constexpr uint32_t tmpRepeatStride = TileDataTmp::Cols / nElemPerBlock;

    TRowSum<T, TileDataOut, TileDataIn, TileDataTmp, nElemPerRepeat, dstRepeatStride, srcRepeatStride,
        tmpRepeatStride>(dst.data(), src.data(), tmp.data(), validCol, validRow);
}
}
#endif