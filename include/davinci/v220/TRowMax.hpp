/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWMAX_HPP
#define TROWMAX_HPP

#include <common/pto_tile.hpp>
#include <common/tile_tensor_impl.hpp>
#include <common/type.hpp>
#include <utils.hpp>

namespace pto {
#define B16_REPEAT_MAX 65535

template <typename T, bool cntModeEn, int cols, uint32_t dstRepeatStride,
          uint32_t srcRepeatStride, uint8_t nElemPerRepeat>
__aicore__ PTO_INLINE void VcmaxByMode(__ubuf__ T *dst, __ubuf__ T *src,
                                       unsigned repeatTimes) {
  if constexpr (dstRepeatStride > B16_REPEAT_MAX ||
                srcRepeatStride > B16_REPEAT_MAX) {
    for (int i = 0; i < repeatTimes; i++) {
      vcmax(dst + i * dstRepeatStride, src + i * cols, 1, 0, 1, 0, false);
    }
  } else if constexpr (cntModeEn) {
    set_mask_count();
    set_vector_mask(0, (uint32_t)repeatTimes * nElemPerRepeat);
    vcmax(dst, src, 0, dstRepeatStride, 1, srcRepeatStride, ONLY_VALUE);
    set_mask_norm();
    set_vector_mask(-1, -1);
  } else {
    vcmax(dst, src, repeatTimes, dstRepeatStride, 1, srcRepeatStride,
          ONLY_VALUE);
  }
}

template <typename T, bool cntModeEn, int dstCols, int src0Cols, int src1Cols,
          uint32_t dstRepeatStride, uint32_t src0RepeatStride,
          uint32_t src1RepeatStride, uint8_t nElemPerRepeat>
__aicore__ PTO_INLINE void VmaxByMode(__ubuf__ T *dst, __ubuf__ T *src0,
                                      __ubuf__ T *src1, unsigned repeatTimes) {
  if constexpr (dstRepeatStride > REPEAT_MAX || src0RepeatStride > REPEAT_MAX ||
                src1RepeatStride > REPEAT_MAX) {
    for (int i = 0; i < repeatTimes; i++) {
      vmax(dst + i * dstCols, src0 + i * src0Cols, src1 + i * src1Cols, 1, 1, 1,
           1, 0, 0, 0);
    }
  } else if constexpr (cntModeEn) {
    set_mask_count();
    set_vector_mask(0, repeatTimes * nElemPerRepeat);
    vmax(dst, src0, src1, 0, 1, 1, 1, dstRepeatStride, src0RepeatStride,
         src1RepeatStride);
    set_mask_norm();
    set_vector_mask(-1, -1);
  } else {
    vmax(dst, src0, src1, repeatTimes, 1, 1, 1, dstRepeatStride,
         src0RepeatStride, src1RepeatStride);
  }
}

template <typename T, typename TileDataOut, typename TileDataIn,
          typename TileDataTmp, uint8_t nElemPerRepeat,
          uint32_t dstRepeatStride, uint32_t srcRepeatStride, uint32_t tmpRepeatStride>
__tf__ __aicore__ PTO_INLINE void
TRowMax(typename TileDataOut::TileDType __out__ dstData,
        typename TileDataIn::TileDType __in__ srcData,
        typename TileDataTmp::TileDType __in__ tmpData, int validCol, int validRow) {
  __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
  __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
  __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
  int srcRepeatPerRow = validCol / nElemPerRepeat; // src一行满足repeat个数
  int remain = validCol % nElemPerRepeat; // src一行repeat之后剩余多少元素
  int rowRepeatTimes = validRow / REPEAT_MAX;

  if (validCol == nElemPerRepeat) {
    VcmaxByMode<T, true, TileDataIn::Cols, dstRepeatStride, srcRepeatStride, nElemPerRepeat>(dst, src, validRow);
    pipe_barrier(PIPE_V);
    return;
  }

  if (validCol < nElemPerRepeat) {
    SetContinuousMask(remain);
    do {
      unsigned repeatTimes = rowRepeatTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
      VcmaxByMode<T, false, TileDataIn::Cols, dstRepeatStride, srcRepeatStride, nElemPerRepeat>(dst, src, repeatTimes);
      pipe_barrier(PIPE_V);
      rowRepeatTimes -= 1;
      dst += repeatTimes * TileDataOut::Cols;
      src += repeatTimes * TileDataIn::Cols;
      tmp += repeatTimes * TileDataTmp::Cols;
    } while (rowRepeatTimes >= 0);
    set_vector_mask(-1, -1);
    return;
  }

  if (tmpRepeatStride >= BLOCK_MAX_PER_REPEAT && srcRepeatStride >= BLOCK_MAX_PER_REPEAT && validCol < 2 * nElemPerRepeat) {
    // 将满足一次repeat部分copy到dst
    copy_ubuf_to_ubuf(tmp, src, 0, validRow, BLOCK_MAX_PER_REPEAT, srcRepeatStride - BLOCK_MAX_PER_REPEAT, tmpRepeatStride - BLOCK_MAX_PER_REPEAT);
  } else {
    // validCol大于等于2次repeat，将完整的2次repeat比较后写入tmp
    VmaxByMode<T, true, TileDataTmp::Cols, TileDataIn::Cols, TileDataIn::Cols, tmpRepeatStride, srcRepeatStride, srcRepeatStride, nElemPerRepeat>(tmp, src, src + nElemPerRepeat, validRow);
  }
  pipe_barrier(PIPE_V);

  for (int i = 2; i < srcRepeatPerRow; i++) {
    VmaxByMode<T, true, TileDataTmp::Cols, TileDataIn::Cols, TileDataTmp::Cols, tmpRepeatStride, srcRepeatStride, tmpRepeatStride, nElemPerRepeat>(tmp, src + i * nElemPerRepeat, tmp, validRow);
    pipe_barrier(PIPE_V);
  }

  if (remain > 0) {
    SetContinuousMask(remain);
    VmaxByMode<T, false, TileDataTmp::Cols, TileDataIn::Cols, TileDataTmp::Cols, tmpRepeatStride, srcRepeatStride, tmpRepeatStride, nElemPerRepeat>(tmp, src + srcRepeatPerRow * nElemPerRepeat, tmp, validRow);
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_V);
  }
  VcmaxByMode<T, true, TileDataTmp::Cols, dstRepeatStride, tmpRepeatStride, nElemPerRepeat>(dst, tmp, validRow);
  pipe_barrier(PIPE_V);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__aicore__ PTO_INLINE void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src,
                                        TileDataTmp &tmp) {
  using T = typename TileDataIn::DType;
  static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>,
                "Only support half and float.");
  static_assert(std::is_same_v<T, typename TileDataOut::DType>,
                "Output type must be same as input type.");
  int srcValidCol = src.GetValidCol();
  int srcValidRow = src.GetValidRow();
  int dstValidCol = dst.GetValidCol();
  int dstValidRow = dst.GetValidRow();
  constexpr uint8_t nElemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
  constexpr uint8_t nElemPerRepeat = REPEAT_BYTE / sizeof(T);
  constexpr uint32_t dstRepeatStride = TileDataOut::Cols;
  constexpr uint32_t srcRepeatStride = TileDataIn::Cols / nElemPerBlock;
  constexpr uint32_t tmpRepeatStride = TileDataTmp::Cols / nElemPerBlock;

  PTO_ASSERT(srcValidRow == dstValidRow,
             "Valid row of src and dst are not equal");
  PTO_ASSERT(dstValidCol == 1, "Valid col in dst is not equal to 1");
  if (srcValidCol == 0 || srcValidRow == 0) {
    return;
  }

  TRowMax<T, TileDataOut, TileDataIn, TileDataTmp, nElemPerRepeat,
          dstRepeatStride, srcRepeatStride, tmpRepeatStride>(
      dst.data(), src.data(), tmp.data(), srcValidCol, srcValidRow);
}
} // namespace pto

#endif