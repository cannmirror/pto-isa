/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINPLUS_HPP
#define TBINPLUS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
  template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk,
    unsigned dstStride, unsigned src0Stride, unsigned src1Stride>
  PTO_INTERNAL void Bin2LNormModeHead(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    unsigned validRow, unsigned rptPerLine) {
    if (rptPerLine > 0) {
      unsigned numLoop = rptPerLine / REPEAT_MAX;
      unsigned remainAfterLoop = rptPerLine % REPEAT_MAX;
      for (int i = 0; i < validRow; i++) {
        if (numLoop) [[unlikely]] {
          for (int j = 0; j < numLoop; j++) {
            unsigned dstOffset = i * dstStride + j * elemPerRpt * REPEAT_MAX;
            unsigned src0Offset = i * src0Stride + j * elemPerRpt * REPEAT_MAX;
            unsigned src1Offset = i * src1Stride + j * elemPerRpt * REPEAT_MAX;
            Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, REPEAT_MAX);
          }
        }
        if (remainAfterLoop) {
          unsigned offset = i * dstStride + numLoop * elemPerRpt * REPEAT_MAX;
          unsigned src0Offset = i * src0Stride + numLoop * elemPerRpt * REPEAT_MAX;
          unsigned src1Offset = i * src1Stride + numLoop * elemPerRpt * REPEAT_MAX;
          Op::BinInstr(dst + offset, src0 + src0Offset, src1 + src1Offset, remainAfterLoop);
        }
      }
    }
  }

  template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk,
    unsigned dstStride, unsigned src0Stride, unsigned src1Stride>
  PTO_INTERNAL void Bin2LNormModeTail(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    unsigned validRow, unsigned remain) {
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    constexpr bool src0StrideOverFlag = (src0Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool src1StrideOverFlag = (src1Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool dstStrideOverFlag = (dstStride / elemPerBlk > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(remain);
    for (int i = 0; i < numLoop; i++) {
      if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
          unsigned src0Offset = i * REPEAT_MAX * src0Stride + j * src0Stride;
          unsigned src1Offset = i * REPEAT_MAX * src1Stride + j * src1Stride;
          unsigned dstOffset = i * REPEAT_MAX * dstStride + j * dstStride;
          Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, 1, 1, 1, 1);
        }
      } else {
        unsigned src0Offset = i * REPEAT_MAX * src0Stride;
        unsigned src1Offset = i * REPEAT_MAX * src1Stride;
        unsigned dstOffset = i * REPEAT_MAX * dstStride;
        uint8_t src0BlkPerLine = src0Stride / elemPerBlk;
        uint8_t src1BlkPerLine = src1Stride / elemPerBlk;
        uint8_t dstBlkPerLine = dstStride / elemPerBlk;
        Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, REPEAT_MAX, dstBlkPerLine,
          src0BlkPerLine, src1BlkPerLine);
      }
    }
    remainAfterLoop = validRow % REPEAT_MAX;
    if (remainAfterLoop) {
      if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
        for (unsigned j = 0; j < remainAfterLoop; j++) {
          unsigned src0Offset = numLoop * REPEAT_MAX * src0Stride + j * src0Stride;
          unsigned src1Offset = numLoop * REPEAT_MAX * src1Stride + j * src1Stride;
          unsigned dstOffset = numLoop * REPEAT_MAX * dstStride + j * dstStride;
          Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, 1, 1, 1, 1);
        }
      } else {
        unsigned dstOffset = numLoop * REPEAT_MAX * dstStride;
        unsigned src0Offset = numLoop * REPEAT_MAX * src0Stride;
        unsigned src1Offset = numLoop * REPEAT_MAX * src1Stride;
        uint8_t dstBlkPerLine = dstStride / elemPerBlk;
        uint8_t src0BlkPerLine = src0Stride / elemPerBlk;
        uint8_t src1BlkPerLine = src1Stride / elemPerBlk;
        Op::BinInstr(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, remainAfterLoop,
          dstBlkPerLine, src0BlkPerLine, src1BlkPerLine);
      }
    }
    SetFullVecMaskByDType<T>();
  }

  template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
    unsigned src1Stride>
  PTO_INTERNAL void Bin2LNormModeRowRpt(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    unsigned validRow, unsigned validCol) {
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    Bin2LNormModeHead<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>
      (dst, src0, src1, validRow, rptPerLine);
    if (remain) {
      unsigned offset = rptPerLine * elemPerRpt;
      dst += offset;
      src0 += offset;
      src1 += offset;
      Bin2LNormModeTail<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>
        (dst, src0, src1, validRow, remain);
    }
  }

  template <typename Op, typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
  PTO_INTERNAL void BinaryPlusInstr(__ubuf__ typename TileDataDst::DType *dst,
    __ubuf__ typename TileDataSrc0::DType *src0, __ubuf__ typename TileDataSrc1::DType *src1, unsigned validRow,
    unsigned validCol) {
    constexpr unsigned elemPerRpt = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned elemPerBlk = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned src0Stride = TileDataSrc0::RowStride;
    constexpr unsigned src1Stride = TileDataSrc1::RowStride;

    Bin2LNormModeRowRpt<Op, T, elemPerRpt, elemPerBlk, dstStride, src0Stride, src1Stride>
      (dst, src0, src1, validRow, validCol);
  }
} // namespace pto
#endif