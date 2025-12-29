/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINSPLUS_HPP
#define TBINSPLUS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
namespace pto
{
  template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned srcStride>
  PTO_INTERNAL void BinSPlusHead(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned rptPerLine) {
    if (rptPerLine > 0) {
      unsigned numLoop = rptPerLine / REPEAT_MAX;
      unsigned remainAfterLoop = rptPerLine % REPEAT_MAX;
      for (int i = 0; i < validRow; i++) {
        if (numLoop) [[unlikely]] {
          for (int j = 0; j < numLoop; j++) {
            unsigned dstOffset = i * dstStride + j * elemPerRpt * REPEAT_MAX;
            unsigned srcOffset = i * srcStride + j * elemPerRpt * REPEAT_MAX;
            Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, REPEAT_MAX);
          }
        }
        if (remainAfterLoop) {
          unsigned dstOffset = i * dstStride + numLoop * elemPerRpt * REPEAT_MAX;
          unsigned srcOffset = i * srcStride + numLoop * elemPerRpt * REPEAT_MAX;
          Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, remainAfterLoop);
        }
      }
    }
  }

  template <typename Op, typename T, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned srcStride>
  PTO_INTERNAL void BinSPlusTail(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned remain) {
    const bool strideOverFlag = ((dstStride / elemPerBlk > REPEAT_STRIDE_MAX) ||
                                  (srcStride / elemPerBlk > REPEAT_STRIDE_MAX));
    unsigned dstOffset = 0;
    unsigned srcOffset = 0;
    SetContMaskByDType<T>(remain);
    if constexpr (strideOverFlag) {
      for (int i = 0; i < validRow; i++) {
        Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, 1, 1, 1);
        dstOffset += dstStride;
        srcOffset += srcStride;
      }
    } else {
      unsigned numLoop = validRow / REPEAT_MAX;
      unsigned remainAfterLoop = validRow % REPEAT_MAX;
      constexpr uint8_t dstRptStride = dstStride / elemPerBlk;
      constexpr uint8_t srcRptStride = srcStride / elemPerBlk;
      for (int i = 0; i < numLoop; i++) {
        Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, REPEAT_MAX, dstRptStride, srcRptStride);
        dstOffset += REPEAT_MAX * dstStride;
        srcOffset += REPEAT_MAX * srcStride;
      }

      if (remainAfterLoop) {
        Op::BinSInstr(dst + dstOffset, src0 + srcOffset, src1, remainAfterLoop, dstRptStride, srcRptStride);
      }
    }
    SetFullVecMaskByDType<T>();
  }

  template <typename Op, typename T, typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void BinSPlusRowRpt(__ubuf__ T *dst, __ubuf__ T *src, T scalar,
    unsigned validRow, unsigned validCol) {

    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr unsigned elemPerBlk = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcRptStride = srcStride / elemPerBlk;
    constexpr unsigned dstRptStride = dstStride / elemPerBlk;
    constexpr bool condRowRpt = ((TileDataDst::Rows <= pto::REPEAT_MAX) && (dstRptStride <= REPEAT_STRIDE_MAX) &&
                                 (TileDataSrc::Rows <= pto::REPEAT_MAX) && (srcRptStride <= REPEAT_STRIDE_MAX));
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    unsigned offset = 0;
    if constexpr (condRowRpt) {
      for (unsigned i = 0; i < rptPerLine; i++) {
        Op::BinSInstr(dst + offset, src + offset, scalar, validRow, dstRptStride, srcRptStride);
        offset += elemPerRpt;
      }

      if (remain) {
        SetContMaskByDType<T>(remain);
        Op::BinSInstr(dst + offset, src + offset, scalar, validRow, dstRptStride, srcRptStride);
        SetFullVecMaskByDType<T>(); 
      }
    } else {
      BinSPlusHead<Op, T, elemPerRpt, elemPerBlk, dstStride, srcStride>(dst, src, scalar, validRow, rptPerLine);
      offset = rptPerLine * elemPerRpt;
      dst += offset;
      src += offset;
      if (remain) {
        BinSPlusTail<Op, T, elemPerRpt, elemPerBlk, dstStride, srcStride>(dst, src, scalar, validRow, remain);
      }
    }
  }
  template <typename Op, typename T, typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TBinSPlusInstr(__ubuf__ T *dst, __ubuf__ T *src, T scalar, unsigned validRow, unsigned validCol) {
    BinSPlusRowRpt<Op, T, TileDataDst, TileDataSrc>(dst, src, scalar, validRow, validCol);
  }
} //namespace pto
#endif
