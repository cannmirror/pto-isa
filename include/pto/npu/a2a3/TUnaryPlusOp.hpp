/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TUNARYPLUSOP_HPP
#define TUNARYPLUSOP_HPP

#include <pto/common/constants.hpp>
#include "pto/npu/a2a3/TUnaryOp.hpp"

namespace pto {
  template <typename Op, typename T, unsigned elemPerRpt, unsigned dstRowStride, unsigned srcRowStride>
  PTO_INTERNAL void UnaryPlusHead(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned rptPerLine) {
    if (rptPerLine) {
      unsigned numLoop = rptPerLine / REPEAT_MAX;
      unsigned remain = rptPerLine % REPEAT_MAX;
      for (unsigned i = 0; i < validRow; i++) {
        if (numLoop) {
          for (unsigned j = 0; j < numLoop; j++) {
            unsigned dstOffset = i * dstRowStride + j * elemPerRpt * REPEAT_MAX;
            unsigned srcOffset = i * srcRowStride + j * elemPerRpt * REPEAT_MAX;
            Op::UnaryInstr(dst + dstOffset, src + srcOffset, REPEAT_MAX);
          }
        }
        if (remain) {
          unsigned dstOffset = i * dstRowStride + numLoop * elemPerRpt * REPEAT_MAX;
          unsigned srcOffset = i * srcRowStride + numLoop * elemPerRpt * REPEAT_MAX;
          Op::UnaryInstr(dst + dstOffset, src + srcOffset, remain);
        }
      }
    }
  }

  template <typename Op, typename T, unsigned rows, unsigned elemPerBlk, unsigned dstRowStride, unsigned srcRowStride>
  PTO_INTERNAL void UnaryPlusTail(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned remainElem) {
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    constexpr uint8_t dstRptStride = dstRowStride / elemPerBlk;
    constexpr uint8_t srcRptStride = srcRowStride / elemPerBlk;
    constexpr bool strideOverFlag = ((dstRowStride / elemPerBlk > REPEAT_STRIDE_MAX) &&
                                     (srcRowStride / elemPerBlk > REPEAT_STRIDE_MAX));
    unsigned dstOffset;
    unsigned srcOffset;
    SetContMaskByDType<T>(remainElem);
    for (uint32_t i = 0; i < numLoop; i++) {
      if constexpr (strideOverFlag) {
        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
          dstOffset = i * REPEAT_MAX * dstRowStride + j * dstRowStride;
          srcOffset = i * REPEAT_MAX * srcRowStride + j * srcRowStride;
          Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, 1, 1, 1);
        }
      } else {
        dstOffset = i * REPEAT_MAX * dstRowStride;
        srcOffset = i * REPEAT_MAX * srcRowStride;
        Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, REPEAT_MAX, dstRptStride, srcRptStride);
      }
    }

    if (remainAfterLoop) {
      if constexpr (strideOverFlag) {
        for (uint32_t j = 0; j < remainAfterLoop; j++) {
          dstOffset = numLoop * REPEAT_MAX * dstRowStride + j * dstRowStride;
          srcOffset = numLoop * REPEAT_MAX * srcRowStride + j * srcRowStride;
          Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, 1, 1, 1);
        }
      } else {
        dstOffset = numLoop * REPEAT_MAX * dstRowStride;
        srcOffset = numLoop * REPEAT_MAX * srcRowStride;
        Op::UnaryInstr(dstPtr + dstOffset, srcPtr + srcOffset, remainAfterLoop, dstRptStride, srcRptStride);
      }
    }
    SetFullVecMaskByDType<T>();
  }

  template <typename T, typename Op, typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TUnaryPlusInstr(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol) {
    constexpr unsigned elemPerBlk = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = dstRowStride / elemPerBlk;
    constexpr unsigned srcStride = srcRowStride / elemPerBlk;
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    unsigned offset = 0;
    constexpr bool condRowRpt = ((TileDataDst::Rows <= pto::REPEAT_MAX) && (dstStride <= REPEAT_STRIDE_MAX) &&
                                 (TileDataSrc::Rows <= pto::REPEAT_MAX) && (srcStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
      for (uint32_t i = 0; i < rptPerLine; i++) {
        Op::UnaryInstr(dst + offset, src + offset, validRow, dstStride, srcStride);
        offset += elemPerRpt;
      }

      if (remain) {
        SetContMaskByDType<T>(remain);
        Op::UnaryInstr(dst + offset, src + offset, validRow, dstStride, srcStride);
        SetFullVecMaskByDType<T>();
      }
    } else {
      UnaryPlusHead<Op, T, elemPerRpt, dstRowStride, srcRowStride>(dst, src, validRow, rptPerLine);
      offset = rptPerLine * elemPerRpt;
      dst += offset;
      src += offset;
      if (remain) {
        UnaryPlusTail<Op, T, elemPerRpt, elemPerBlk, dstRowStride>(dst, src, validRow, remain);
      }
    }
  }

  template <typename TileDataDst, typename TileDataSrc, unaryFuncPtr<typename TileDataDst::DType> func,
    typename T = typename TileDataDst::DType>
  __tf__ PTO_INTERNAL void TUnaryPlusOp(typename TileDataDst::TileDType __out__ dstData,
    typename TileDataSrc::TileDType __in__ srcData, unsigned validRow, unsigned validCol) {
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    TUnaryPlusInstr<T, UnaryOperation<T, func>, TileDataDst, TileDataSrc>(dst, src, validRow, validCol);
  }

  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TUnaryPlusStaticCheck() {
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc::DType>,
                  "TUnaryPlusStaticCheck: The data type of dst must be consistent with src.");

    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor,
                  "TUnaryPlusStaticCheck: The src and dst Tile only support row major layout.");

    static_assert(std::is_same_v<T, float32_t> || std::is_same_v<T, float> ||
                  std::is_same_v<T, half> || std::is_same_v<T, float16_t>,
                  "TUnaryPlusStaticCheck: Invalid data type");

    static_assert(TileDataDst::Loc == TileType::Vec,
                  "TUnaryPlusStaticCheck: TileType of src and dst tiles must be TileType::Vec.");

    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "TUnaryPlusStaticCheck: Number of valid columns must not be greater than number of tile columns.");

    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "TUnaryPlusStaticCheck: Number of valid rows must not be greater than number of tile rows.");
  }

  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TRSQRT_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TRSQRT: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TRSQRT: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        constexpr unsigned rowStride = TileDataDst::RowStride;
#ifdef ACCURATE_RSQRT
        TRsqrtCustom<TileDataDst>(dst.data(), src.data(), dstValidRow, dstValidCol);
#else
        TUnaryOp<TileDataDst, _vrsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
#endif
    } else {
        TUnaryPlusOp<TileDataDst, TileDataSrc, _vrsqrt>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }

  /* SQRT */
  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TSQRT_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TSQRT: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TSQRT: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        constexpr unsigned rowStride = TileDataDst::RowStride;
        TUnaryOp<TileDataDst, _vsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
    } else {
        TUnaryPlusOp<TileDataDst, TileDataSrc, _vsqrt>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }

  /* ABS */
  template <typename TileDataDst, typename TileDataSrc>
  AICORE void TABS_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TABS: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TABS: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
      constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
      constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
      constexpr unsigned rowStride = TileDataDst::RowStride;
      TUnaryOp<TileDataDst, _vabs, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
    } else {
      TUnaryPlusOp<TileDataDst, TileDataSrc, _vabs>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }

  /* EXP */
  template <typename TileDataDst, typename TileDataSrc>
  PTO_INTERNAL void TEXP_IMPL(TileDataDst &dst, TileDataSrc &src) {
    TUnaryPlusStaticCheck<TileDataDst, TileDataSrc>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TEXP: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TEXP: Number of columns of src and dst must be the same.");
    if constexpr (std::is_same_v<TileDataDst, TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
        constexpr unsigned rowStride = TileDataDst::RowStride;
        TUnaryOp<TileDataDst, _vexp, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), dstValidRow, dstValidCol);
    } else {
        TUnaryPlusOp<TileDataDst, TileDataSrc, _vexp>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
  }
}

#endif