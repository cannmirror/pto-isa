/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __ROW_REDUCE__
#define __ROW_REDUCE__

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

using namespace std;

template <typename T> struct ROWSUM {
  static constexpr T InitVal = 0;
  using RegType = typename TypeGet<T>::T;
  static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0,
                                      RegType &src1, MaskReg &pred) {
    vadd(dst, src0, src1, pred, MODE_ZEROING);
  }
  static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred) {
    vcadd(dst, src, pred, MODE_ZEROING);
  }
};

template <typename T> struct ROWMAX {
  static constexpr T InitVal = -INFINITY;
  using RegType = typename TypeGet<T>::T;
  static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0,
                                      RegType &src1, MaskReg &pred) {
    vmax(dst, src0, src1, pred, MODE_ZEROING);
  }
  static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred) {
    vcmax(dst, src, pred, MODE_ZEROING);
  }
};

template <typename T> struct ROWMIN {
  static constexpr T InitVal = INFINITY;
  using RegType = typename TypeGet<T>::T;
  static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0,
                                      RegType &src1, MaskReg &pred) {
    vmin(dst, src0, src1, pred, MODE_ZEROING);
  }
  static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred) {
    vcmin(dst, src, pred, MODE_ZEROING);
  }
};

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRowReduceCheck() {
  using T = typename TileDataIn::DType;
  static_assert(is_same_v<T, half> || is_same_v<T, float>,
                "Only support half and float.");
  static_assert(is_same_v<T, typename TileDataOut::DType>,
                "Output type must be same as input type.");
  static_assert(TileDataOut::Loc == pto::TileType::Vec &&
                    TileDataIn::Loc == pto::TileType::Vec,
                "This instruction only support Vec Tile.");
  static_assert(TileDataIn::isRowMajor && !TileDataIn::isBoxedLayout,
                "This instruction only support ND layout for input tile.");
  static_assert(
      (!TileDataOut::isBoxedLayout &&
       (TileDataOut::isRowMajor ||
        (!TileDataOut::isRowMajor && TileDataOut::Cols == 1))),
      "This instruction only support ND or DN with Col = 1 for output tile.");
  return;
}

template <typename ReduceOp, typename TileDataOut, typename TileDataIn, 
          unsigned elementsPerRepeat>
PTO_INTERNAL void TRowReduceImpl(__ubuf__ typename TileDataOut::DType *dstPtr,
                                 __ubuf__ typename TileDataOut::DType *srcPtr,
                                  uint32_t rows, uint32_t cols, unsigned version) {
  using TIN = typename TileDataIn::DType;
  uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
  __VEC_SCOPE__ {
    RegTensor<TIN> vreg0;
    RegTensor<TIN> vreg1;
    RegTensor<TIN> vregdst;
    constexpr auto distValue =
        std::integral_constant<::DistVST,
                               static_cast<::DistVST>(GetDistVst<TIN, DistVST::DIST_ONEPT>())>();
    uint32_t destItems = 1;
    MaskReg pregdst = CreatePredicate<TIN>(destItems);
    if (version == VFIMPL_2D_NO_POST_UPDATE) {
      for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
        vbr(vregdst, ReduceOp::InitVal);
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
          MaskReg preg = CreatePredicate<TIN>(sreg);
          vlds(vreg0, srcPtr,  i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
          ReduceOp::Reduce(vreg1, vreg0, preg);
          ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
        }
        vsts(vregdst, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
      }
    } else {
      for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
        vbr(vregdst, ReduceOp::InitVal);
        __ubuf__ TIN *row_ptr = srcPtr + i * TileDataIn::RowStride;
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
          MaskReg preg = CreatePredicate<TIN>(sreg);
          vlds(vreg0, row_ptr, elementsPerRepeat, NORM, POST_UPDATE);
          ReduceOp::Reduce(vreg1, vreg0, preg);
          ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
        }
        vsts(vregdst, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
      }
    }
  } // end VF
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMAX) OP_TYPE(reduce)
void TRowMax(typename TileDataOut::TileDType __out__ dst,
             typename TileDataIn::TileDType __in__ src,
             uint32_t rows, uint32_t cols, unsigned version = VFImplKind::VFIMPL_DEFAULT) {
  TRowReduceCheck<TileDataOut, TileDataIn>();

  using TIN = typename TileDataIn::DType;
  __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
  __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

  using rowReduceOp = ROWMAX<typename TileDataIn::DType>;
  TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(
      dstPtr, srcPtr, rows, cols, version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWSUM) OP_TYPE(reduce)
void TRowSum(typename TileDataOut::TileDType __out__ dst,
             typename TileDataIn::TileDType __in__ src,
             uint32_t rows, uint32_t cols, unsigned version = VFImplKind::VFIMPL_DEFAULT) {
  TRowReduceCheck<TileDataOut, TileDataIn>();

  using TIN = typename TileDataIn::DType;
  __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
  __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

  using rowReduceOp = ROWSUM<typename TileDataIn::DType>;
  TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(
      dstPtr, srcPtr, rows, cols, version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMIN) OP_TYPE(reduce)
void TRowMin(typename TileDataOut::TileDType __out__ dst,
             typename TileDataIn::TileDType __in__ src,
             uint32_t rows, uint32_t cols, unsigned version = VFImplKind::VFIMPL_DEFAULT) {
  TRowReduceCheck<TileDataOut, TileDataIn>();

  using TIN = typename TileDataIn::DType;
  __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
  __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

  using rowReduceOp = ROWMIN<typename TileDataIn::DType>;
  TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(
      dstPtr, srcPtr, rows, cols, version);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src,
                               TileDataTmp &tmp) {
  using T = typename TileDataIn::DType;
  constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
  unsigned rows = src.GetValidRow();
  unsigned cols = src.GetValidCol();

  TRowMax<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), rows, cols);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src,
                               TileDataTmp &tmp) {
  using T = typename TileDataIn::DType;
  constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
  unsigned rows = src.GetValidRow();
  unsigned cols = src.GetValidCol();

  TRowSum<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), rows, cols);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src,
                               TileDataTmp &tmp) {
  using T = typename TileDataIn::DType;
  constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
  unsigned rows = src.GetValidRow();
  unsigned cols = src.GetValidCol();

  TRowMin<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), rows, cols);
}
} // namespace pto

#endif