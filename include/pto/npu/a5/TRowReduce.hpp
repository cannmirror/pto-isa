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
  return;
}

template <typename ReduceOp, typename TileDataOut, typename TileDataIn,
          VFImplKind VFK>
__tf__ AICORE void TRowReduceImpl(typename TileDataOut::TileDType __out__ dst,
                                  typename TileDataIn::TileDType __in__ src,
                                  uint32_t rows, uint32_t cols) {
  using TIN = typename TileDataIn::DType;
  __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
  __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);
  constexpr uint32_t elementsPerRepeat =
      std::is_same_v<TIN, float> ? ELE_CNT_B32 : ELE_CNT_B16;
  __VEC_SCOPE__ {
    RegTensor<TIN> vreg0;
    RegTensor<TIN> vreg1;
    RegTensor<TIN> vregdst;
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    constexpr auto distValue =
        std::integral_constant<::DistVST,
                               static_cast<::DistVST>(GetDistVst<TIN, DistVST::DIST_ONEPT>())>();
    uint32_t destItems = 1;
    MaskReg pregdst = CreatePredicate<TIN>(destItems);
    if constexpr (VFK == VFIMPL_2D_NO_POST_UPDATE) {
      for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
        vbr(vregdst, ReduceOp::InitVal);
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
          MaskReg preg = CreatePredicate<TIN>(sreg);
          vlds(vreg0, srcPtr + i * TileDataIn::RowStride, j * elementsPerRepeat, NORM);
          ReduceOp::Reduce(vreg1, vreg0, preg);
          ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
        }
        vsts(vregdst, dstPtr + i * TileDataOut::RowStride, 0, distValue, pregdst);
      }
    } else {
      static_assert(VFK == VFIMPL_2D_POST_UPDATE,
                    "VFImplKind value not expected.");
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

template <typename ReduceOp, typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRowReduce(TileDataOut &dst, TileDataIn &src) {
  TRowReduceCheck<TileDataOut, TileDataIn>();
  TRowReduceImpl<ReduceOp, TileDataOut, TileDataIn, VFIMPL_2D_NO_POST_UPDATE>(
      dst.data(), src.data(), src.GetValidRow(), src.GetValidCol());
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src,
                               TileDataTmp &tmp) {
  using rowReduceOp = ROWMAX<typename TileDataIn::DType>;
  TRowReduce<rowReduceOp, TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src,
                               TileDataTmp &tmp) {
  using rowReduceOp = ROWSUM<typename TileDataIn::DType>;
  TRowReduce<rowReduceOp, TileDataOut, TileDataIn>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src,
                               TileDataTmp &tmp) {
  using rowReduceOp = ROWMIN<typename TileDataIn::DType>;
  TRowReduce<rowReduceOp, TileDataOut, TileDataIn>(dst, src);
}
} // namespace pto

#endif