/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPACK_HPP
#define TPACK_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TPACK(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                               uint32_t validRows, uint32_t validCols)
{
    using T = typename TileDataDst::DType;
    using U = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ U *srcPtr = (__ubuf__ U *)__cce_get_tile_ptr(src);
    uint32_t repeatTimes = CeilDivision(validCols, REPEAT_BYTE / sizeof(U));
    constexpr auto distType = std::conditional_t<
        (sizeof(U) == 4 && sizeof(T) == 2), decltype(PK_B32),
        std::conditional_t<(sizeof(U) == 4 && sizeof(T) == 1), decltype(PK4_B32),
                           std::conditional_t<(sizeof(U) == 2 && sizeof(T) == 1), decltype(PK_B16), decltype(NORM)>>>();
    __VEC_SCOPE__
    {
        RegTensor<T> vreg;
        MaskReg preg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = validCols;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<typename TileDataSrc::DType>(sreg);
                vlds((RegTensor<U> &)vreg, srcPtr, i * TileDataSrc::RowStride + j * REPEAT_BYTE / sizeof(U), NORM);
                vsts((RegTensor<T> &)vreg, dstPtr, i * TileDataDst::RowStride + j * REPEAT_BYTE / sizeof(U), distType,
                     preg);
            }
        }
    }
}

// Allowed packing directions:
//   b32 in -> b16/b8 out
//   b16 in -> b8 out
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TPackCheck(const TileDataDst &dst, const TileDataSrc &src)
{
    using T = typename TileDataDst::DType;
    using U = typename TileDataSrc::DType;
    static_assert(sizeof(U) > sizeof(T),
                  "Fix: TPack requires source element type wider than destination element type.");
    static_assert(
        (sizeof(U) == 4 && sizeof(T) == 2) || (sizeof(U) == 4 && sizeof(T) == 1) || (sizeof(U) == 2 && sizeof(T) == 1),
        "Fix: TPack only supports b32->b16, b32->b8, or b16->b8.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src.GetValidRow() == validRows && src.GetValidCol() == validCols,
               "Fix: TPack input tile src valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TPACK_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    TPackCheck(dst, src);
    TPACK<TileDataDst, TileDataSrc>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol());
}
} // namespace pto

#endif // TPACK_HPP