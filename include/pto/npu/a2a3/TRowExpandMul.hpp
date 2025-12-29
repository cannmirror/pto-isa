/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPANDMUL_HPP
#define TROWEXPANDMUL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/TRowExpandBinOp.hpp>

namespace pto {
template <typename T>
struct RowExpandMulOp {
    PTO_INTERNAL static void RowExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats) {
        vmul(dst, src0, src1, repeats, 1, 1, 0, 8, 8, 0);
    }
    PTO_INTERNAL static void RowExpandBinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeats,
        uint8_t dstRepeatStride, uint8_t src0RepeatStride) {
        vmul(dst, src0, src1, repeats, 1, 1, 0, dstRepeatStride, src0RepeatStride, 1);
    }
};

template <typename TileDataDst, typename TileDataSrc1, unsigned rowStride>
__tf__ PTO_INTERNAL void TRowExpandMul(typename TileDataDst::TileDType __out__ dst,
    typename TileDataDst::TileDType __in__ src0, typename TileDataSrc1::TileDType __in__ src1, unsigned validRow,
    unsigned validCol) {
    using T = typename TileDataDst::DType;
    using U = typename std::conditional<sizeof(typename TileDataDst::DType) == 4, uint32_t, uint16_t>::type;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ U *src1Ptr = (__ubuf__ U *)__cce_get_tile_ptr(src1);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB tmpbuf address
    __ubuf__ U *tmpPtr_ = (__ubuf__ U *)(TMP_UB_OFFSET); // 8KB tmpbuf address
    TRowExpandBinaryInstr<RowExpandMulOp<T>, T, U, TileDataDst::Rows, rowStride>(
        dstPtr, src0Ptr, src1Ptr, tmpPtr, tmpPtr_, validRow, validCol);
}

template <typename TileDataDst, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDMUL_IMPL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1) {
    static_assert(std::is_same_v<typename TileDataDst::DType, typename TileDataSrc1::DType>,
        "Fix: TROWEXPANDMUL src and dst data type is different!");
    static_assert(
        std::is_same_v<typename TileDataDst::DType, half> || std::is_same_v<typename TileDataDst::DType, float>,
        "Fix: TROWEXPANDMUL has invalid data type.");
    static_assert(TileDataDst::isRowMajor && !TileDataSrc1::isRowMajor && TileDataSrc1::Cols == 1,
        "Fix: TROWEXPANDMUL has invalid tile shape.");
    constexpr unsigned rowStride = TileDataDst::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(src1.GetValidRow() == 1 && src1.GetValidCol() == validRow, "TROWEXPANDMUL: invalid src1 shape.");
    TRowExpandMul<TileDataDst, TileDataSrc1, rowStride>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}
} // namespace pto
#endif