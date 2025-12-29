/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TADD_HPP
#define TADD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a5/TBinOp.hpp"
#include "pto/npu/a5/common.hpp"
#include "pto/npu/a5/utils.hpp"

using namespace pto;
using namespace std;

namespace pto {

template <typename T>
struct AddOp {
    PTO_INTERNAL static void BinInstr(
        RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1, MaskReg &preg) {
        vadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL OP_NAME(TADD) OP_TYPE(element_wise) void TAdd(typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0, typename TileData::TileDType __in__ src1, unsigned kValidRows,
    unsigned kValidCols, BinOpsImpl version = BinOpsImpl::BinOpsIMPL_DEFAULT) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<AddOp<T>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
        dstPtr, src0Ptr, src1Ptr, kValidRows, kValidCols, version);
}

template <typename T, typename TileData>
PTO_INTERNAL
void TAddCheck() {
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                  std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
                  std::is_same_v<T, bfloat16_t> ||std::is_same_v<T, uint8_t> ||std::is_same_v<T, int8_t>,
                  "Fix: TAdd Invalid data type.");
    static_assert(TileData::isRowMajor, "Fix: TAdd not supported Layout type");
}

template <typename TileData>
PTO_INTERNAL void TADD_IMPL(TileData &dst, TileData &src0, TileData &src1) {
    using T = typename TileData::DType;
    TAddCheck<T, TileData>();
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(
        dst.data(), src0.data(), src1.data(), validRow, validCol);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
    static_assert(std::is_same_v<TileDataDst, TileDataSrc0> && std::is_same_v<TileDataDst, TileDataSrc1>,
        "Fix: TADD Input tileshape must be consistent with the out tileshape.");

    using T = typename TileDataDst::DType;
    TAddCheck<T, TileDataDst>();
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned rowStride = TileDataDst::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    TAdd<TileDataDst, elementsPerRepeat, blockSizeElem, rowStride>(
        dst.data(), src0.data(), src1.data(), validRow, validCol);
}
} // namespace pto
#endif
