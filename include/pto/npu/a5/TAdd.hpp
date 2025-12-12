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
#include "common.hpp"
#include "utils.hpp"
#include "TBinOp.hpp"

using namespace pto;
using namespace std;

namespace pto {

template <typename T> struct AddOp {
    PTO_INTERNAL static void BinInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1, MaskReg &preg)
    {
        vadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL OP_NAME(TADD) OP_TYPE(element_wise)
void TAdd(typename TileData::TileDType __out__ dst, 
                            typename TileData::TileDType __in__ src0, 
                            typename TileData::TileDType __in__ src1,
                            unsigned kValidRows,
                            unsigned kValidCols,
                            BinOpsImpl version = BinOpsImpl::BinOpsIMPL_DEFAULT) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    BinaryInstr<AddOp<T>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, src0Ptr, src1Ptr, kValidRows, kValidCols, version);
}

template <typename TileData>
PTO_INTERNAL void TADD_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                  std::is_same<typename TileData::DType, uint32_t>::value ||
                  std::is_same<typename TileData::DType, float>::value ||
                  std::is_same<typename TileData::DType, int16_t>::value ||
                  std::is_same<typename TileData::DType, uint16_t>::value ||
                  std::is_same<typename TileData::DType, half>::value ||
                  std::is_same<typename TileData::DType, bfloat16_t>::value ||
                  std::is_same<typename TileData::DType, uint8_t>::value ||
                  std::is_same<typename TileData::DType, int8_t>::value,
                  "TADD: Invalid data type.");
    static_assert(TileData::isRowMajor, "TADD: not supported Layout type");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), src1.data(), validRow, validCol);
}
}  // namespace pto
#endif

