/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMIN_HPP
#define TMIN_HPP

#include <pto/common/constants.hpp>
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {

    template <typename T> struct MinOp {
        __aicore__ PTO_INLINE static void BinInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1, MaskReg &preg)
        {
            vmin(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    };

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ __aicore__ PTO_INLINE
    void TMin(typename TileData::TileDType __out__ dst, 
                                typename TileData::TileDType __in__ src0, 
                                typename TileData::TileDType __in__ src1,
                                unsigned kValidRows,
                                unsigned kValidCols,
                                BinOpsImpl version = BinOpsImpl::BinOpsIMPL_DEFAULT) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
        BinaryInstr<MinOp<T>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
                    dstPtr, src0Ptr, src1Ptr, kValidRows, kValidCols, version);
    }

    template <typename TileData>
    __aicore__ void TMIN_IMPL(TileData &dst, TileData &src0, TileData &src1) {
        static_assert(TileData::isRowMajor, "TMIN: not supported Layout type");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType); 
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType); 
        constexpr unsigned rowStride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();

        TMin<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), src1.data(), validRow, validCol);
    }
}
#endif
