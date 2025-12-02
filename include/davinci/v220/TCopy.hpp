/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOPY_HPP
#define TCOPY_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto{
    template <typename TileDataDst, typename TileDataSrc, unsigned blockSizeElem, unsigned srcStride,
        unsigned dstStride>
    __tf__ __aicore__ PTO_INLINE void TCopy(typename TileDataDst::TileDType __out__ dst,
        typename TileDataSrc::TileDType __in__ src, uint64_t validRow, uint64_t validCol) {
        if (validRow ==0 || validCol == 0) {
            return;
        }
        using T = typename TileDataSrc::DType;
        using U = typename TileDataDst::DType;
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);

        static_assert(sizeof(T) == sizeof(U), "TCOPY: src and dst data type is different!");
        
        if constexpr ((TileDataDst::isRowMajor && TileDataDst::Cols == TileDataSrc::Cols)||
        (!TileDataDst::isRowMajor && TileDataDst::Rows == TileDataSrc::Rows)) {
            set_mask_count();
            set_vector_mask(0, TileDataDst::Cols*TileDataDst::Rows);
            uint64_t blockLen = (TileDataDst::Cols*TileDataDst::Rows*sizeof(T) + BLOCK_BYTE_SIZE-1) / BLOCK_BYTE_SIZE;
            copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, 1, blockLen, 1, 1);
            set_mask_norm();
            set_vector_mask(-1, -1);
        } else {
            set_mask_count();
            set_vector_mask(0, validCol);
            uint64_t blockLen = (validCol*sizeof(T) + BLOCK_BYTE_SIZE-1) / BLOCK_BYTE_SIZE;
            for (uint64_t row = 0; row < validRow; ++row) {
                copy_ubuf_to_ubuf(dstPtr + row * dstStride, srcPtr + row * srcStride, 0, 1, blockLen, 1, 1);
            }
            set_mask_norm();
            set_vector_mask(-1, -1);
        }   
    }  // end of tf
}
#endif