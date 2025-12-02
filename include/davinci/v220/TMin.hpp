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

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template <typename TileData, unsigned stride, int dataTypeSize>
    __tf__ __aicore__ void TMin(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::TileDType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *src0Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileData::DType *src1Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src1);
        
        unsigned TShape0 = TileData::Rows;
        unsigned TShape1 = TileData::Cols;

        set_mask_count();
        set_vector_mask(0, validCol);
        for (int i = 0; i < TShape0; ++i){
            vmin((dstPtr + i * TShape1), (src0Ptr + i * TShape1), (src1Ptr + i * TShape1),
            1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template <typename TileData>
    __aicore__ void TMIN_IMPL(TileData &dst, TileData &src0, TileData &src1) {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TMIN: Invalid data type");
        
        constexpr int size = sizeof(typename TileData::DType);              
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();

        TMin<TileData, stride, size>(dst.data(), src0.data(), src1.data(), validRow, validCol);
    }
}
#endif