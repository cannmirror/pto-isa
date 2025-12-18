/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSCATTER_HPP
#define TSCATTER_HPP

#include <pto/common/constants.hpp>
#include "TBinOp.hpp"
namespace pto 
{
    template <typename TileData, typename TileIndex>
    __tf__ PTO_INTERNAL void TScatter(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileIndex::TileDType __in__ src1,
                                unsigned validRow,
                                unsigned validCol
                                ) 
    {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileIndex::DType *indPtr = (__ubuf__ typename TileIndex::DType *)__cce_get_tile_ptr(src1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int i = 0; i < validRow; i++) {
            for (int j = 0; j < validCol; j++) {
                typename TileIndex::DType index = (typename TileIndex::DType)(*(indPtr + i * TileIndex::Cols + j));
                int dstOffset = index * TileData::Cols + j;
                dstPtr[dstOffset] = src0Ptr[i * TileData::Cols + j];
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }

    template <typename TileData, typename TileIndex>
    PTO_INTERNAL void TSCATTER_IMPL(TileData &dst, TileData &src0, TileIndex &indexes)
    {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TSCATTER: Invalid data type");
        static_assert(std::is_same<typename TileIndex::DType, uint16_t>::value || std::is_same<typename TileIndex::DType, uint32_t>::value, "TSCATTER: Invalid data type of indexes");    
        static_assert(TileData::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "Number of valid rows must not be greater than number of tile rows.");
        
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidCol() == indexes.GetValidCol(), "Number of columns of src and indexes must be the same.");
        PTO_ASSERT(src0.GetValidRow() == indexes.GetValidRow(), "Number of rows of src and indexes must be the same.");
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TScatter<TileData, TileIndex>(dst.data(), src0.data(), indexes.data(), validRow, validCol);
    }
}

#endif