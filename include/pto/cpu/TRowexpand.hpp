/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TROWEXPAND_HPP
#define TROWEXPAND_HPP


#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto{
    template <typename tile_shape_out, typename tile_shape_in>
    void TRowexpand_Impl(typename tile_shape_out::TileDType dst,
                            typename tile_shape_in::TileDType src,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            size_t src_idx = GetTileElementOffset<tile_shape_in>(i, 0);
            for (int j = 0; j < validCol; ++j) {
                size_t dst_idx = GetTileElementOffset<tile_shape_out>(i, j);
                dst[dst_idx] = src[src_idx];
            }
        }
    }

  template <typename TileDataOut, typename TileDataIn>
  PTO_INTERNAL void TROWEXPAND_IMPL(TileDataOut &dst, TileDataIn &src) {
        unsigned row = dst.GetValidRow();
        unsigned col = dst.GetValidCol();
        TRowexpand_Impl<TileDataOut, TileDataIn>(dst.data(), src.data(), row, col);
    }
}
#endif