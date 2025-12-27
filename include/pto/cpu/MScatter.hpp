/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MSCATTER_HPP
#define MSCATTER_HPP

#include <pto/common/pto_tile.hpp>
#include <cmath>

namespace pto{

    template <typename TileData, typename TileDataIdx>
    void MScatter_Impl(typename TileData::TileDType src0,
                            typename TileData::TileDType data,
                            typename TileDataIdx::TileDType src1,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            for (int j = 0; j < validCol; ++j) {
                const size_t idx = static_cast<size_t>(src1[GetTileElementOffset<TileDataIdx>(i, j)]);
                data[idx] = src0[GetTileElementOffset<TileData>(i, j)];
            }
        }
    }

  template <typename TileData, typename TileDataIdx>
  PTO_INTERNAL void MSCATTER_IMPL(TileData &src0, typename TileData::TileDType data, TileDataIdx &src1) {
        unsigned row = src0.GetValidRow();
        unsigned col = src0.GetValidCol();
        MScatter_Impl<TileData, TileDataIdx>(src0.data(), data, src1.data(), row, col);
    }
}

#endif
