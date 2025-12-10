/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMUL_HPP
#define TMUL_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto{
    template<typename tile_shape, int stride>
    void TMul_Impl(typename tile_shape::TileDType dst,
                            typename tile_shape::TileDType src0,
                            typename tile_shape::TileDType src1,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            for (int j = 0; j < validCol; ++j) {
                size_t offset = GetTileElementOffset<tile_shape>(i,j);
                dst[offset] = src0[offset] * src1[offset];
            }
        }
    }

    template <typename tile_shape>
    PTO_INTERNAL void TMUL_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
        unsigned row = dst.GetValidRow();
        unsigned col = dst.GetValidCol();
        constexpr unsigned stride = tile_shape::RowStride;
        TMul_Impl<tile_shape, stride>(dst.data(), src0.data(), src1.data(), row, col);
    }
}
#endif