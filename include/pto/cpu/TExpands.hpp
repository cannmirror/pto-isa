/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXPANDS_HPP
#define TEXPANDS_HPP

#include <pto/common/pto_tile.hpp>
#include <cmath>

namespace pto{

    template <typename tile_shape>
    void TExpands_Impl(typename tile_shape::TileDType dst,
                            typename tile_shape::DType src,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            for (int j = 0; j < validCol; ++j) {
                size_t idx = GetTileElementOffset<tile_shape>(i,j);
                dst[idx] = src;
            }
        }
    }

    template <typename tile_shape>
    void TEXPANDS_IMPL(tile_shape &dst, typename tile_shape::DType &src) {
        unsigned row = dst.GetValidRow();
        unsigned col = dst.GetValidCol();
        TExpands_Impl<tile_shape>(dst.data(), src, row, col);
    }
}

#endif