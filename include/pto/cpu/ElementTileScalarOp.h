/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ELEMENT_TILE_SCLAR_OP_HPP
#define ELEMENT_TILE_SCLAR_OP_HPP

#include "pto/cpu/ElementOp.h"

namespace pto {
    template<typename tile_shape, ElementOp op>
    void ZeroTileScalarOp_Impl(tile_shape &dst, typename tile_shape::DType &scalar, unsigned validRow,
                               unsigned validCol)
    {
        using DType = typename tile_shape::DType;
        for(size_t c = 0; c < validCol; c++) {
            for(size_t r = 0; r < validRow; r++) {
                size_t idx = GetTileElementOffset<tile_shape>(r, c);
                ElementOpCal<DType, op>::apply(dst[idx], scalar);
            }
        }
    }

    template<typename tile_shape, ElementOp op>
    void UnaryTileScalarOpImpl(tile_shape &dst, tile_shape &src, typename tile_shape::DType &scalar, unsigned validRow,
                               unsigned validCol, size_t extra = 0)
    {
        using DType = typename tile_shape::DType;
        for (int i = 0; i < validRow; ++i) {
            for (int j = 0; j < validCol; ++j) {
                size_t idx = GetTileElementOffset<tile_shape>(i, j);
                ElementOpCal<DType, op>::apply(dst[idx], src[idx], scalar, extra);
            }
        }
    }
}
#endif