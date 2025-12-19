/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSQRT_HPP
#define TSQRT_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <cmath>

namespace pto{

    template <typename tile_shape>
    void TSqrt_Impl(typename tile_shape::TileDType dst,
                            typename tile_shape::TileDType src,
                            unsigned validRow, unsigned validCol
                        ) {
        if constexpr (tile_shape::SFractal == SLayout::NoneBox && tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                const std::size_t base = r * tile_shape::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = base + c;
                    dst[idx] = std::sqrt(static_cast<double>(src[idx]));
                }
            });
        } else {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    dst[idx] = std::sqrt(static_cast<double>(src[idx]));
                }
            });
        }
    }

    template <typename tile_shape>
    PTO_INTERNAL void TSQRT_IMPL(tile_shape &dst, tile_shape &src) {
        static_assert(std::is_same<typename tile_shape::DType, half>::value ||
                      std::is_same<typename tile_shape::DType, float>::value,
                      "TSQRT: Invalid data type");
        TSqrt_Impl<tile_shape>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
    }
}
#endif  // TSQRT_HPP
