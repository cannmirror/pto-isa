/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TSEL_HPP
#define PTO_CPU_TSEL_HPP

#include <cstdint>

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileData, typename MaskTile>
PTO_INTERNAL void TSEL_IMPL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t byteIdx = c / 8;
            const uint8_t bit = static_cast<uint8_t>(1u << (c % 8));
            const uint8_t m = static_cast<uint8_t>(selMask.data()[GetTileElementOffset<MaskTile>(r, byteIdx)]);
            const bool pick0 = (m & bit) != 0;
            const auto v = pick0 ? src0.data()[GetTileElementOffset<TileData>(r, c)]
                                 : src1.data()[GetTileElementOffset<TileData>(r, c)];
            dst.data()[GetTileElementOffset<TileData>(r, c)] = v;
        }
    });
}

} // namespace pto

#endif

