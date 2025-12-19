/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TCOLREDUCE_HPP
#define PTO_CPU_TCOLREDUCE_HPP

#include <algorithm>
#include <type_traits>

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileOut, typename TileIn, typename TileTmp>
PTO_INTERNAL void TCOLSUM_IMPL(TileOut &dst, TileIn &src, TileTmp &tmp, bool isBinary)
{
    (void)tmp;
    const std::size_t rows = static_cast<std::size_t>(src.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(src.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_1d(0, cols, rows * cols, [&](std::size_t c) {
        using AccT = std::conditional_t<std::is_floating_point_v<typename TileOut::DType>, typename TileOut::DType, double>;
        AccT sum = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            const auto v = static_cast<AccT>(src.data()[GetTileElementOffset<TileIn>(r, c)]);
            sum += isBinary ? static_cast<AccT>(v != 0) : v;
        }
        dst.data()[GetTileElementOffset<TileOut>(0, c)] = static_cast<typename TileOut::DType>(sum);
    });
}

template <typename TileOut, typename TileIn>
PTO_INTERNAL void TCOLMAX_IMPL(TileOut &dst, TileIn &src)
{
    const std::size_t rows = static_cast<std::size_t>(src.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(src.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_1d(0, cols, rows * cols, [&](std::size_t c) {
        auto maxVal = static_cast<typename TileOut::DType>(src.data()[GetTileElementOffset<TileIn>(0, c)]);
        for (std::size_t r = 1; r < rows; ++r) {
            const auto v = static_cast<typename TileOut::DType>(src.data()[GetTileElementOffset<TileIn>(r, c)]);
            maxVal = std::max(maxVal, v);
        }
        dst.data()[GetTileElementOffset<TileOut>(0, c)] = maxVal;
    });
}

} // namespace pto

#endif
