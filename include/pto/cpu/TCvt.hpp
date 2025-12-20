/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TCVT_HPP
#define PTO_CPU_TCVT_HPP

#include <cmath>
#include <type_traits>

#include <pto/common/constants.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

namespace {
PTO_INTERNAL double apply_round(double x, RoundMode mode)
{
    constexpr double NUM_HALF = 0.5;
    switch (mode) {
        case RoundMode::CAST_NONE:
            return x;
        case RoundMode::CAST_RINT:
            return std::nearbyint(x);
        case RoundMode::CAST_ROUND:
            return std::round(x);
        case RoundMode::CAST_FLOOR:
            return std::floor(x);
        case RoundMode::CAST_CEIL:
            return std::ceil(x);
        case RoundMode::CAST_TRUNC:
            return std::trunc(x);
        case RoundMode::CAST_ODD: {
            const double f = std::floor(x);
            const double c = std::ceil(x);
            if (x - f == NUM_HALF) {
                const long long fi = static_cast<long long>(f);
                const long long ci = static_cast<long long>(c);
                return (fi & 1LL) ? f : c;
            }
            if (c - x == NUM_HALF) {
                const long long fi = static_cast<long long>(f);
                const long long ci = static_cast<long long>(c);
                return (ci & 1LL) ? c : f;
            }
            return std::nearbyint(x);
        }
        default:
            return x;
    }
}

template <typename DstT, typename SrcT>
PTO_INTERNAL DstT convert(SrcT v, RoundMode mode)
{
    if constexpr (std::is_integral_v<DstT> && std::is_floating_point_v<SrcT>) {
        return static_cast<DstT>(apply_round(static_cast<double>(v), mode));
    } else {
        (void)mode;
        return static_cast<DstT>(v);
    }
}
} // namespace

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const auto v = src.data()[GetTileElementOffset<TileDataS>(r, c)];
            dst.data()[GetTileElementOffset<TileDataD>(r, c)] = convert<typename TileDataD::DType>(v, mode);
        }
    });
}

} // namespace pto

#endif

