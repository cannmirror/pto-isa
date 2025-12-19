/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TCMPS_HPP
#define PTO_CPU_TCMPS_HPP

#include <pto/common/type.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

namespace {
template <typename T>
PTO_INTERNAL bool apply_cmp(T a, T b, CmpMode mode)
{
    switch (mode) {
        case CmpMode::EQ:
            return a == b;
        case CmpMode::NE:
            return a != b;
        case CmpMode::LT:
            return a < b;
        case CmpMode::GT:
            return a > b;
        case CmpMode::GE:
            return a >= b;
        case CmpMode::LE:
            return a <= b;
        default:
            return a == b;
    }
}
} // namespace

template <typename TileDataDst, typename TileDataSrc0, typename T>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc0 &src0, T src1, CmpMode cmpMode)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const auto s0 = static_cast<T>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
            const bool pred = apply_cmp<T>(s0, static_cast<T>(src1), cmpMode);
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = static_cast<typename TileDataDst::DType>(pred);
        }
    });
}

} // namespace pto

#endif

