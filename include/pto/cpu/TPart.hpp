/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TPART_HPP
#define PTO_CPU_TPART_HPP

#include <algorithm>

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

namespace {
template <typename TileData>
PTO_INTERNAL bool in_valid(TileData &tile, std::size_t r, std::size_t c)
{
    return r < static_cast<std::size_t>(tile.GetValidRow()) && c < static_cast<std::size_t>(tile.GetValidCol());
}
} // namespace

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const bool v0 = in_valid(src0, r, c);
            const bool v1 = in_valid(src1, r, c);
            typename TileDataDst::DType out = 0;
            if (v0 && v1) {
                out = static_cast<typename TileDataDst::DType>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)] +
                                                              src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
            } else if (v0) {
                out = static_cast<typename TileDataDst::DType>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
            } else if (v1) {
                out = static_cast<typename TileDataDst::DType>(src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
            }
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = out;
        }
    });
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTMAX_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const bool v0 = in_valid(src0, r, c);
            const bool v1 = in_valid(src1, r, c);
            typename TileDataDst::DType out = 0;
            if (v0 && v1) {
                const auto a = static_cast<typename TileDataDst::DType>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
                const auto b = static_cast<typename TileDataDst::DType>(src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
                out = std::max(a, b);
            } else if (v0) {
                out = static_cast<typename TileDataDst::DType>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
            } else if (v1) {
                out = static_cast<typename TileDataDst::DType>(src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
            }
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = out;
        }
    });
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTMIN_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const bool v0 = in_valid(src0, r, c);
            const bool v1 = in_valid(src1, r, c);
            typename TileDataDst::DType out = 0;
            if (v0 && v1) {
                const auto a = static_cast<typename TileDataDst::DType>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
                const auto b = static_cast<typename TileDataDst::DType>(src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
                out = std::min(a, b);
            } else if (v0) {
                out = static_cast<typename TileDataDst::DType>(src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)]);
            } else if (v1) {
                out = static_cast<typename TileDataDst::DType>(src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)]);
            }
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = out;
        }
    });
}

} // namespace pto

#endif

