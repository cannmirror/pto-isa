/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TSORT32_HPP
#define PTO_CPU_TSORT32_HPP

#include <algorithm>
#include <array>
#include <cstdint>

#include "pto/cpu/tile_offsets.hpp"

namespace pto {

template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INTERNAL void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx)
{
    constexpr std::size_t kBlock = 32;
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(src.GetValidCol());
    if (rows == 0 || cols < kBlock) {
        return;
    }

    const std::size_t blocks = cols / kBlock;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < blocks; ++b) {
            std::array<std::pair<typename SrcTileData::DType, uint32_t>, kBlock> items;
            for (std::size_t k = 0; k < kBlock; ++k) {
                const std::size_t c = b * kBlock + k;
                items[k] = {src.data()[GetTileElementOffset<SrcTileData>(r, c)], static_cast<uint32_t>(c)};
            }
            std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
                if (a.first < b.first) {
                    return true;
                }
                if (b.first < a.first) {
                    return false;
                }
                return a.second < b.second;
            });
            for (std::size_t k = 0; k < kBlock; ++k) {
                const std::size_t c = b * kBlock + k;
                dst.data()[GetTileElementOffset<DstTileData>(r, c)] = static_cast<typename DstTileData::DType>(items[k].first);
                idx.data()[GetTileElementOffset<IdxTileData>(r, c)] = items[k].second;
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INTERNAL void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp)
{
    (void)tmp;
    TSORT32_IMPL(dst, src, idx);
}

} // namespace pto

#endif
