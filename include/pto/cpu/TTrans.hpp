/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_HPP
#define TTRANS_HPP


#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto
{
    template <typename DstTileData, typename SrcTileData>
    void TTrans_Impl(typename DstTileData::TileDType dst,
                            typename SrcTileData::TileDType src,
                            unsigned validRow, unsigned validCol
                        ) {
        cpu::parallel_for_1d(0, validCol, static_cast<std::size_t>(validRow) * validCol, [&](std::size_t c) {
            for (std::size_t r = 0; r < validRow; ++r) {
                const std::size_t src_idx = GetTileElementOffset<SrcTileData>(r, c);
                const std::size_t dst_idx = GetTileElementOffset<DstTileData>(c, r);
                dst[dst_idx] = src[src_idx];
            }
        });
    }

    template <typename DstTileData, typename SrcTileData, typename TmpTileData>
    PTO_INTERNAL void TTRANS_IMPL(DstTileData &dst, SrcTileData &src, TmpTileData &tmp) {
        static_assert (SrcTileData::ValidRow == DstTileData::ValidCol && SrcTileData::ValidCol == DstTileData::ValidRow);
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans_Impl<DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol);
    }
} 


#endif  // TMOV_HPP
