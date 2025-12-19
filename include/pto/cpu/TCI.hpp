/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TCI_HPP
#define PTO_CPU_TCI_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileData, typename T, int descending>
PTO_INTERNAL void TCI_IMPL(TileData &dst, T start)
{
    static_assert(std::is_same_v<typename TileData::DType, T>, "TCI: scalar type must match tile element type");

    const std::size_t len = static_cast<std::size_t>(dst.GetValidCol());
    if (len == 0) {
        return;
    }

    cpu::parallel_for_1d(0, len, len, [&](std::size_t i) {
        const T v = (descending != 0) ? static_cast<T>(start - static_cast<T>(i))
                                      : static_cast<T>(start + static_cast<T>(i));
        // Per spec: length is validCol, do not consult validRow; write into row 0.
        dst.data()[GetTileElementOffset<TileData>(0, i)] = v;
    });
}

} // namespace pto

#endif
