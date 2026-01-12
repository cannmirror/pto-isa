/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRI_HPP
#define TTRI_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// Helper to construct element `one`/`zero` with correct literal type for
// integer vs floating/half element types. Avoids duplicated code and
// accidental shadowing from in-function redeclarations.
template <typename T>
PTO_INTERNAL constexpr T make_one() {
    if constexpr (std::is_floating_point<T>::value || std::is_same<T, half>::value) {
        return static_cast<T>(1.0);
    } else {
        return static_cast<T>(1);
    }
}

template <typename T>
PTO_INTERNAL constexpr T make_zero() {
    return static_cast<T>(0);
}

template <typename T, int isUpperOrLower, int diagonal, unsigned rowStride>
PTO_INTERNAL void TTril(__ubuf__ T *dstPtr, unsigned validRow, unsigned validCol) {
    T one = make_one<T>();
    T zero = make_zero<T>();
    // lower-triangular
    for (unsigned i = 0; i < validRow; ++i) {
        __ubuf__ T *drow = dstPtr + i * rowStride;

        // clear full row first
        set_vector_mask(0, validCol);
        vector_dup(drow, zero, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        int lastCol = static_cast<int>(i) + static_cast<int>(diagonal);
        if (lastCol >= 0) {
            int want = lastCol + 1;
            unsigned fillCol =
                (want <= 0) ? 0 : (want >= static_cast<int>(validCol) ? validCol : static_cast<unsigned>(want));
            if (fillCol > 0) {
                set_vector_mask(0, fillCol);
                vector_dup(drow, one, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
            }
        }
    }
}

template <typename T, int isUpperOrLower, int diagonal, unsigned rowStride>
PTO_INTERNAL void TTriu(__ubuf__ T *dstPtr, unsigned validRow, unsigned validCol) {
    T one = make_one<T>();
    T zero = make_zero<T>();
    // upper-triangular
    for (unsigned i = 0; i < validRow; ++i) {
        __ubuf__ T *drow = dstPtr + i * rowStride;

        // clear full row first
        set_vector_mask(0, validCol);
        vector_dup(drow, zero, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        int firstCol = static_cast<int>(i) + static_cast<int>(diagonal);
        if (firstCol < 0)
            firstCol = 0;
        if (firstCol < static_cast<int>(validCol)) {
            unsigned start = static_cast<unsigned>(firstCol);
            set_vector_mask(0, validCol - start);
            vector_dup(drow + start, one, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename TileData, int isUpperOrLower, int diagonal, unsigned rowStride>
__tf__ PTO_INTERNAL void TTri(typename TileData::TileDType __out__ dst, unsigned validRow, unsigned validCol) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);

    set_mask_count();
    if constexpr (isUpperOrLower == 0) {
        TTril<T, isUpperOrLower, diagonal, rowStride>(dstPtr, validRow, validCol);
    } else {
        TTriu<T, isUpperOrLower, diagonal, rowStride>(dstPtr, validRow, validCol);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    return;
}

template <typename TileData, int isUpperOrLower, int diagonal>
PTO_INTERNAL void TTRI_IMPL(TileData &dst) {
    static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                  std::is_same<typename TileData::DType, int16_t>::value ||
                  std::is_same<typename TileData::DType, uint32_t>::value ||
                  std::is_same<typename TileData::DType, uint16_t>::value ||
                  std::is_same<typename TileData::DType, half>::value ||
                  std::is_same<typename TileData::DType, float>::value || "Fix: TTri has invalid data type.");
    static_assert(TileData::isRowMajor, "Fix: TTri has not supported Layout type.");
    static_assert(isUpperOrLower == 0 || isUpperOrLower == 1, "Fix: isUpperOrLower must be 0 or 1.");
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TTri<TileData, isUpperOrLower, diagonal, rowStride>(dst.data(), validRow, validCol);
}

} // namespace pto
#endif