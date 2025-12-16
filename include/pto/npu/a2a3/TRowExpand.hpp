/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPAND_HPP
#define TROWEXPAND_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
    PTO_INTERNAL uint16_t DupB8ToB16(uint8_t value) {
        auto u16 = static_cast<uint16_t>(value);
        return u16 + (u16 * 0x100);  // 相当于 extended | (extended << 8)
    }

    PTO_INTERNAL uint16_t DupB8ToB16(int8_t value) {
        auto ub8 = static_cast<uint8_t>(value);
        return DupB8ToB16(ub8);
    }

    template<typename T>
    struct VdupTrait {
        static constexpr bool isB8 = (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
        using DupType = std::conditional_t<isB8, int16_t, T>;

        PTO_INTERNAL DupType DupValue(T value) {
            if constexpr (isB8) {
                return DupB8ToB16(value);
            } else {
                return value;
            }
        }

        PTO_INTERNAL uint64_t DupSize(uint64_t size) {
            if constexpr (isB8) {
                // UB是32B对齐，这是安全的
                return (size + sizeof(DupType) - 1) / sizeof(DupType);
            } else {
                return size;
            }
        }

        PTO_INTERNAL constexpr uint64_t DupDstStride(uint64_t stride) {
            if constexpr (isB8) {
                return stride / sizeof(DupType);
            } else {
                return stride;
            }
        }
    };
    
    template <typename T, typename TileDataDst, typename TileDataSrc, int dststride, int srcstride>
    __tf__ PTO_INTERNAL void TRowExpand(typename TileDataDst::TileDType __out__ dst,
                                        typename TileDataSrc::TileDType __in__ src,
                                        int validRow, int validCol) {
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        using DupType = typename VdupTrait<T>::DupType;
        __ubuf__ DupType *dupDst = (__ubuf__ DupType *)dstPtr;
        VdupTrait<T> trait;
        constexpr int elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        int numRepeatPerLine = validCol / elementsPerRepeat;
        int numRemainPerLine = validCol % elementsPerRepeat;
        constexpr int dupStride = trait.DupDstStride(dststride);

        set_mask_count();
        set_vector_mask(0, validCol);
        for (int i = 0; i < validRow; i++) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            T tempValue = (T)(*(srcPtr + i * srcstride));
            DupType dupValue = trait.DupValue(tempValue);
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            vector_dup(dupDst + i * dupStride, dupValue, 0, 1, 1, 8, 0);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template <typename TileDataDst, typename TileDataSrc>
    PTO_INTERNAL void TROWEXPAND_IMPL(TileDataDst &dst, TileDataSrc &src) {
        using T = typename TileDataSrc::DType;
        static_assert((sizeof(typename TileDataSrc::DType) == 1) || (sizeof(typename TileDataSrc::DType) == 2) ||
                      (sizeof(typename TileDataSrc::DType) == 4), "Data type must be b8/b16/b32");
        static_assert(TileDataSrc::Loc == pto::TileType::Vec, "Src TileType must be Vec!");
        static_assert((TileDataSrc::isRowMajor && (TileDataSrc::SFractal == SLayout::NoneBox)) &&
                      (TileDataDst::isRowMajor && (TileDataDst::SFractal == SLayout::NoneBox)),
                      "Src and dst layout must be ND!");
        static_assert(std::is_same_v<typename TileDataDst::DType, T>,
                      "The input data type must be consistent with the output data type");
        constexpr int dststride = TileDataDst::RowStride;
        constexpr int srcstride = TileDataSrc::RowStride;
        int validRow = dst.GetValidRow();
        int validCol = dst.GetValidCol();
        if (validRow == 0 || validCol == 0 || src.GetValidRow() == 0 || src.GetValidCol() == 0) {
            return;
        }
        TRowExpand<T, TileDataDst, TileDataSrc, dststride, srcstride>(dst.data(), src.data(), validRow, validCol);
    }
}
#endif