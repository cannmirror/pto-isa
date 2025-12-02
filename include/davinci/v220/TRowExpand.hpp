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

#include "common/constants.hpp"
#include "utils.hpp"

namespace pto {
    __aicore__ PTO_INLINE uint16_t DupB8ToB16(uint8_t value) {
        auto u16 = static_cast<uint16_t>(value);
        return u16 + (u16 * 0x100);  // 相当于 extended | (extended << 8)
    }

    __aicore__ PTO_INLINE uint16_t DupB8ToB16(int8_t value) {
        auto ub8 = static_cast<uint8_t>(value);
        return DupB8ToB16(ub8);
    }

    template<typename T>
    struct VdupTrait {
        static constexpr bool isB8 = (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
        using DupType = std::conditional_t<isB8, int16_t, T>;

        __aicore__ PTO_INLINE DupType DupValue(T value) {
            if constexpr (isB8) {
                return DupB8ToB16(value);
            } else {
                return value;
            }
        }

        __aicore__ PTO_INLINE uint64_t DupSize(uint64_t size) {
            if constexpr (isB8) {
                // UB是32B对齐，这是安全的
                return (size + sizeof(DupType) - 1) / sizeof(DupType);
            } else {
                return size;
            }
        }

        __aicore__ PTO_INLINE constexpr uint64_t DupDstStride(uint64_t stride) {
            if constexpr (isB8) {
                return stride / sizeof(DupType);
            } else {
                return stride;
            }
        }
    };
    
    template <typename TileDataDst, typename TileDataSrc, unsigned dststride, unsigned srcstride>
    __tf__ __aicore__ PTO_INLINE void TRowExpand(typename TileDataDst::TileDType __out__ dst,
                                                 typename TileDataSrc::TileDType __in__ src,
                                                 unsigned validRow, unsigned validCol) {
        __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataSrc::DType *srcPtr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src);

        using DupType = typename VdupTrait<typename TileDataDst::DType>::DupType;
        __ubuf__ DupType *dupDst = (__ubuf__ DupType *)dstPtr;
        VdupTrait<typename TileDataDst::DType> trait;
        constexpr unsigned DTypeSize = sizeof(typename TileDataDst::DType);
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / DTypeSize;
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / DTypeSize;
        unsigned dupValidCol = 
            (validCol * sizeof(typename TileDataDst::DType) + DTypeSize - 1) / DTypeSize;
        unsigned numRepeatPerLine = dupValidCol / elementsPerRepeat;
        unsigned numRemainPerLine = dupValidCol % elementsPerRepeat;
        unsigned numBlockPerLine =
            (srcstride * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        unsigned dupSrcStride = numBlockPerLine * blockSizeElem;
        constexpr unsigned dupStride = trait.DupDstStride(dststride);

        if (numRepeatPerLine > 0) {
            set_mask_count();
            set_vector_mask(0, numRepeatPerLine * elementsPerRepeat);
            for (int i = 0; i < validRow; i++) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                typename TileDataSrc::DType tempValue =
                    (typename TileDataSrc::DType)(*(srcPtr + i * dupSrcStride));
                DupType dupValue = trait.DupValue(tempValue);
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                vector_dup(dupDst + i * dupStride, dupValue, 0, 1, 1, 8, 0);
            }
            set_mask_norm();
            set_vector_mask(-1, -1);
        }

        dupDst += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine) {
            SetContinuousMask(numRemainPerLine);
            for (int i = 0; i < validRow; i++) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                typename TileDataSrc::DType tempValue =
                    (typename TileDataSrc::DType)(*(srcPtr + i * srcstride));
                DupType dupValue = trait.DupValue(tempValue);
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                vector_dup(dupDst + i * dupStride, dupValue, 1, 1, 1, 8, 0);
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TROWEXPAND_IMPL(TileDataDst &dst, TileDataSrc &src) {
        static_assert((sizeof(typename TileDataSrc::DType) == 1) || (sizeof(typename TileDataSrc::DType) == 2) ||
                      (sizeof(typename TileDataSrc::DType) == 4), "Data type must be b8/b16/b32");
        static_assert(TileDataSrc::Loc == pto::Location::Vec, "Src location must be Vec!");
        static_assert((TileDataSrc::isRowMajor && (TileDataSrc::SFractal == SLayout::NoneBox)) &&
                      (TileDataDst::isRowMajor && (TileDataDst::SFractal == SLayout::NoneBox)),
                      "Src and dst layout must be ND!");
        constexpr unsigned dststride = TileDataDst::RowStride;
        constexpr unsigned srcstride = TileDataSrc::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TRowExpand<TileDataDst, TileDataSrc, dststride, srcstride>(dst.data(), src.data(), validRow, validCol);
    }
}
#endif