/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMPS_HPP
#define TCMPS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

constexpr const uint64_t NUM_BITS_IN_BYTE = 8;

    template <typename TileDataDst, typename TileDataSrc, typename T>
    AICORE void GenCmpCall(__ubuf__ typename TileDataDst::DType *dst,
        __ubuf__ typename TileDataSrc::DType *src0, T src1, CmpMode cmpMode,
        uint8_t repeat, uint16_t dstblockstride, uint16_t srcblockstride,
        uint16_t dstrepeatstride, uint16_t srcrepeatstride)
{
        if constexpr (std::is_same<typename TileDataSrc::DType, int32_t>::value) {
            vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
        }
        else {
            switch (static_cast<CmpMode>(cmpMode)) {
                case CmpMode::EQ:
                    vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::NE:
                    vcmpvs_ne(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::LT:
                    vcmpvs_lt(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::GT:
                    vcmpvs_gt(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::GE:
                    vcmpvs_ge(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                case CmpMode::LE:
                    vcmpvs_le(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
                default:
                    vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
                    break;
            }
        }
    }


    template <typename TileDataDst, typename TileDataSrc, typename T, unsigned SS, unsigned DS>
    __tf__ AICORE void TCmps(typename TileDataDst::TileDType __out__ dst,
        typename TileDataSrc::TileDType __in__ src0, T src1, 
        CmpMode mode, unsigned numRepeatPerLine,
        unsigned numRemainPerLine, unsigned validRow,
        unsigned elementsPerRepeat, unsigned blockSizeElem) 
    {
        __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataSrc::DType *srcPtr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src0);
        
        set_mask_count();
        set_vector_mask(0, TileDataDst::Cols);
        size_t dst_offset = 0;
        for (size_t i = 0; i < validRow * numRepeatPerLine; i++) {
            GenCmpCall<TileDataDst, TileDataSrc>(dstPtr + i * BLOCK_BYTE_SIZE,
                                    srcPtr + i * SS,
                                    src1,
                                    mode,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1);
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (size_t index = 0; index < validRow * numRepeatPerLine; index++) {
            for (size_t bit_index = 0; bit_index < DS; bit_index++){
                dstPtr[dst_offset + bit_index] = dstPtr[index * BLOCK_BYTE_SIZE + bit_index];
            }
            dst_offset = dst_offset + DS;
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template <typename TileDataDst, typename TileDataSrc0, typename T>
    AICORE void TCMPS_IMPL(TileDataDst &dst, TileDataSrc0 &src0, T src1, CmpMode cmpMode) {
        static_assert(TileDataSrc0::Loc == TileType::Vec, "TileType of src tiles must be TileType::Vec.");
        static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tiles must be TileType::Vec.");
        static_assert(TileDataSrc0::ValidCol <= TileDataSrc0::Cols, "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileDataSrc0::ValidRow <= TileDataSrc0::Rows, "Number of valid rows must not be greater than number of tile rows.");
        
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc0::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataSrc0::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat + 1;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned SS = REPEAT_BYTE / sizeof(typename TileDataSrc0::DType);
        unsigned validRow = dst.GetValidRow();
        constexpr uint64_t DS = NUM_BITS_IN_BYTE * (sizeof(float)/sizeof(T));
        TCmps<TileDataDst, TileDataSrc0, T, SS, DS>(dst.data(), src0.data(), src1, cmpMode, numRepeatPerLine, numRemainPerLine,
                                                validRow, elementsPerRepeat, blockSizeElem);
    }
}
#endif