/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP

#include "constants.hpp"

namespace pto {
    template <typename GlobalData, typename TileData>
    __aicore__ PTO_INLINE void TStoreInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src,
        uint16_t nBurst, uint32_t lenBurst, uint32_t gmGap, uint32_t ubGap)
    {
    }

    template <typename GlobalData, typename TileData>
    __tf__ __aicore__ void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            int64_t srcStride2 = gShape3 * TileData::Cols;
            int64_t srcStride1 = gShape2 * srcStride2;
            int64_t srcStride0 = gShape1 * srcStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t srcAddr0 = i * srcStride0;
                int64_t dstAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t srcAddr1 = j * srcStride1;
                    int64_t dstAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        int64_t srcAddr2 = k * srcStride2;
                        int64_t dstAddr2 = k * gStride2;
                        for (uint32_t l = 0; l < validRow; l++) {
                            for (uint32_t m = 0; m < validCol; m++) {
                                size_t offsetSrc = srcAddr0 + srcAddr1 + srcAddr2 + l*TileData::Cols + m;
                                size_t offsetDst = dstAddr0 + dstAddr1 + dstAddr2 + l*gStride3 + m*gStride4;
                                dst[offsetDst] = src[offsetSrc];
                            }
                        }
                    }
                }
            }
        } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            int64_t srcStride2 = gShape4 * TileData::Rows;
            int64_t srcStride1 = gShape2 * srcStride2;
            int64_t srcStride0 = gShape1 * srcStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t srcAddr0 = i * srcStride0;
                int64_t dstAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t srcAddr1 = j * srcStride1;
                    int64_t dstAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        int64_t srcAddr2 = k * srcStride2;
                        int64_t dstAddr2 = k * gStride2;
                        for (uint32_t l = 0; l < validRow; l++) {
                            for (uint32_t m = 0; m < validCol; m++) {
                                size_t offsetSrc = srcAddr0 + srcAddr1 + srcAddr2 + m*TileData::Rows + l;
                                size_t offsetDst = dstAddr0 + dstAddr1 + dstAddr2 + l*gStride3 + m*gStride4;
                                dst[offsetDst] = src[offsetSrc];
                            }
                        }
                    }
                }
            }
        } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) { // Nz layout
            for(size_t c=0; c<TileData::ValidCol; c++) {
                size_t subTileC = c / TileData::InnerCols;
                size_t innerC = c % TileData::InnerCols;
                for(size_t r=0; r<TileData::ValidRow; r++) {
                    size_t subTileR = r / TileData::InnerRows;
                    size_t innerR = r % TileData::InnerRows;

                    size_t tile_idx = subTileC*TileData::Rows*TileData::InnerCols +
                        subTileR*TileData::InnerNumel + innerR*TileData::InnerCols + innerC;

                    size_t gd_idx = r*gStride3+c;

                    dst[gd_idx] = src[tile_idx];
                }
            }
        } else if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::ColMajor)) { // Zn layout
            for(size_t c=0; c<TileData::ValidCol; c++) {
                size_t subTileC = c / TileData::InnerCols;
                size_t innerC = c % TileData::InnerCols;
                for(size_t r=0; r<TileData::ValidRow; r++) {
                    size_t subTileR = r / TileData::InnerRows;
                    size_t innerR = r % TileData::InnerRows;

                    size_t tile_idx = subTileR*TileData::Cols*TileData::InnerRows +
                        subTileC*TileData::InnerNumel + innerC*TileData::InnerRows + innerR;

                    size_t gd_idx = r*gStride3+c;

                    dst[gd_idx] = src[tile_idx];
                }
            }
        }
    }

    template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
    __aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
    {
        static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                      (sizeof(typename TileData::DType) == 4), "Data type must be b8/16/32");

        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype!");
        static_assert(GlobalData::layout == pto::Layout::ND, "Only ND GLobal Tensors are currently supported");
        TStore<GlobalData, TileData>(dst.data(),
            src.data(),
            dst.GetShape(0),
            dst.GetShape(1),
            dst.GetShape(2),
            dst.GetShape(3),
            dst.GetShape(4),
            dst.GetStride(0),
            dst.GetStride(1),
            dst.GetStride(2),
            dst.GetStride(3),
            dst.GetStride(4),
            src.GetValidRow(),
            src.GetValidCol());
    }
}
#endif