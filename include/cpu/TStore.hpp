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

#include "common/constants.hpp"
#include <cassert>

namespace pto {
    template <typename GlobalData, typename TileData>
    __tf__ __aicore__ void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        assert((gShape0*gShape1*gShape2*gShape3 == validRow && gShape4==validCol && TileData::isRowMajor) ||
            (gShape0*gShape1*gShape2*gShape4 == validCol && gShape3==validRow && !TileData::isRowMajor));
        if(TileData::SFractal == SLayout::NoneBox) {
            int64_t srcStride1 = gShape2;
            int64_t srcStride0 = gShape1 * srcStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t srcAddr0 = i * srcStride0;
                int64_t dstAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t srcAddr1 = j * srcStride1;
                    int64_t dstAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        size_t offsetDstBase = dstAddr0 + dstAddr1 + k * gStride2;

                        if constexpr (TileData::isRowMajor) { // ND
                            size_t offsetSrcBase =  (srcAddr0 + srcAddr1 + k)*gShape3*TileData::Cols;
                            for (uint32_t r = 0; r < gShape3; r++) {
                                for (uint32_t c = 0; c < gShape4; c++) {
                                    size_t offsetSrc = offsetSrcBase + r*TileData::Cols + c;
                                    size_t offsetDst = offsetDstBase + r*gStride3 + c*gStride4;
                                    dst[offsetDst] = src[offsetSrc];
                                }
                            }
                        } else { // DN
                            size_t offsetSrcBase =  (srcAddr0 + srcAddr1 + k)*gShape4*TileData::Rows;
                            for (uint32_t r = 0; r < gShape3; r++) {
                                for (uint32_t c = 0; c < gShape4; c++) {
                                    size_t offsetSrc = offsetSrcBase + c*TileData::Rows + r;
                                    size_t offsetDst = offsetDstBase + r*gStride3 + c*gStride4;
                                    dst[offsetDst] = src[offsetSrc];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            assert(gShape0==1 && gShape1==1 && gShape2==1 && "Nz,Zn -> ND,DN convertion does support only 2D GMs");
            if constexpr (!TileData::isRowMajor) { // Nz layout
                for(size_t c=0; c<gShape4; c++) {
                    size_t subTileC = c / TileData::InnerCols;
                    size_t innerC = c % TileData::InnerCols;
                    for(size_t r=0; r < gShape3; r++) {
                        size_t subTileR = r / TileData::InnerRows;
                        size_t innerR = r % TileData::InnerRows;

                        size_t tile_idx = subTileC*TileData::Rows*TileData::InnerCols +
                            subTileR*TileData::InnerNumel + innerR*TileData::InnerCols + innerC;
                        size_t gd_idx = r*gStride3 + c*gStride4;

                        dst[gd_idx] = src[tile_idx];
                    }
                }
            } else { // Zn layout
                for(size_t c=0; c<gShape4; c++) {
                    size_t subTileC = c / TileData::InnerCols;
                    size_t innerC = c % TileData::InnerCols;
                    for(size_t r=0; r < gShape3; r++) {
                        size_t subTileR = r / TileData::InnerRows;
                        size_t innerR = r % TileData::InnerRows;

                        size_t tile_idx = subTileR*TileData::Cols*TileData::InnerRows +
                            subTileC*TileData::InnerNumel + innerC*TileData::InnerRows + innerR;

                        size_t gd_idx = r*gStride3 + c*gStride4;

                        dst[gd_idx] = src[tile_idx];
                    }
                }
            }
        }
    }

    template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
    __aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
    {
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype!");
        static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN , "Only ND and DN GLobal Tensors are currently supported");
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