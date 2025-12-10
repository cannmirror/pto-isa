/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP

#include <unistd.h>
#include <cassert>

namespace pto {
    template <typename TileData>
    AICORE constexpr TileData::DType getPadValue()
    {    
        switch (TileData::PadVal)
        {
            case PadValue::Null:
            case PadValue::Zero: return typename TileData::DType(0);
            case PadValue::Min:
                if constexpr(std::numeric_limits<typename TileData::DType>::has_infinity) {
                    return -std::numeric_limits<typename TileData::DType>::infinity();
                } else {
                    return std::numeric_limits<typename TileData::DType>::min();
                }
            case PadValue::Max:
                if constexpr(std::numeric_limits<typename TileData::DType>::has_infinity) {
                    return std::numeric_limits<typename TileData::DType>::infinity();
                } else {
                    return std::numeric_limits<typename TileData::DType>::max();
                }
        }
        return 0;
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<TileData::isRowMajor, int> = 0>
    __tf__ __aicore__ PTO_INLINE void LoadPlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol, size_t idx3) {
        size_t offsetDstBase =  idx3*gShape3*TileData::Cols;
        for (uint32_t r = 0; r < gShape3; r++) {
            for (uint32_t c = 0; c < gShape4; c++) {
                size_t offsetDst = offsetDstBase + r*TileData::Cols + c;
                size_t offsetSrc = r*gStride3 + c*gStride4;
                dst[offsetDst] = src[offsetSrc];
            }
        }
    }
    template <typename GlobalData, typename TileData, std::enable_if_t<!TileData::isRowMajor, int> = 0>
    __tf__ __aicore__ PTO_INLINE void LoadPlainMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol, size_t idx3) {
        size_t offsetDstBase =  idx3*gShape4*TileData::Rows;
        for (uint32_t r = 0; r < gShape3; r++) {
            for (uint32_t c = 0; c < gShape4; c++) {
                size_t offsetDst = offsetDstBase + c*TileData::Rows + r;
                size_t offsetSrc = r*gStride3 + c*gStride4;
                dst[offsetDst] = src[offsetSrc];
            }
        }
    }

    template <typename GlobalData, typename TileData>
    __tf__ __aicore__ PTO_INLINE void LoadPlain(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol) {
        int64_t dstStride1 = gShape2;
        int64_t dstStride0 = gShape1 * dstStride1;

        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t dstAddr0 = i * dstStride0;
            int64_t srcAddr0 = i * gStride0;
            for (uint32_t j = 0; j < gShape1; j++) {
                int64_t dstAddr1 = j * dstStride1;
                int64_t srcAddr1 = j * gStride1;
                for (uint32_t k = 0; k < gShape2; k++) {
                    size_t offsetSrcBase = srcAddr0 + srcAddr1 + k * gStride2;
                    LoadPlainMatrix<GlobalData, TileData>(dst, src+offsetSrcBase, gShape3, gShape4, gStride3, gStride4, validRow, validCol, dstAddr0 + dstAddr1 + k);
                }
            }
        }
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<TileData::isRowMajor, int> = 0>
    __tf__ __aicore__ PTO_INLINE void LoadSubfractalMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol) {
        // Zn layout
        for(size_t c=0; c<gShape4; c++) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            for(size_t r=0; r < gShape3; r++) {
                size_t subTileR = r / TileData::InnerRows;
                size_t innerR = r % TileData::InnerRows;

                size_t tile_idx = subTileR*TileData::Cols*TileData::InnerRows +
                    subTileC*TileData::InnerNumel + innerC*TileData::InnerRows + innerR;

                size_t gd_idx = r*gStride3 + c*gStride4;

                dst[tile_idx] = src[gd_idx];
            }
        }
    }

    template <typename GlobalData, typename TileData, std::enable_if_t<!TileData::isRowMajor, int> = 0>
    __tf__ __aicore__ PTO_INLINE void LoadSubfractalMatrix(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape3, int gShape4, int gStride3, int gStride4, int validRow, int validCol) {
        // Nz layout
        for(size_t c=0; c<gShape4; c++) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            for(size_t r=0; r < gShape3; r++) {
                size_t subTileR = r / TileData::InnerRows;
                size_t innerR = r % TileData::InnerRows;

                size_t tile_idx = subTileC*TileData::Rows*TileData::InnerCols +
                    subTileR*TileData::InnerNumel + innerR*TileData::InnerCols + innerC;
                size_t gd_idx = r*gStride3 + c*gStride4;

                dst[tile_idx] = src[gd_idx];
            }
        }
    }

    template <typename TileData, typename GlobalData>
    __tf__ AICORE void TLoad(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        assert((gShape0*gShape1*gShape2*gShape3 == validRow && gShape4==validCol && TileData::isRowMajor) ||
            (gShape0*gShape1*gShape2*gShape4 == validCol && gShape3==validRow && !TileData::isRowMajor));

        // Filling padding
        std::fill(dst,dst+(TileData::Cols*TileData::Rows),getPadValue<TileData>());

        //Filling data
        if(TileData::SFractal == SLayout::NoneBox) {
            LoadPlain<GlobalData, TileData>(dst, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2,
                gStride3, gStride4, validRow, validCol);
        } else {
            assert(gShape0==1 && gShape1==1 && gShape2==1 && "ND,DN -> Nz,Zn convertion does support only 2D GMs");
            LoadSubfractalMatrix<GlobalData, TileData>(dst, src, gShape3, gShape4, gStride3, gStride4, validRow, validCol);
        }
    }

    template <typename TileData, typename GlobalData>
    AICORE void TLOAD_IMPL(TileData &dst, GlobalData &src)
    {
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype");
        static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN , "Only ND and DN GLobal Tensors are currently supported");
        TLoad<TileData, GlobalData>(dst.data(),
            src.data(),
            src.GetShape(0),
            src.GetShape(1),
            src.GetShape(2),
            src.GetShape(3),
            src.GetShape(4),
            src.GetStride(0),
            src.GetStride(1),
            src.GetStride(2),
            src.GetStride(3),
            src.GetStride(4),
            dst.GetValidRow(),
            dst.GetValidCol());
    }
}
#endif