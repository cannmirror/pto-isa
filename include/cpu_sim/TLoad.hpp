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

namespace pto {
    template <typename TileData>
    __aicore__ constexpr auto getPadValue()
    {
        if constexpr (std::is_same<typename TileData::DType, float>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint32_t(0);
                case PadValue::Min: return uint32_t(0xff800000UL);
                case PadValue::Max: return uint32_t(0x7f800000UL);
            }
        } else if constexpr (std::is_same<typename TileData::DType, int32_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint32_t(0);
                case PadValue::Min: return uint32_t(0xffffffffUL);
                case PadValue::Max: return uint32_t(0x7fffffffUL);
            }
        } else if constexpr (std::is_same<typename TileData::DType, uint32_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero:
                case PadValue::Min: return uint32_t(0);
                case PadValue::Max: return uint32_t(0xffffffffUL);
            }
        } else if constexpr (std::is_same<typename TileData::DType, bfloat16_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint16_t(0);
                case PadValue::Min: return uint16_t(0xff80);
                case PadValue::Max: return uint16_t(0x7f80);
            }
        } else if constexpr (std::is_same<typename TileData::DType, half>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint16_t(0);
                case PadValue::Min: return uint16_t(0xfc00);
                case PadValue::Max: return uint16_t(0x7c00);
            }
        } else if constexpr (std::is_same<typename TileData::DType, int16_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint16_t(0);
                case PadValue::Min: return uint16_t(0xffff);
                case PadValue::Max: return uint16_t(0x7fff);
            }
        } else if constexpr (std::is_same<typename TileData::DType, uint16_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero:
                case PadValue::Min: return uint16_t(0);
                case PadValue::Max: return uint16_t(0xffff);
            }
        } else if constexpr (std::is_same<typename TileData::DType, int8_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint8_t(0);
                case PadValue::Min: return uint8_t(0xff);
                case PadValue::Max: return uint8_t(0x7f);
            }
        } else if constexpr (std::is_same<typename TileData::DType, uint8_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero:
                case PadValue::Min: return uint8_t(0);
                case PadValue::Max: return uint8_t(0xff);
            }
        } else {
            static_assert(sizeof(TileData::DType) < 0, "TLOAD: Unsupported DType for PadValue");
        }
    }

    template <typename TileData, typename GlobalData>
    __tf__ __aicore__ void TLoad(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            int64_t dstStride2 = gShape3 * TileData::Cols;
            int64_t dstStride1 = gShape2 * dstStride2;
            int64_t dstStride0 = gShape1 * dstStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t dstAddr0 = i * dstStride0;
                int64_t srcAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t dstAddr1 = j * dstStride1;
                    int64_t srcAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        int64_t dstAddr2 = k * dstStride2;
                        int64_t srcAddr2 = k * gStride2;
                        for (uint32_t l = 0; l < validRow; l++) {
                            for (uint32_t m = 0; m < validCol; m++) {
                                size_t offsetDst =  dstAddr0 + dstAddr1 + dstAddr2 + l*TileData::Cols + m;
                                size_t offsetSrc =  srcAddr0 + srcAddr1 + srcAddr2 + l*gStride3 + m*gStride4;
                                dst[offsetDst] = src[offsetSrc];
                            }
                        }
                    }
                }
            }
        } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            int64_t dstStride2 = gShape4 * TileData::Rows;
            int64_t dstStride1 = gShape2 * dstStride2;
            int64_t dstStride0 = gShape1 * dstStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t dstAddr0 = i * dstStride0;
                int64_t srcAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t dstAddr1 = j * dstStride1;
                    int64_t srcAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        int64_t dstAddr2 = k * dstStride2;
                        int64_t srcAddr2 = k * gStride2;
                        for (uint32_t l = 0; l < validRow; l++) {
                            for (uint32_t m = 0; m < validCol; m++) {
                                size_t offsetDst =  dstAddr0 + dstAddr1 + dstAddr2 + m*TileData::Rows + l;
                                size_t offsetSrc =  srcAddr0 + srcAddr1 + srcAddr2 + l*gStride3 + m*gStride4;
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

                    dst[tile_idx] = src[gd_idx];
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

                    dst[tile_idx] = src[gd_idx];
                }
            }
        }
    }

    template <typename TileData, typename GlobalData>
    __aicore__ void TLOAD_IMPL(TileData &dst, GlobalData &src)
    {
        static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                      (sizeof(typename TileData::DType) == 4), "Data type must be b8/16/32");
        //static_assert(TileData::Loc == pto::Location::Vec, "Dst location must be Vec!");
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype");
        static_assert(GlobalData::layout == pto::Layout::ND, "Only ND GLobal Tensors are currently supported");
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