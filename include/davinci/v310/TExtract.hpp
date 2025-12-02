/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

namespace pto 
{
    template <typename DstTileData, typename SrcTileData, bool Transpose>
    __tf__ __aicore__ void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
        uint16_t indexRow, uint16_t indexCol) {
        using SrcType = typename SrcTileData::DType;
        using DstType = typename DstTileData::DType;
        constexpr int32_t srcRow = SrcTileData::Rows;
        constexpr int32_t srcCol = SrcTileData::Cols;
        constexpr int32_t dstRow = DstTileData::Rows;
        constexpr int32_t dstCol = DstTileData::Cols;
        constexpr const int LOG2_BLOCK_LEN = 4; // 2^4 = 16
        constexpr const int LOG2_BLOCK_BYTE_SIZE = 5; // 2^5 = 32
        constexpr const int BLOCK_BYTE_SIZE = 32;
        constexpr int typeSize = sizeof(SrcType);
        constexpr int c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
        __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
        __ca__ DstType *dstAddr = (__ca__ DstType *)(dst);

        if constexpr (!Transpose) {
            static_assert((srcRow % 16) == 0, "srcRow must be aligned to 16");
            static_assert((srcCol % c0Size) == 0, "srcCol must be aligned to C0Size");
            static_assert((dstRow % 16) == 0, "dstRow must be aligned to 16");
            static_assert((dstCol % c0Size) == 0, "dstCol must be aligned to C0Size");

            uint16_t mStartPosition = indexRow >> LOG2_BLOCK_LEN;   
            uint16_t kStartPosition = (indexCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint8_t mStep = dstRow >> LOG2_BLOCK_LEN;   
            constexpr uint8_t kStep = (dstCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint16_t srcStride = srcRow >> LOG2_BLOCK_LEN;
            constexpr uint16_t dstStride = dstRow >> LOG2_BLOCK_LEN;   

            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        } else {
            static_assert((srcRow % (typeSize == 1 ? 32 : 16)) == 0, "srcRow must be aligned");
            static_assert((srcCol % (typeSize == 1 ? 32 : 16)) == 0, "srcCol must be aligned");
            static_assert((dstRow % (typeSize == 1 ? 32 : 16)) == 0, "dstRow must be aligned");
            static_assert((dstCol % (typeSize == 1 ? 32 : 16)) == 0, "dstCol must be aligned");

            uint16_t mStartPosition = indexCol >> LOG2_BLOCK_LEN;   
            uint16_t kStartPosition = (indexRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint8_t mStep = dstCol >> LOG2_BLOCK_LEN;
            constexpr uint8_t kStep = (dstRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint16_t srcStride = srcCol >> LOG2_BLOCK_LEN;
            constexpr uint16_t dstStride = dstRow >> LOG2_BLOCK_LEN;

            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }

    template <typename DstTileData, typename SrcTileData, bool Transpose>
    __tf__ __aicore__ void TExtractToB(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
        uint16_t indexRow, uint16_t indexCol) {
        using SrcType = typename SrcTileData::DType;
        using DstType = typename DstTileData::DType;
        constexpr int32_t srcRow = SrcTileData::Rows;
        constexpr int32_t srcCol = SrcTileData::Cols;
        constexpr int32_t dstRow = DstTileData::Rows;
        constexpr int32_t dstCol = DstTileData::Cols;
        constexpr const int LOG2_BLOCK_LEN = 4; // 2^4 = 16
        constexpr const int LOG2_BLOCK_BYTE_SIZE = 5; // 2^5 = 32
        constexpr const int BLOCK_BYTE_SIZE = 32;
        constexpr int typeSize = sizeof(SrcType);
        constexpr int c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
        __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
        __cb__ DstType *dstAddr = (__cb__ DstType *)(dst);

        if constexpr (!Transpose) {
            static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
            static_assert((srcCol % 16) == 0, "srcCol must be aligned to 16");
            static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
            static_assert((dstCol % 16) == 0, "dstCol must be aligned to 16"); 

            uint16_t mStartPosition = indexCol >> LOG2_BLOCK_LEN;   
            uint16_t kStartPosition = (indexRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint8_t mStep = dstCol >> LOG2_BLOCK_LEN;   
            constexpr uint8_t kStep = (dstRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint16_t srcStride = srcCol >> LOG2_BLOCK_LEN;
            constexpr uint16_t dstStride = dstCol >> LOG2_BLOCK_LEN;   

            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        } else {
            static_assert((srcRow % (typeSize == 1 ? 32 : 16)) == 0, "srcRow must be aligned");
            static_assert((srcCol % (typeSize == 1 ? 32 : 16)) == 0, "srcCol must be aligned");
            static_assert((dstRow % (typeSize == 1 ? 32 : 16)) == 0, "dstRow must be aligned");
            static_assert((dstCol % (typeSize == 1 ? 32 : 16)) == 0, "dstCol must be aligned");

            uint16_t mStartPosition = indexRow >> LOG2_BLOCK_LEN;   
            uint16_t kStartPosition = (indexCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint8_t mStep = dstRow >> LOG2_BLOCK_LEN;
            constexpr uint8_t kStep = (dstCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint16_t srcStride = srcRow >> LOG2_BLOCK_LEN;
            constexpr uint16_t dstStride = dstCol >> LOG2_BLOCK_LEN;

            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }

    template<typename T>
    constexpr bool is_textract_supported_type = std::disjunction_v<
        std::is_same<T, int8_t>,
        std::is_same<T, float8_e4m3_t>,
        std::is_same<T, float8_e5m2_t>,
        std::is_same<T, hifloat8_t>,
        std::is_same<T, half>,
        std::is_same<T, bfloat16_t>,
        std::is_same<T, float>
    >;

    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol) {
        static_assert(is_textract_supported_type<typename DstTileData::DType>,
            "Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
            half, bfloat16_t, float");

        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TExtract: Destination and Source tile data types must be the same");

        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor), 
                      "TExtract: SrcTile Invalid Fractal");

        if constexpr (DstTileData::Loc == Location::Left) {
            static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                "TExtract: DstTile Invalid Fractal");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            } else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }
        } else if constexpr (DstTileData::Loc == Location::Right){
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                "TExtract: DstTile Invalid Fractal");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }
        }
    }
}
#endif