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

namespace pto {

constexpr const int LOG2_BLOCK_LEN = 4; // 2^4 = 16
constexpr const int LOG2_BLOCK_BYTE_SIZE = 5; // 2^5 = 32
constexpr const int KHALF = 2; // for b4 data

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol) 
{
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)(dst);
    constexpr bool isFp4Type = std::is_same<DataType, float4_e2m1x2_t>::value || std::is_same<DataType, float4_e1m2x2_t>::value;
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        static_assert((srcRow % FRACTAL_NZ_ROW) == 0, "srcRow must be aligned to 16");
        static_assert((srcCol % c0Size) == 0, "srcCol must be aligned to C0Size");
        static_assert((dstRow % FRACTAL_NZ_ROW) == 0, "dstRow must be aligned to 16");
        static_assert((dstCol % c0Size) == 0, "dstCol must be aligned to C0Size");

        uint16_t mStartPosition = indexRow >> LOG2_BLOCK_LEN;   
        uint16_t kStartPosition = (indexCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint8_t mStep = dstRow >> LOG2_BLOCK_LEN;   
        constexpr uint8_t kStep = (dstCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = srcRow >> LOG2_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> LOG2_BLOCK_LEN;   

        if constexpr (isFp4Type) {
            load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride, 0);
        } else {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        }
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcRow must be aligned"); //fp16, fp32 should be aligned to 16
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexCol >> LOG2_BLOCK_LEN;   
        uint16_t kStartPosition = (indexRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint8_t mStep = dstCol >> LOG2_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = srcCol >> LOG2_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> LOG2_BLOCK_LEN;

        if constexpr (isFp4Type) {
            load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride, 1);
        } else {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToB(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol) 
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)(dst);
    constexpr bool isFp4Type = std::is_same<DataType, float4_e2m1x2_t>::value || std::is_same<DataType, float4_e1m2x2_t>::value;
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;
    
    if constexpr (!Transpose) {
        static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((srcCol % FRACTAL_NZ_ROW) == 0, "srcCol must be aligned to 16");
        static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
        static_assert((dstCol % FRACTAL_NZ_ROW) == 0, "dstCol must be aligned to 16"); 

        uint16_t mStartPosition = indexCol >> LOG2_BLOCK_LEN;   
        uint16_t kStartPosition = (indexRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint8_t mStep = dstCol >> LOG2_BLOCK_LEN;   
        constexpr uint8_t kStep = (dstRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = srcCol >> LOG2_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> LOG2_BLOCK_LEN; 

        if constexpr (isFp4Type) {
            load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride, 0);
        } else {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        }
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcRow must be aligned"); //fp16, fp32 should be aligned to 16
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexRow >> LOG2_BLOCK_LEN;   
        uint16_t kStartPosition = (indexCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint8_t mStep = dstRow >> LOG2_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = srcRow >> LOG2_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> LOG2_BLOCK_LEN;

        if constexpr (isFp4Type) {
            load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride, 1);
        } else {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToMat(typename DstTileData::TileDType __out__ dst,
    typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol, uint32_t srcValidRow,
    uint32_t srcValidCol, uint32_t dstValidRow, uint32_t dstValidCol)
{
    using T = typename SrcTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t offset = SrcTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    if constexpr (SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) {
        offset = indexRow * srcValidCol + indexCol;
    }

    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src) + offset;
    __cbuf__ T *dstPtr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    if constexpr (SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) {
        uint16_t blockLen = dstValidRow * dstValidCol * sizeof(T) / BLOCK_BYTE_SIZE;
        // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, 1, blockLen, 0, 0);
    } else if constexpr (!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor)) {
        uint16_t blockCout = CeilDivision(dstValidCol, c0Size);
        uint16_t blockLen = dstValidRow * c0Size * sizeof(T) / BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = SrcTileData::Rows - DstTileData::Rows;
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, blockCout, blockLen, srcStride, 0);
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
    std::is_same<T, float>,
    std::is_same<T, float4_e2m1x2_t>,
    std::is_same<T, float4_e1m2x2_t>
>;

template <typename DstTileData, typename SrcTileData>
AICORE void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(is_textract_supported_type<typename DstTileData::DType>,
        "TExtract: Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
        half, bfloat16_t, float, float4_e2m1x2_t, float4_e1m2x2_t");

    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TExtract: Destination and Source tile data types must be the same");

    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                    (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor), 
                    "TExtract: SrcTile Invalid Fractal");

    if constexpr (DstTileData::Loc == TileType::Left) {
        static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
            "TExtract: DstTile Invalid Fractal");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else if constexpr (DstTileData::Loc == TileType::Right){
        static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
            "TExtract: DstTile Invalid Fractal");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol,
            src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
    }
}
} // namespace pto
#endif // TEXTRACT_HPP