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

constexpr const int SHIFT_FRACTAL_NZ_ROW = 4; // 2^4 = 16
constexpr const int KHALF = 2;                // for b4 data
constexpr const int M_STEP_MIN_VAL_B8 = 2;    // m_step per loop

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
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr bool isFp4Type = std::is_same<DataType, float4_e2m1x2_t>::value || std::is_same<DataType, float4_e1m2x2_t>::value;
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        static_assert((srcRow % FRACTAL_NZ_ROW) == 0, "srcRow must be aligned to 16");
        static_assert((srcCol % c0Size) == 0, "srcCol must be aligned to C0Size");
        static_assert((dstRow % FRACTAL_NZ_ROW) == 0, "dstRow must be aligned to 16");
        static_assert((dstCol % c0Size) == 0, "dstCol must be aligned to C0Size");

        uint16_t mStartPosition = indexRow >> SHIFT_FRACTAL_NZ_ROW;   
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_FRACTAL_NZ_ROW;   
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_FRACTAL_NZ_ROW;
        constexpr uint16_t dstStride = dstRow >> SHIFT_FRACTAL_NZ_ROW;   

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

        uint16_t mStartPosition = indexCol >> SHIFT_FRACTAL_NZ_ROW;   
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_FRACTAL_NZ_ROW;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_FRACTAL_NZ_ROW;
        constexpr uint16_t dstStride = dstRow >> SHIFT_FRACTAL_NZ_ROW;

        if constexpr (isFp4Type) {
            load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride, 1);
        } else {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToACompact(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol, uint16_t madM, uint16_t madK) 
{
    using DataType = typename SrcTileData::DType;
    static_assert(!std::is_same<DataType, float4_e2m1x2_t>::value && !std::is_same<DataType, float4_e1m2x2_t>::value, 
        "TExtractCompact: fp4 type is currently not supported!");

    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint16_t madMAlign = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexRow >> SHIFT_FRACTAL_NZ_ROW;   
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madMAlign >> SHIFT_FRACTAL_NZ_ROW;   
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_FRACTAL_NZ_ROW;
    uint16_t dstStride = madMAlign >> SHIFT_FRACTAL_NZ_ROW;   

    load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToATransCompact(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol, uint16_t madM, uint16_t madK) 
{
    using DataType = typename SrcTileData::DType;
    static_assert(!std::is_same<DataType, float4_e2m1x2_t>::value && !std::is_same<DataType, float4_e1m2x2_t>::value, 
        "TExtractCompact: fp4 type is currently not supported!");
    
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madMAlign = CeilDivision(madM, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexCol >> SHIFT_FRACTAL_NZ_ROW;   
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_FRACTAL_NZ_ROW;
    uint8_t kStep = (madMAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_FRACTAL_NZ_ROW;
    uint16_t dstStride = madMAlign >> SHIFT_FRACTAL_NZ_ROW;

    if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t mLoop = mStep / M_STEP_MIN_VAL_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for(uint16_t idx = 0; idx < mLoop; ++idx) {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
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
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    constexpr bool isFp4Type = std::is_same<DataType, float4_e2m1x2_t>::value || std::is_same<DataType, float4_e1m2x2_t>::value;
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;
    
    if constexpr (!Transpose) {
        static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((srcCol % FRACTAL_NZ_ROW) == 0, "srcCol must be aligned to 16");
        static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
        static_assert((dstCol % FRACTAL_NZ_ROW) == 0, "dstCol must be aligned to 16"); 

        uint16_t mStartPosition = indexCol >> SHIFT_FRACTAL_NZ_ROW;   
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_FRACTAL_NZ_ROW;   
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_FRACTAL_NZ_ROW;
        constexpr uint16_t dstStride = dstCol >> SHIFT_FRACTAL_NZ_ROW; 

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

        uint16_t mStartPosition = indexRow >> SHIFT_FRACTAL_NZ_ROW;   
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_FRACTAL_NZ_ROW;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_FRACTAL_NZ_ROW;
        constexpr uint16_t dstStride = dstCol >> SHIFT_FRACTAL_NZ_ROW;

        if constexpr (isFp4Type) {
            load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride, dstStride, 1);
        } else {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBCompact(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol, uint16_t madK, uint16_t madN) 
{
    using DataType = typename SrcTileData::DType;
    static_assert(!std::is_same<DataType, float4_e2m1x2_t>::value && !std::is_same<DataType, float4_e1m2x2_t>::value, 
        "TExtractCompact: fp4 type is currently not supported!");
    
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint16_t madNAlign = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_FRACTAL_NZ_ROW;   
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madNAlign >> SHIFT_FRACTAL_NZ_ROW;   
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_FRACTAL_NZ_ROW;
    uint16_t dstStride = madNAlign >> SHIFT_FRACTAL_NZ_ROW; 

    load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBTransCompact(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol, uint16_t madK, uint16_t madN) 
{
    using DataType = typename SrcTileData::DType;
    static_assert(!std::is_same<DataType, float4_e2m1x2_t>::value && !std::is_same<DataType, float4_e1m2x2_t>::value, 
        "TExtractCompact: fp4 type is currently not supported!");
    
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madNAlign = CeilDivision(madN, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexRow >> SHIFT_FRACTAL_NZ_ROW;   
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_FRACTAL_NZ_ROW;   
    uint8_t kStep = (madNAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_FRACTAL_NZ_ROW;
    uint16_t dstStride = madNAlign >> SHIFT_FRACTAL_NZ_ROW; 
    
    if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t nLoop = mStep / M_STEP_MIN_VAL_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for(uint16_t idx = 0; idx < nLoop; ++idx) {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
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
        uint32_t alignRow = (dstValidRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        uint16_t blockLen = alignRow * c0Size * sizeof(T) / BLOCK_BYTE_SIZE;
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
        (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor), "TExtract: SrcTile Invalid Fractal");

    if constexpr (DstTileData::Loc == TileType::Left) {
        static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
            "TExtract: DstTile Invalid Fractal");

        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            if constexpr (DstTileData::Compact == CompactMode::Normal) {
                TExtractToACompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, 
                    dst.GetValidRow(), dst.GetValidCol());
            } else {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            }
        } else {
            if constexpr (DstTileData::Compact == CompactMode::Normal) {
                TExtractToATransCompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, 
                    dst.GetValidRow(), dst.GetValidCol());
            } else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }
        }
    } else if constexpr (DstTileData::Loc == TileType::Right){
        static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
            "TExtract: DstTile Invalid Fractal");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            if constexpr (DstTileData::Compact == CompactMode::Normal) {
                TExtractToBCompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, 
                    dst.GetValidRow(), dst.GetValidCol());
            } else {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            }
        } else {
            if constexpr (DstTileData::Compact == CompactMode::Normal) {
                TExtractToBTransCompact<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, 
                    dst.GetValidRow(), dst.GetValidCol());
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }
        }
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol,
            src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
    }
}
} // namespace pto
#endif // TEXTRACT_HPP