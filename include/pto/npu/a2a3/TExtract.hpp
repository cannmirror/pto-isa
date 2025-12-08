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

constexpr const int32_t LOG2_BLOCK_BYTE_SIZE = 5; // 2^5 = 32
constexpr const int32_t LOG2_BLOCK_LEN = 4;       // 2^4 = 16

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
__tf__ __aicore__ void TExtractToANonTranspose(
    __ca__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow, uint16_t indexCol)
{
    constexpr int config = srcRow | (1u << 16);
    set_fmatrix(config);
    img2colv2_cbuf_to_ca(
        dstAddr, srcAddr, dstCol, dstRow, indexCol, indexRow, 1, 1, 1, 1, 1, 1, false, false, false, false, srcCol);
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
__tf__ __aicore__ void TExtractToATranspose(
    __ca__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow, uint16_t indexCol)
{
    // b8采用Load2D转置
    if constexpr (sizeof(SrcType) == 1) {
        constexpr uint16_t fractNum = 2;
        constexpr uint16_t srcColNum = srcCol >> (LOG2_BLOCK_LEN + fractNum - 1);
        constexpr uint16_t dstColNum = dstCol * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint16_t dstRowNum = dstRow >> (LOG2_BLOCK_LEN + fractNum - 1);
        uint16_t dstGap = 0;
        uint16_t dstFracGap = 0;
        uint16_t startIdx0 = (indexCol >> (LOG2_BLOCK_LEN + fractNum - 1)) +
                             (indexRow * srcColNum * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE);
        // 判断行优先&列优先的搬运路径，减少for循环次数
        if constexpr (dstRowNum >= dstColNum) {
            dstGap = fractNum * dstColNum - 1;
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                load_cbuf_to_ca_transpose(
                    dstAddr, srcAddr, startIdx0 + i, dstRowNum, srcColNum, dstGap, false, dstFracGap);
                dstAddr += CUBE_BLOCK_SIZE;
            }
        } else {
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                load_cbuf_to_ca_transpose(
                    dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0, false, dstFracGap);
                dstAddr += dstColNum * CUBE_BLOCK_SIZE * fractNum;
            }
        }
    } else {
        // b16和b32采用load3DV2转置，减少scalar次数
        constexpr int config = srcCol | (1u << 16);
        set_fmatrix(config);
        img2colv2_cbuf_to_ca(
            dstAddr, srcAddr, dstRow, dstCol, indexRow, indexCol, 1, 1, 1, 1, 1, 1, false, false, true, false, srcRow);
    }
}
template <typename DstTileData, typename SrcTileData, bool Transpose>
__aicore__ void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
    __ca__ DstType *dstAddr = (__ca__ DstType *)(dst);

    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;

    if constexpr (!Transpose) {
        // srcRow/srcCol/dstRow/dstCol对齐校验
        static_assert((srcRow % 16) == 0, "srcRow must be aligned to 16");
        static_assert((srcCol % c0Size) == 0, "srcCol must be aligned to C0Size");
        static_assert((dstRow % 16) == 0, "dstRow must be aligned to 16");
        static_assert((dstCol % c0Size) == 0, "dstCol must be aligned to C0Size");
        PTO_ASSERT((indexRow % 16) == 0, "indexRow must be aligned to 16");
        PTO_ASSERT((indexCol % c0Size) == 0, "indexCol must be aligned to C0Size");
        TExtractToANonTranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    } else {
        // L1->L0A:load_cbuf_to_ca_transpose
        static_assert((srcRow % fractalSize) == 0, "srcRow must be aligned");
        static_assert((srcCol % fractalSize) == 0, "srcCol must be aligned");
        static_assert((dstRow % fractalSize) == 0, "dstRow must be aligned");
        static_assert((dstCol % fractalSize) == 0, "dstCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
        TExtractToATranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    }
}
template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
__tf__ __aicore__ void TExtractToBNonTranspose(
    __cb__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow, uint16_t indexCol)
{
    uint16_t dstGap = 0;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr uint16_t dstRowNum = (dstRow * sizeof(DstType)) >> LOG2_BLOCK_BYTE_SIZE; // 分型个数
    constexpr uint16_t dstColNum = dstCol >> LOG2_BLOCK_LEN;
    constexpr uint16_t srcColNum = srcCol >> LOG2_BLOCK_LEN;
    constexpr uint16_t srcRowNum = (srcRow * sizeof(SrcType)) >> LOG2_BLOCK_BYTE_SIZE;
    // 计算源矩阵、目标矩阵行列中512B小分型矩阵的个数
    uint16_t blockNum = CUBE_BLOCK_SIZE >> (sizeof(SrcType) == 1    ? 0 :
                                               sizeof(SrcType) == 2 ? 1 :
                                               sizeof(SrcType) == 4 ? 2 :
                                                                      0);
    uint16_t startIdx0 =
        (indexRow * sizeof(SrcType) * srcColNum >> LOG2_BLOCK_BYTE_SIZE) + (indexCol >> LOG2_BLOCK_LEN);
    if constexpr (dstRowNum >= dstColNum) {
        dstGap = dstColNum - 1;
        for (uint16_t i = 0; i < dstColNum; i++) {
            load_cbuf_to_cb(
                dstAddr, srcAddr, startIdx0 + i, dstRowNum, srcColNum, dstGap, 0, false, addr_cal_mode_t(0));
            dstAddr += blockNum;
        }
    } else {
        for (uint16_t i = 0; i < dstRowNum; i++) {
            load_cbuf_to_cb(dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0, 0, false, addr_cal_mode_t(0));
            dstAddr += dstCol * c0Size;
        }
    }
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
__tf__ __aicore__ void TExtractToBTranspose(
    __cb__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow, uint16_t indexCol)
{
    // b8使用Load2D
    if constexpr (sizeof(SrcType) == 1) {
        constexpr uint16_t fractNum = 2;
        // 计算源矩阵、目标矩阵行列中方块矩阵的个数
        constexpr uint16_t srcColNum = srcCol * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
        constexpr uint16_t srcRowNum = srcRow >> (LOG2_BLOCK_LEN + fractNum - 1);
        constexpr uint16_t dstColNum = dstCol >> (LOG2_BLOCK_LEN + fractNum - 1);
        constexpr uint16_t dstRowNum = dstRow * sizeof(DstType) >> LOG2_BLOCK_BYTE_SIZE;
        uint16_t dstGap = 0;
        uint16_t startIdx0 = (indexRow >> (LOG2_BLOCK_LEN + fractNum - 1)) +
                             (indexCol * sizeof(SrcType) * srcRowNum >> LOG2_BLOCK_BYTE_SIZE);
        if constexpr (dstRowNum >= dstColNum) {
            dstGap = fractNum * dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0 + i * srcRowNum, dstRowNum, 1, dstGap, false, 0);
                dstAddr += fractNum * CUBE_BLOCK_SIZE;
            }
        } else {
            dstGap = fractNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0 + i, dstColNum, srcRowNum, dstGap, false, 0);
                dstAddr += dstColNum * fractNum * CUBE_BLOCK_SIZE;
            }
        }
    } else {
        // b16&b32使用Load3DV2
        constexpr int config = srcRow | (1u << 16);
        set_fmatrix_b(config);
        img2colv2_cbuf_to_cb(
            dstAddr, srcAddr, dstCol, dstRow, indexCol, indexRow, 1, 1, 1, 1, 1, 1, false, false, false, true, srcCol);
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
__aicore__ void TExtractToB(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
    __cb__ DstType *dstAddr = (__cb__ DstType *)(dst);
    if constexpr (!Transpose) {
        static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((srcCol % 16) == 0, "srcCol must be aligned to 16");
        static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
        static_assert((dstCol % 16) == 0, "dstCol must be aligned to 16");
        PTO_ASSERT((indexRow % c0Size) == 0, "indexRow must be aligned to c0Size");
        PTO_ASSERT((indexCol % 16) == 0, "indexCol must be aligned to 16");
        TExtractToBNonTranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    } else {
        static_assert((srcRow % fractalSize) == 0, "srcRow must be aligned");
        static_assert((srcCol % fractalSize) == 0, "srcCol must be aligned");
        static_assert((dstRow % fractalSize) == 0, "dstRow must be aligned");
        static_assert((dstCol % fractalSize) == 0, "dstCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
        TExtractToBTranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    }
}
template <typename DstTileData, typename SrcTileData>
__aicore__ void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TExtract: Destination and Source tile data types must be the same.");
    static_assert(std::is_same<typename DstTileData::DType, int8_t>::value ||
                      std::is_same<typename DstTileData::DType, half>::value ||
                      std::is_same<typename DstTileData::DType, bfloat16_t>::value ||
                      std::is_same<typename DstTileData::DType, float>::value,
        "TExtract: Invalid data type.");
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
        "TExtract: SrcTile Invalid Fractal.");
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    if constexpr (DstTileData::Loc == Location::Left) {
        static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
            "TExtract: LeftTile Invalid Fractal.");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
            "TExtract: RightTile Invalid Fractal.");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}
} // namespace pto
#endif // TEXTRACT_HPP