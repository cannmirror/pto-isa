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
        constexpr const int32_t LOG2_BLOCK_BYTE_SIZE = 5;   // 2^5 = 32
        constexpr const int32_t LOG2_BLOCK_LEN = 4;         // 2^4 = 16
        constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
        __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
        __cbuf__ SrcType *srcAddrP = srcAddr;
        __ca__ DstType *dstAddr = (__ca__ DstType *)(dst);
        __ca__ DstType *dstAddrP = dstAddr;

        uint8_t repeatTimes = 0;
        uint16_t srcStride = 0;
        uint16_t dstGap = 0;
        uint16_t startIdx = 0;
        uint16_t dstFracGap = 0;
        // The number of elements in a 512B fractal matrix
        uint16_t blockNum = CUBE_BLOCK_SIZE >> (sizeof(SrcType) == 1 ? 0
                                    : sizeof(SrcType) == 2 ? 1
                                    : sizeof(SrcType) == 4 ? 2
                                    : 0);
        if constexpr (!Transpose) {
            // srcRow/srcCol/dstRow/dstCol对齐校验
            static_assert((srcRow % 16) == 0, "srcRow must be aligned to 16");
            static_assert((srcCol % c0Size) == 0, "srcCol must be aligned to C0Size");
            static_assert((dstRow % 16) == 0, "dstRow must be aligned to 16");
            static_assert((dstCol % c0Size) == 0, "dstCol must be aligned to C0Size");
            // 计算源矩阵、目标矩阵行列中512B小分型矩阵的个数
            constexpr uint16_t srcColNum = (srcCol * sizeof(SrcType)) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint16_t srcRowNum = srcRow >> LOG2_BLOCK_LEN;
            constexpr uint16_t dstColNum = (dstCol * sizeof(DstType)) >> LOG2_BLOCK_BYTE_SIZE;
            constexpr uint16_t dstRowNum = dstRow >> LOG2_BLOCK_LEN;
            uint16_t startIdx0 = (indexRow >> LOG2_BLOCK_LEN) + (indexCol * srcRowNum * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE);
            if constexpr (dstRowNum >= dstColNum) {
                repeatTimes = dstRowNum;
                srcStride = 1;
                dstGap = dstColNum - 1;
                for (uint16_t i = 0; i < dstColNum; i++) {
                    startIdx = startIdx0 + i * srcRowNum;
                    dstAddrP = dstAddr + i * blockNum;
                    load_cbuf_to_ca(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
                }
            } else {
                repeatTimes = dstColNum;
                srcStride = srcRowNum;
                dstGap = 0;
                for (uint16_t i = 0; i < dstRowNum; i++) {
                    startIdx = startIdx0 + i;
                    dstAddrP = dstAddr + i * dstCol * BLOCK_LEN;
                    load_cbuf_to_ca(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
                }
            }
        } else {
            // L1->L0A:load_cbuf_to_ca_transpose
            static_assert((srcRow % (sizeof(SrcType) == 1 ? 32 : 16)) == 0, "srcRow must be aligned");
            static_assert((srcCol % (sizeof(SrcType) == 1 ? 32 : 16)) == 0, "srcCol must be aligned");
            static_assert((dstRow % (sizeof(DstType) == 1 ? 32 : 16)) == 0, "dstRow must be aligned");
            static_assert((dstCol % (sizeof(DstType) == 1 ? 32 : 16)) == 0, "dstCol must be aligned");
            if constexpr (sizeof(SrcType) == 1 || sizeof(SrcType) == 2) {
                // 方块矩阵的512B小分型矩阵个数
                constexpr uint16_t fractNum = (sizeof(SrcType) == 1) ? 2u : 1u;
                // 计算源矩阵、目标矩阵行列中方块矩阵的个数
                constexpr uint16_t srcColNum = srcCol >> (LOG2_BLOCK_LEN + fractNum - 1);
                constexpr uint16_t srcRowNum = srcRow * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
                constexpr uint16_t dstColNum = dstCol * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
                constexpr uint16_t dstRowNum = dstRow >> (LOG2_BLOCK_LEN + fractNum - 1);
                uint16_t startIdx0 = (indexCol >> (LOG2_BLOCK_LEN + fractNum - 1)) + (indexRow * srcColNum * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE);

                if constexpr (dstRowNum >= dstColNum) {
                    repeatTimes = dstRowNum;
                    srcStride = srcColNum;
                    dstGap = fractNum * dstColNum - 1;
                    dstFracGap = dstColNum - 1;
                    for (uint16_t i = 0; i < dstColNum; i++) {
                        startIdx = startIdx0 + i;
                        dstAddrP = dstAddr + i * blockNum;
                        load_cbuf_to_ca_transpose(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
                    }
                } else {
                    repeatTimes = dstColNum;
                    srcStride = 1;
                    dstGap = 0;
                    dstFracGap = dstColNum - 1;
                    for (uint16_t i = 0; i < dstRowNum; i++) {
                        startIdx = startIdx0 + i * srcColNum;
                        dstAddrP = dstAddr + i * dstColNum * blockNum * fractNum;
                        load_cbuf_to_ca_transpose(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
                    }
                }
            } else {
                // b32
                constexpr uint16_t stepK = dstRow, stepM = dstCol;
                uint16_t posK = indexRow, posM = indexCol;
                constexpr uint8_t Wk = 1, Hk = 1, strideW = 1, strideH = 1;
                constexpr uint8_t dilationW = 1, dilationH = 1;
                constexpr bool filterW = false, filterH = false, transpose = true, fmatrixCtrl = false;
                constexpr uint16_t sizeChannel = srcRow;
                constexpr int config = srcCol | (1u << 16);
                set_fmatrix(config);
                img2colv2_cbuf_to_ca(
                    dstAddrP, 
                    srcAddrP, 
                    stepK, 
                    stepM, 
                    posK, 
                    posM, 
                    strideW, 
                    strideH, 
                    Wk, 
                    Hk,
                    dilationW, 
                    dilationH, 
                    filterW, 
                    filterH, 
                    transpose, 
                    fmatrixCtrl, 
                    sizeChannel);
            }
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
        constexpr const int32_t LOG2_BLOCK_BYTE_SIZE = 5;  // 2^5 = 32
        constexpr const int32_t LOG2_BLOCK_LEN = 4;        // 2^4 = 16
        constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
        __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
        __cbuf__ SrcType *srcAddrP = srcAddr;
        __cb__ DstType *dstAddr = (__cb__ DstType *)(dst);
        __cb__ DstType *dstAddrP = dstAddr;

        uint8_t repeatTimes = 0;
        uint16_t srcStride = 0;
        uint16_t dstGap = 0;
        uint16_t startIdx = 0;
        uint16_t dstFracGap = 0;
        // The number of elements in a 512B fractal matrix
        uint16_t blockNum = CUBE_BLOCK_SIZE >> (sizeof(SrcType) == 1 ? 0
                                    : sizeof(SrcType) == 2 ? 1
                                    : sizeof(SrcType) == 4 ? 2
                                    : 0);
                
        if constexpr (!Transpose) {
            static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
            static_assert((srcCol % 16) == 0, "srcCol must be aligned to 16");
            static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
            static_assert((dstCol % 16) == 0, "dstCol must be aligned to 16");
            // 计算源矩阵、目标矩阵行列中512B小分型矩阵的个数
            uint32_t rowUnit = BLOCK_BYTE_SIZE / sizeof(SrcType);
            constexpr uint16_t dstRowNum = (dstRow * sizeof(DstType)) >> LOG2_BLOCK_BYTE_SIZE;//分型个数
            constexpr uint16_t dstColNum = dstCol >> LOG2_BLOCK_LEN;
            constexpr uint16_t srcColNum = srcCol >> LOG2_BLOCK_LEN;
            constexpr uint16_t srcRowNum = (srcRow * sizeof(SrcType)) >> LOG2_BLOCK_BYTE_SIZE;

            uint16_t startIdx0 = (indexRow * sizeof(SrcType) * srcColNum >> LOG2_BLOCK_BYTE_SIZE) + (indexCol >> LOG2_BLOCK_LEN);
            if constexpr (dstRowNum >= dstColNum) {
                repeatTimes = dstRowNum;
                srcStride = srcColNum;
                dstGap = dstColNum - 1;
                for (uint16_t i = 0; i < dstColNum; i++) {
                    startIdx = startIdx0 + i;
                    dstAddrP = dstAddr + i * blockNum;
                    load_cbuf_to_cb(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
                }
            } else {
                repeatTimes = dstColNum;
                srcStride = 1;
                dstGap = 0;
                for (uint16_t i = 0; i < dstRowNum; i++) {
                    startIdx = startIdx0 + i * srcColNum;
                    dstAddrP = dstAddr + i * dstCol * rowUnit;
                    load_cbuf_to_cb(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
                }
            }
        } else {
            // L1->L0B:load_cbuf_to_cb_transpose
            if constexpr (sizeof(SrcType) == 1 || sizeof(SrcType) == 2) {
            static_assert((srcRow % (sizeof(SrcType) == 1 ? 32 : 16)) == 0, "srcRow must be aligned");
            static_assert((srcCol % (sizeof(SrcType) == 1 ? 32 : 16)) == 0, "srcCol must be aligned");
            static_assert((dstRow % (sizeof(DstType) == 1 ? 32 : 16)) == 0, "dstRow must be aligned");
            static_assert((dstCol % (sizeof(DstType) == 1 ? 32 : 16)) == 0, "dstCol must be aligned");
                // 方块矩阵的512B小分型矩阵个数
                constexpr uint16_t fractNum = (sizeof(SrcType) == 1) ? 2u : 1u;
                // 计算源矩阵、目标矩阵行列中方块矩阵的个数
                constexpr uint16_t srcColNum = srcCol * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
                constexpr uint16_t srcRowNum = srcRow >> (LOG2_BLOCK_LEN + fractNum - 1);
                constexpr uint16_t dstColNum = dstCol >> (LOG2_BLOCK_LEN + fractNum - 1);
                constexpr uint16_t dstRowNum = dstRow * sizeof(DstType) >> LOG2_BLOCK_BYTE_SIZE;
                uint16_t startIdx0 = (indexRow >> (LOG2_BLOCK_LEN + fractNum - 1)) + (indexCol * sizeof(SrcType) * srcRowNum >> LOG2_BLOCK_BYTE_SIZE);

                if constexpr (dstRowNum >= dstColNum) {
                    repeatTimes = dstRowNum;
                    srcStride = 1;
                    dstGap = fractNum * dstColNum - 1;
                    dstFracGap = 0;
                    for (uint16_t i = 0; i < dstColNum; i++) {
                        startIdx = startIdx0 + i * srcRowNum;
                        dstAddrP = dstAddr + i * fractNum * blockNum;
                        load_cbuf_to_cb_transpose(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
                    }
                } else {
                    repeatTimes = dstColNum;
                    srcStride = srcRowNum;
                    dstGap = fractNum - 1;
                    dstFracGap = 0;
                    for (uint16_t i = 0; i < dstRowNum; i++) {
                        startIdx = startIdx0 + i;
                        dstAddrP = dstAddr + i * dstColNum * fractNum * blockNum;
                        load_cbuf_to_cb_transpose(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
                    }
                }
            } else {
                // b32
                constexpr uint16_t stepK = dstCol, stepM = dstRow;
                uint16_t posK = indexCol, posM = indexRow;
                constexpr uint8_t Wk = 1, Hk = 1, strideW = 1, strideH = 1;
                constexpr uint8_t dilationW = 1, dilationH = 1;
                constexpr bool filterW = false, filterH = false, transpose = false, fmatrixCtrl = false;
                constexpr uint16_t sizeChannel = srcCol;
                constexpr int config = srcRow | (1u << 16);
                set_fmatrix(config);
                img2colv2_cbuf_to_cb(
                    dstAddrP, 
                    srcAddrP, 
                    stepK, 
                    stepM, 
                    posK, 
                    posM, 
                    strideW, 
                    strideH, 
                    Wk, 
                    Hk,
                    dilationW, 
                    dilationH, 
                    filterW, 
                    filterH, 
                    transpose, 
                    fmatrixCtrl, 
                    sizeChannel);
            }
        }                    
    }
    template<typename DstTileData, typename SrcTileData>
    __aicore__ void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0) {
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
        if constexpr (DstTileData::Loc == Location::Left) {
            static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                      "TExtract: LeftTile Invalid Fractal.");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            }
            else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }
        }
        else {
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                        "TExtract: RightTile Invalid Fractal."); 
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            }  
            else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }         
        }         
    }
}
#endif  // TEXTRACT_HPP