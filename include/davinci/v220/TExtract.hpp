#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

namespace pto {
    template <typename DstTileData, typename SrcTileData, bool Transpose>
    __tf__ __aicore__ void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
        uint16_t indexRow, uint16_t indexCol) {
        using SrcType = typename SrcTileData::DType;
        using DstType = typename DstTileData::DType;
        constexpr int32_t srcRow = SrcTileData::Rows;
        constexpr int32_t srcCol = SrcTileData::Cols;
        constexpr int32_t dstRow = DstTileData::Rows;
        constexpr int32_t dstCol = DstTileData::Cols;
        constexpr const int LOG2_BLOCK_BYTE_SIZE = 5;  // 2^5 = 32
        constexpr const int LOG2_BLOCK_LEN = 4;  // 2^4 = 16
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
            static_assert((srcCol % (sizeof(SrcType) == 1 ? 32 : sizeof(SrcType) == 2 ? 16 : 8)) == 0, "srcCol must be aligned to C0Size");
            static_assert((dstRow % 16) == 0, "dstRow must be aligned to 16");
            static_assert((dstCol % (sizeof(DstType) == 1 ? 32 : sizeof(DstType) == 2 ? 16 : 8)) == 0, "dstCol must be aligned to C0Size");
            // 计算源矩阵、目标矩阵行列中512B小分型矩阵的个数
            uint16_t srcColNum = (srcCol * sizeof(SrcType)) >> LOG2_BLOCK_BYTE_SIZE;
            uint16_t srcRowNum = srcRow >> LOG2_BLOCK_LEN;
            uint16_t dstColNum = (dstCol * sizeof(DstType)) >> LOG2_BLOCK_BYTE_SIZE;
            uint16_t dstRowNum = dstRow >> LOG2_BLOCK_LEN;
            uint16_t startIdx0 = (indexRow >> LOG2_BLOCK_LEN) + (indexCol * srcRowNum * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE);
            if (dstRow >= dstCol) {
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
            static_assert((srcRow % (sizeof(SrcType) == 1 ? 32 : sizeof(SrcType) == 2 ? 16 : 16)) == 0, "srcRow must be aligned to 16");
            static_assert((srcCol % (sizeof(SrcType) == 1 ? 32 : sizeof(SrcType) == 2 ? 16 : 16)) == 0, "srcCol must be aligned to C0Size");
            static_assert((dstRow % (sizeof(DstType) == 1 ? 32 : sizeof(DstType) == 2 ? 16 : 16)) == 0, "dstRow must be aligned to 16");
            static_assert((dstCol % (sizeof(DstType) == 1 ? 32 : sizeof(DstType) == 2 ? 16 : 16)) == 0, "dstCol must be aligned to C0Size");
            if constexpr (sizeof(SrcType) == 1 || sizeof(SrcType) == 2) {
                // 方块矩阵的512B小分型矩阵个数
                constexpr uint16_t fractNum = (sizeof(SrcType) == 1) ? 2u : 1u;
                // 计算源矩阵、目标矩阵行列中方块矩阵的个数
                uint16_t srcColNum = srcCol >> (LOG2_BLOCK_LEN + fractNum - 1);
                uint16_t srcRowNum = srcRow * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
                uint16_t dstColNum = dstCol * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE;
                uint16_t dstRowNum = dstRow >> (LOG2_BLOCK_LEN + fractNum - 1);
                uint16_t startIdx0 = (indexCol >> (LOG2_BLOCK_LEN + fractNum - 1)) + (indexRow * srcColNum * sizeof(SrcType) >> LOG2_BLOCK_BYTE_SIZE);

                if (dstRowNum >= dstColNum) {
                    repeatTimes = dstRowNum;
                    srcStride = srcColNum;
                    dstGap = fractNum * dstColNum - 1;
                    dstFracGap = dstColNum - 1;
                    for (int i = 0; i < dstColNum; i++) {
                        startIdx = startIdx0 + i;
                        dstAddrP = dstAddr + i * blockNum;
                        load_cbuf_to_ca_transpose(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
                    }
                } else {
                    repeatTimes = dstColNum;
                    srcStride = 1;
                    dstGap = 0;
                    dstFracGap = dstColNum - 1;
                    for (int i = 0; i < dstRowNum; i++) {
                        startIdx = startIdx0 + i * srcColNum;
                        dstAddrP = dstAddr + i * dstColNum * blockNum * fractNum;
                        load_cbuf_to_ca_transpose(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, false, dstFracGap);
                    }
                }
            } else {
                // b32
                uint16_t stepK = dstRow, stepM = dstCol;
                uint16_t posK = indexRow, posM = indexCol;
                uint8_t Wk = 1, Hk = 1, strideW = 1, strideH = 1;
                uint8_t dilationW = 1, dilationH = 1;
                bool filterW = false, filterH = false, transpose = true, fmatrixCtrl = false;
                uint16_t sizeChannel = srcRow;
                constexpr int config = srcCol | (1u << 16);
                set_fmatrix(config);
                img2colv2_cbuf_to_ca(
                    dstAddrP, srcAddrP, stepK, stepM, posK, posM, strideW, strideH, Wk, Hk,
                    dilationW, dilationH, filterW, filterH, transpose, fmatrixCtrl, sizeChannel);
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

        __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
        __cbuf__ SrcType *srcAddrP = srcAddr;
        __cb__ DstType *dstAddr = (__cb__ DstType *)(dst);
        __cb__ DstType *dstAddrP = dstAddr;

        uint8_t repeatTimes = 0;
        uint16_t srcStride = 0;
        uint16_t dstGap = 0;
        uint16_t startIdx = 0;
        if constexpr (!Transpose) {
            uint32_t rowUnit = BLOCK_BYTE_SIZE / sizeof(SrcType);
            uint32_t colUnit = BLOCK_LEN;
            if (dstRow >= dstCol) {
                repeatTimes = dstRow / rowUnit;
                srcStride = srcCol / colUnit;
                dstGap = dstCol / colUnit - 1;
                for (uint32_t i = 0; i < dstCol / colUnit; i++) {
                    startIdx = indexRow / rowUnit * (srcCol / colUnit) + (indexCol / colUnit + i);
                    dstAddrP = dstAddr + i * CUBE_BLOCK_SIZE / sizeof(SrcType);
                    load_cbuf_to_cb(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
                }
            } else {
                repeatTimes = dstCol / colUnit;
                srcStride = 1;
                dstGap = 0;
                for (uint32_t i = 0; i < dstRow / rowUnit; i++) {
                    startIdx = (indexRow / rowUnit + i) * (srcCol / colUnit) + indexCol / colUnit;
                    dstAddrP = dstAddr + i * dstCol * rowUnit;
                    load_cbuf_to_cb(dstAddrP, srcAddrP, startIdx, repeatTimes, srcStride, dstGap, 0, false, addr_cal_mode_t(0));
                }
            }
        } else {
            // L1->L0B:load_cbuf_to_cb_transpose
        }
    }

    template<typename DstTileData, typename SrcTileData>
    __aicore__ void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0) {
        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TEXTRACT: Destination and Source tile data types must be the same.");
        static_assert(std::is_same<typename DstTileData::DType, int8_t>::value ||
            std::is_same<typename DstTileData::DType, half>::value ||
            std::is_same<typename DstTileData::DType, bfloat16_t>::value ||
            std::is_same<typename DstTileData::DType, float>::value,
            "TEXTRACT: Invalid data type");
        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                      "TEXTRACT: SrcTile Invalid Fractal.");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            if constexpr (DstTileData::Loc == Location::Left) {
                static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                              "TEXTRACT: Invalid Fractal.");
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            } else {
                static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                              "TEXTRACT: Invalid Fractal.");
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
            }
        } else {
            if constexpr (DstTileData::Loc == Location::Left) {
                static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                              "TEXTRACT: Invalid Fractal.");
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            } else {
                static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                              "TEXTRACT: Invalid Fractal.");
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
            }
        }
    }
}
#endif  // TEXTRACT_HPP