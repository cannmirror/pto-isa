#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

namespace pto 
{
    template <typename DstTileData, typename SrcTileData>
    __tf__ __aicore__ void TExtract(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
        uint16_t indexRow, uint16_t indexCol) {
        using SrcType = typename SrcTileData::DType;
        using DstType = typename DstTileData::DType;
        constexpr int32_t srcRow = SrcTileData::Rows;
        constexpr int32_t srcCol = SrcTileData::Cols;
        constexpr int32_t dstRow = DstTileData::Rows;
        constexpr int32_t dstCol = DstTileData::Cols;
        constexpr const int LOG2_BLOCK_LEN = 4;
        constexpr const int LOG2_BLOCK_BYTE_SIZE = 5;
        constexpr int typeSize = sizeof(SrcType);
        __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)(src);
        __ca__ DstType *dstAddr = (__ca__ DstType *)(dst);

        uint16_t mStartPosition = 0;
        uint16_t kStartPosition = 0; 
        uint8_t mStep = 0;
        uint8_t kStep = 0;
        uint16_t srcStride = 0;
        uint16_t dstStride = 0;

        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            // 非转置场景 src sRowMajor
            mStartPosition = indexRow >> LOG2_BLOCK_LEN;   
            kStartPosition = (indexCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            mStep = dstRow >> LOG2_BLOCK_LEN;   
            kStep = (dstCol * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            srcStride = SrcTileData::isRowMajor ? 1 : srcRow >> LOG2_BLOCK_LEN;
            dstStride = dstRow >> LOG2_BLOCK_LEN;   

            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        } else {
            // 转置场景 src sColMajor
            mStartPosition = indexRow >> LOG2_BLOCK_LEN;   
            kStartPosition = indexCol >> LOG2_BLOCK_LEN;
            mStep = dstCol >> LOG2_BLOCK_LEN;
            kStep = dstRow >> LOG2_BLOCK_LEN; 
            srcStride = SrcTileData::isRowMajor ? 1 : (srcRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;
            dstStride = (dstRow * typeSize) >> LOG2_BLOCK_BYTE_SIZE;

            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }

    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol) {
        TExtract<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol);
    }
}
#endif