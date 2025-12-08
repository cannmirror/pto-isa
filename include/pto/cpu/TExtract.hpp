#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include <cassert>

namespace pto
{
    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow = 0, uint32_t idxCol = 0) {
        assert(src.GetValidRow() - idxRow == dst.GetValidRow() && src.GetValidCol() - idxCol == dst.GetValidCol());
        for(size_t c = idxCol; c < src.GetValidCol(); c++) {
            const size_t subTileSrcC = c / SrcTileData::InnerCols;
            const size_t innerSrcC = c % SrcTileData::InnerCols;
            const size_t cDst = c - idxCol;
            const size_t subTileDstC = cDst / DstTileData::InnerCols;
            const size_t innerDstC = cDst % DstTileData::InnerCols;

            for(size_t r = idxRow; r < src.GetValidRow(); r++) {
                size_t srcTileIdx;
                size_t dstTileIdx;
                if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                    srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(r,c);
                } else {
                    const size_t subTileR = r / SrcTileData::InnerRows;
                    const size_t innerR = r % SrcTileData::InnerRows;
                    srcTileIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileR,innerR,subTileSrcC,innerSrcC);
                }
                const size_t rDst = r - idxRow;

                if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                    dstTileIdx = GetTileElementOffsetPlain<DstTileData>(rDst,cDst);
                } else {
                    const size_t subTileR = rDst / DstTileData::InnerRows;
                    const size_t innerR = rDst % DstTileData::InnerRows;
                    dstTileIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileR,innerR,subTileDstC,innerDstC);
                }
                dst.data()[dstTileIdx] = src.data()[srcTileIdx];
            }
        }
    }
}
#endif  // TEXTRACT_HPP
