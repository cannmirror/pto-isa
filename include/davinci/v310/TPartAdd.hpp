#ifndef TPATIALADD_HPP
#define TPATIALADD_HPP

#include "TAdd.hpp"
#include "TCopy.hpp"

namespace pto {
template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ __aicore__ PTO_INLINE void TPartAdd(typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0, typename TileData::TileDType __in__ src1, unsigned src0ValidRow,
    unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    bool condSrc0EqDst = (src0ValidRow == dstValidRow && src0ValidCol == dstValidCol);
    bool condSrc0LtDst = (src0ValidRow < dstValidRow && src0ValidCol <= dstValidCol) ||
                         (src0ValidRow <= dstValidRow && src0ValidCol < dstValidCol);
    bool condSrc1EqDst = (src1ValidRow == dstValidRow && src1ValidCol == dstValidCol);
    bool condSrc1LtDst = (src1ValidRow < dstValidRow && src1ValidCol <= dstValidCol) ||
                         (src1ValidRow <= dstValidRow && src1ValidCol < dstValidCol);

    if (condSrc0EqDst && condSrc1EqDst) {  // src0 == src1 == dst
        unsigned validRow = dstValidRow;
        TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, dstValidRow, dstValidCol);
    } else if (condSrc0LtDst && condSrc1EqDst) {  // src0 < dst && src1 == dst
        TCopy<TileData, TileData, TCopyMode::DEEP_COPY, blockSizeElem, rowStride, rowStride>(
            dst, src1, src1ValidRow, src1ValidCol);
        if ((src0ValidRow != 0) && (src0ValidCol != 0)) {
            TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, dst, src0ValidRow, src0ValidCol);
        }
    } else if (condSrc1LtDst && condSrc0EqDst) {  // src1 < dst && src0 == dst
        TCopy<TileData, TileData, TCopyMode::DEEP_COPY, blockSizeElem, rowStride, rowStride>(
            dst, src0, src0ValidRow, src0ValidCol);
        if ((src1ValidRow != 0) && (src1ValidCol != 0)) {
            TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src1, dst, src1ValidRow, src1ValidCol);
        }
    }  // unsupport other conditions
}  // end tf

template <typename TileData>
__aicore__ PTO_INLINE void TPARTADD(TileData &dst, TileData &src0, TileData &src1)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;

    unsigned src0ValidRow = src0.GetValidRow(), src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow(), src1ValidCol = src1.GetValidCol();
    unsigned dstValidRow = dst.GetValidRow(), dstValidCol = dst.GetValidCol();

    TPartAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(),
        src0.data(),
        src1.data(),
        src0ValidRow,
        src0ValidCol,
        src1ValidRow,
        src1ValidCol,
        dstValidRow,
        dstValidCol);
}
}  // namespace pto
#endif