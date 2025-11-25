#ifndef TMRGSORT_HPP
#define TMRGSORT_HPP

#include "common/constants.hpp"
#define TRUE 1
#define FALSE 0
#define STRUCTSIZE 8
#define UBSIZE 262144  // 256 * 1024 B
#define ELEMSIZE 4

namespace pto
{
    struct MrgSortExecutedNumList {
        uint16_t mrgSortList0;
        uint16_t mrgSortList1;
        uint16_t mrgSortList2;
        uint16_t mrgSortList3;
    };

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              typename Src2TileData, bool exhausted>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1,
                                        Src2TileData &src2) {
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              bool exhausted>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1) {
    }

    // blockLen大小包含值+索引，比如32个值+索引：blockLen=64
    template <typename DstTileData, typename SrcTileData>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t blockLen) {
    }

    template <typename Src0TileData, typename Src1TileData, typename Src2TileData, typename Src3TileData>
    __aicore__ PTO_INLINE constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols + Src3TileData::Cols;
    }

    template <typename Src0TileData, typename Src1TileData, typename Src2TileData>
    __aicore__ PTO_INLINE constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols;
    }

    template <typename Src0TileData, typename Src1TileData>
    __aicore__ PTO_INLINE constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols;
    }
}
#endif