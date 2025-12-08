#ifndef TROWSUM_HPP
#define TROWSUM_HPP
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
    template <typename TileDst, typename TileSrc>
    void TRowSum(typename TileDst::TileDType dst, typename TileSrc::TileDType src, uint16_t M, uint16_t N)
    {
        static constexpr int srcRows = TileSrc::Rows;
        static constexpr int dstRows = TileDst::Rows;

        for (uint16_t i = 0; i < M; i++) {
            typename TileDst::DType sum = 0;

            for (uint16_t j = 0; j < N; j++) {
               sum += src[GetTileElementOffset<TileSrc>(i,j)];
            }
            dst[GetTileElementOffset<TileDst>(i,0)] = sum;
        }
    }

    template <typename TileDst, typename TileSrc>
    __aicore__ PTO_INLINE void CheckRSValid()
    {
        using SrcType = TileSrc::DType;
        using DstType = TileDst::DType;
        static_assert(
            (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>) ||  // f162f16
                (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) ||  // f162f32
                (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float>)  // f322f32
            , "Not supported data type");
        static_assert(
            (TileSrc::Rows == TileDst::Rows),
            "Inconsistent number of m, n");
    }

    template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
    __aicore__ PTO_INLINE void TROWSUM_IMPL(TileDataOut &dstTile, TileDataIn &srcTile, TileDataTmp &tmp)
    {
        CheckRSValid<TileDataOut, TileDataIn>();

        uint16_t m = dstTile.GetValidRow();
        uint16_t n = dstTile.GetValidCol();

        TRowSum<TileDataOut, TileDataIn>(dstTile.data(), srcTile.data(), m, n);
    }
}
#endif