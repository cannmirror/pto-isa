#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"

namespace pto
{
    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src) {
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
            "The shape of src needs to be the same as that of dst.");
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            if constexpr (DstTileData::Loc == Location::Left) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            }
        } else {
            if constexpr (DstTileData::Loc == Location::Left) {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        }
    }
}
#endif  // TMOV_HPP