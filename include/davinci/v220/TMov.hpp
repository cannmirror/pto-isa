#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"

namespace pto
{
    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src) {
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
            "TMov: The shape of src needs to be the same as that of dst.");
        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TMov: Destination and Source tile data types must be the same.");   
        static_assert(std::is_same<typename DstTileData::DType, int8_t>::value ||
            std::is_same<typename DstTileData::DType, half>::value ||
            std::is_same<typename DstTileData::DType, bfloat16_t>::value ||
            std::is_same<typename DstTileData::DType, float>::value
            , "TMov: Invalid data type."); 
        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor)
                        , "TMov: SrcTile Invalid Fractal.");
        if constexpr (DstTileData::Loc == Location::Left) {
            static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                        "TMov: LeftTile Invalid Fractal."); 
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            }
            else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        }
        else {
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                        "TMov: RightTile Invalid Fractal.");   
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);   
            }  
            else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }                   
        }
    }
}
#endif  // TMOV_HPP
