#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"

namespace pto 
{
    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src) {
         // 增加校验
        static_assert(is_textract_supported_type<typename DstTileData::DType>,
            "Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
            half, bfloat16_t, float");

        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TMov: Destination and Source tile data types must be the same");

        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor), 
                      "TMov: SrcTile Invalid Fractal");

        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)), 
            "The shape of src needs to be the same as that of dst");

        if constexpr (DstTileData::Loc == Location::Left) {
            static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        } else if constexpr (DstTileData::Loc == Location::Right){
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        }
    }
}
#endif