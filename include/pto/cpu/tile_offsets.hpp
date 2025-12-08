#ifndef TILE_OFFSETS_HPP
#define TILE_OFFSETS_HPP

#include <unistd.h>
namespace pto {
    template <typename TileData>
    size_t GetTileElementOffsetSubfractals( size_t subTileR, size_t innerR, size_t subTileC, size_t innerC) {
        if constexpr(!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
            // Nz
            return subTileC*TileData::Rows*TileData::InnerCols +
                subTileR*TileData::InnerNumel + innerR*TileData::InnerCols + innerC;
        } else if constexpr(TileData::isRowMajor & (TileData::SFractal == SLayout::ColMajor)) {
            // Zn
            return subTileR*TileData::Cols*TileData::InnerRows +
                subTileC*TileData::InnerNumel + innerC*TileData::InnerRows + innerR;
        } else if constexpr(TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
            // Zz
            return subTileR*TileData::Cols*TileData::InnerRows +
                subTileC*TileData::InnerNumel + innerR*TileData::InnerCols + innerC;
        } else {
            static_assert(false, "Invalid layout");
        }
    }

    template <typename TileData>
    size_t GetTileElementOffsetPlain(size_t r, size_t c) {
        if constexpr(TileData::isRowMajor) {
            return r*TileData::Cols+c;
        } else {
            return c*TileData::Rows+r;            
        }
    }

    template <typename TileData>
    size_t GetTileElementOffset(size_t r, size_t c) {
        if constexpr (TileData::SFractal == SLayout::NoneBox)
            return GetTileElementOffsetPlain<TileData>(r,c);
        else {
            size_t subTileR = r / TileData::InnerRows;
            size_t innerR = r % TileData::InnerRows;
            return GetTileElementOffsetSubfractals<TileData>(r/TileData::InnerRows, r%TileData::InnerRows,
                                                            c/TileData::InnerCols, c%TileData::InnerCols);
        }
    }

}
#endif