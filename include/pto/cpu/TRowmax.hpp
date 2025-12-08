#ifndef TROWMAX_HPP
#define TROWMAX_HPP

#include <pto/common/pto_tile.hpp>
#include <cmath>

namespace pto{

    template <typename tile_shape_out, typename tile_shape_in>
    void TRowmax_Impl(typename tile_shape_out::TileDType dst,
                            typename tile_shape_in::TileDType src,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            size_t idx = GetTileElementOffset<tile_shape_in>(i, 0);
            typename tile_shape_out::DType max_val = src[idx];
            for (int j = 0; j < validCol; ++j) {
                idx = GetTileElementOffset<tile_shape_in>(i, j);
                if (src[idx] > max_val) {
                    max_val =  src[idx];
                }                
            }
            idx = GetTileElementOffset<tile_shape_out>(i, 0);
            dst[idx] = max_val;
        }
    }

    template <typename tile_shape_out, typename tile_shape_in>
    __aicore__ PTO_INLINE void TROWMAX_IMPL(tile_shape_out &dst, tile_shape_in &src) {
        unsigned row = src.GetValidRow();
        unsigned col = src.GetValidCol();
        TRowmax_Impl<tile_shape_out, tile_shape_in>(dst.data(), src.data(), row, col);
    }
}

#endif