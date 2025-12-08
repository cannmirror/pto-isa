#ifndef TADD_HPP
#define TADD_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto{
    template<typename tile_shape>
    void TAdd_Impl(typename tile_shape::TileDType dst,
                            typename tile_shape::TileDType src0,
                            typename tile_shape::TileDType src1,
                            unsigned validRow, unsigned validCol
                        ) {
        for(size_t c=0; c<validCol; c++) {
            for(size_t r=0; r<validRow; r++) {
                size_t idx = GetTileElementOffset<tile_shape>(r,c);
                dst[idx] = src0[idx] + src1[idx];
            }
        }
    }

    template <typename tile_shape>
    __aicore__ PTO_INLINE void TADD_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
        unsigned row = dst.GetValidRow();
        unsigned col = dst.GetValidCol();
        TAdd_Impl<tile_shape>(dst.data(), src0.data(), src1.data(), row, col);
    }
}
#endif