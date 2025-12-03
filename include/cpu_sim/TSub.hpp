#ifndef TSUB_HPP
#define TSUB_HPP

#include "common/pto_tile.hpp"
#include "cpu_sim/tile_offsets.hpp"

namespace pto{
    template<typename tile_shape, int stride>
    void TSub_Impl(typename tile_shape::TileDType dst,
                            typename tile_shape::TileDType src0,
                            typename tile_shape::TileDType src1,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            for (int j = 0; j < validCol; ++j) {
                size_t offset = GetTileElementOffset<tile_shape>(i,j);
                dst[offset] = src0[offset] - src1[offset];
            }
        }
    }

    template <typename tile_shape>
    __aicore__ PTO_INLINE void TSUB_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
        unsigned row = dst.GetValidRow();
        unsigned col = dst.GetValidCol();
        constexpr unsigned stride = tile_shape::RowStride;
        TSub_Impl<tile_shape, stride>(dst.data(), src0.data(), src1.data(), row, col);
    }
}
#endif