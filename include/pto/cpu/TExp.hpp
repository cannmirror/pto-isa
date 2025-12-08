#ifndef TEXP_HPP
#define TEXP_HPP

#include <pto/common/pto_tile.hpp>
#include <cmath>

namespace pto{

    template <typename tile_shape>
    void TExp_Impl(typename tile_shape::TileDType dst,
                            typename tile_shape::TileDType src,
                            unsigned validRow, unsigned validCol
                        ) {
        for (int i = 0; i < validRow; ++i) {
            for (int j = 0; j < validCol; ++j) {
                size_t idx = GetTileElementOffset<tile_shape>(i,j);
                if constexpr (std::is_same_v<typename tile_shape::TileDType, aclFloat16>) {
                    dst[idx] = static_cast<aclFloat16>(std::expf(static_cast<float>(src[idx])));
                } else {    
                    dst[idx] = std::exp(src[idx]);
                }
            }
        }
    }

    template <typename tile_shape>
    __aicore__ PTO_INLINE void TEXP_IMPL(tile_shape &dst, tile_shape &src) {
        unsigned row = dst.GetValidRow();
        unsigned col = dst.GetValidCol();
        TExp_Impl<tile_shape>(dst.data(), src.data(), row, col);
    }
}

#endif