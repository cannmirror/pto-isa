#ifndef TCI_HPP
#define TCI_HPP

#include "common/constants.hpp"


namespace pto {
    template <typename TileData, typename T>
    __aicore__ PTO_INLINE void CheckValid() {
        static_assert((std::is_same<typename TileData::DType, T>::value),
                      "expect src and dst same datatype");
        static_assert((sizeof(typename TileData::DType) == 4 || (sizeof(typename TileData::DType) == 2)),
                      "expect b32 or b16");
        static_assert((TileData::Cols != 1),
                      "expect row is 1");
    }

    template <typename TileData, typename T, int descending>
    __tf__ __aicore__ void TCI(typename TileData::TileDType __out__ dst, T S, unsigned validCol)
    {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);

        //scalar
        if(descending)
        {
            for(int32_t j = 0; j < validCol; j++) {
                *(dstPtr + j) = S - j;
            }
        }
        else
        {
            for(int32_t j = 0; j < validCol; j++) {
                *(dstPtr + j) = S + j;
            }
        }
    }

    template <typename TileData, typename T, int descending>
    __aicore__ void TCI_IMPL(TileData &dst, T S) {
        CheckValid<TileData, T>();

        unsigned validCol = dst.GetValidCol();

        TCI<TileData, T, descending>(dst.data(), S, validCol);
    }
}
#endif
