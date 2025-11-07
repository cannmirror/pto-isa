#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"

namespace pto 
{
    template <typename DstTileData, typename SrcTileData>
    __aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src) {
         // 增加校验
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)), 
            "The shape of src needs to be the same as that of dst.");
        TExtract<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0);
    }
}
#endif