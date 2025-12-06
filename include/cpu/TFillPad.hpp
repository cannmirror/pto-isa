#ifndef TFILLPAD_HPP
#define TFILLPAD_HPP
#include "common/pto_tile.hpp"
#include "cpu/tile_offsets.hpp"

using namespace std;

namespace pto{

    template <typename TileDataDst, typename TileDataSrc>
    void TFillPad(typename TileDataDst::TileDType dst,
                                typename TileDataSrc::TileDType src,
                                unsigned validDstRow, 
                                unsigned validDstCol, 
                                unsigned validSrcRow, 
                                unsigned validSrcCol) {
        typename TileDataDst::DType padVal = 0;

        constexpr auto PadVal_ = TileDataDst::PadVal;
        if constexpr (std::numeric_limits<typename TileDataDst::DType>::has_infinity)
        {
            if constexpr(PadVal_ == PadValue::Max)
                padVal = std::numeric_limits<typename TileDataDst::DType>::infinity();
            else if constexpr (PadVal_ == PadValue::Min)
                padVal = -std::numeric_limits<typename TileDataDst::DType>::infinity();
        }
        else 
        {
            if constexpr (PadVal_ == PadValue::Max)
                padVal = std::numeric_limits<typename TileDataDst::DType>::max();
            else if constexpr (PadVal_ == PadValue::Min)
                padVal = std::numeric_limits<typename TileDataDst::DType>::min();
        }

        for (unsigned int i = 0; i < TileDataDst::Rows; ++i){
            for (unsigned int j = 0; j < TileDataDst::Cols; ++j){
                unsigned int dstIndex;
                if(i < validSrcRow && j < validSrcCol){
                    dst[GetTileElementOffset<TileDataDst>(i,j)] = src[GetTileElementOffset<TileDataSrc>(i,j)];
                } else {
                    dst[GetTileElementOffset<TileDataDst>(i,j)] = padVal;
                }
            }
        }
    }

    template <typename TileDataDst, typename TileDataSrc, bool inplace>
    __aicore__ PTO_INLINE void TFILLPAD_IMPL(TileDataDst &dst, TileDataSrc &src) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);
        constexpr unsigned srcStride = TileDataSrc::RowStride;
        constexpr unsigned dstStride = TileDataDst::RowStride;
        unsigned validSrcRow = src.GetValidRow();
        unsigned validSrcCol = src.GetValidCol();
        unsigned validDstRow = dst.GetValidRow();
        unsigned validDstCol = dst.GetValidCol();

        using T = typename TileDataSrc::DType;
        using U = typename TileDataDst::DType;
        static_assert(TileDataDst::PadVal != PadValue::Null, "TFillPad, dst vecTile pad value can't be Null!");
        static_assert(sizeof(T) == sizeof(U), "TFillPad, src and dst data type shouuld be the same!");
        static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TFillPad: Invalid data type!");
        
        if(validDstRow == 0 || validDstCol == 0) {
            return;
        }
        if constexpr (!inplace) 
        {
            TFillPad<TileDataDst, TileDataSrc>(dst.data(), src.data(), validDstRow, validDstCol, validSrcRow, validSrcCol);
        }
        TFillPad<TileDataDst, TileDataSrc>(dst.data(), src.data(), validDstRow, validDstCol, validSrcRow, validSrcCol);
    }

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TFILLPAD(TileDataDst &dst, TileDataSrc &src) {
        static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows, 
        "TFillPad: dst and src should have the same rows/cols!");

        TFILLPAD_IMPL<TileDataDst, TileDataSrc, false>(dst, src);
    }

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TFILLPAD_INPLACE(TileDataDst &dst, TileDataSrc &src) {
        static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows, 
        "TFillPad: dst and src should have the same rows/cols!");

        TFILLPAD_IMPL<TileDataDst, TileDataSrc, true>(dst, src);
    }

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TFILLPAD_EXPAND(TileDataDst &dst, TileDataSrc &src) {
        static_assert(TileDataDst::Cols >= TileDataSrc::Cols && TileDataDst::Rows >= TileDataSrc::Rows, 
        "TFillPad: dst and src should have the same rows/cols!");

        TFILLPAD_IMPL<TileDataDst, TileDataSrc, false>(dst, src);
    }
}
#endif