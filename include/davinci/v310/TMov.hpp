#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"

namespace pto 
{
    template <typename DstTileData, typename SrcTileData>
    __tf__ __aicore__ void TMovToBt(typename DstTileData::TileData &dst, typename SrcTileData::TileData &src) {
        using SrcType = typename SrcTileData::TileData;
        using DstType = typename DstTileData::TileData;
        static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
            (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float>) ||
            (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) ||
            (std::is_same_v<SrcType, bfloat16_t> && std::is_same_v<DstType, float>), "Incorrect data type.");

        constexpr int32_t srcRow = SrcTileData::Rows;
        constexpr int32_t srcCol = SrcTileData::Cols;
        constexpr int32_t dstCol = DstTileData::Cols;

        static_assert(srcRow == 1, "TMov: When Location is Bias, row must be 1.");
        static_assert(dstCol * sizeof(DstType) % 64 == 0, 
                      "TMov: When Location is Bias, col * sizeof(Dtype) must be aligned to 64.");
        static_assert(dstCol * sizeof(DstType) <= 4096, 
                      "TMov: The memory occupation of BiasTile exceeds 4.0KB boas table size.");

        constexpr const int BURST_LNE_UNIT = 32;
        __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(src);
        uint64_t dstAddrP = (uint64_t)dst;

        uint16_t convControl = 0;
        uint16_t burstNum = 1;
        uint16_t burstLen = srcRow * srcCol *sizeof(SrcType) / BURST_LNE_UNIT;
        uint16_t srcGap = 0;
        uint16_t dstGap = 0;

        if constexpr (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) {
            convControl = 1;
        }

        copy_cbuf_to_bt(dstAddrP, srcAddrP, convControl, burstNum, burstLen, srcGap, dstGap);
    }

    template <typename DstTileData, typename SrcTileData>
    __tf__ __aicore__ void TMovToFb(typename DstTileData::TileData &dst, typename SrcTileData::TileData &src) {
        using SrcType = typename SrcTileData::TileData;
        using DstType = typename DstTileData::TileData;
        constexpr int32_t srcRow = SrcTileData::Rows;
        constexpr int32_t srcCol = SrcTileData::Cols;
        constexpr int32_t dstCol = DstTileData::Cols;

        static_assert(srcRow == 1, "TMov: When Location is Scaling, row must be 1.");
        static_assert(dstCol * sizeof(DstType) % 128 == 0, 
                      "TMov: When Location is Scaling, col * sizeof(Dtype) must be aligned to 128.");
        static_assert(dstCol * sizeof(DstType) <= 4096, 
                      "TMov: The memory occupation of FbTile exceeds 4.0KB boas table size.");

        __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(src);
        __fbuf__ DstType *dstAddrP = (__fbuf__ DstType *)(dst);

        uint16_t burstNum = 1;
        constexpr const int BURST_LNE_UNIT = 64;
        uint16_t burstLen = srcRow * srcCol *sizeof(SrcType) / BURST_LNE_UNIT;
        uint16_t srcGap = 0;
        uint16_t dstGap = 0;

        copy_cbuf_to_fbuf(dstAddrP, srcAddrP, burstNum, burstLen, srcGap, dstGap);

    }

    template <typename DstTileData, typename SrcTileData>
    __aicore__ PTO_INLINE constexpr void CommonCheck()
    {
        static_assert(is_textract_supported_type<typename DstTileData::DType>,
            "Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
            half, bfloat16_t, float");
        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TMov: Destination and Source tile data types must be the same");

        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor), 
                      "TMov: SrcTile Invalid Fractal");
    }

    template <typename DstTileData, typename SrcTileData>
    __aicore__ PTO_INLINE void TMOV_IMPL(DstTileData &dst, SrcTileData &src) {
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)), 
            "TMov: The shape of destination and source tile must be the same.");
        if constexpr (DstTileData::Loc == Location::Bias) {
            TMovToBt<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == Location::Scaling) {
            TMovToFb<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == Location::Left) {
            CommonCheck<DstTileData, SrcTileData>();
            static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        } else if constexpr (DstTileData::Loc == Location::Right) {
            CommonCheck<DstTileData, SrcTileData>();
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        
    }
}
#endif