#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
__tf__ __aicore__ void TMovToBt(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
                  (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float>) ||
                  (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) ||
                  (std::is_same_v<SrcType, bfloat16_t> && std::is_same_v<DstType, float>),
        "Incorrect data type.");

    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstCol = DstTileData::Cols;

    static_assert(srcRow == 1, "TMov: When Location is Bias, row must be 1.");
    static_assert(dstCol * sizeof(DstType) % 64 == 0,
                    "TMov: When Location is Bias, col * sizeof(Dtype) must be aligned to 64.");
    static_assert(
        dstCol * sizeof(DstType) <= 4096, "TMov: The memory occupation of BiasTile exceeds 4.0KB boas table size.");

    constexpr const int BURST_LEN_UNIT = 32;
    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(src);
    uint64_t dstAddrP = (uint64_t)dst;

    bool convControl = false;
    uint16_t burstNum = 1;
    uint16_t burstLen = srcRow * srcCol *sizeof(SrcType) / BURST_LEN_UNIT;
    uint16_t srcGap = 0;
    uint16_t dstGap = 0;

    if constexpr (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) {
        convControl = true;
    }
    copy_cbuf_to_bt(dstAddrP, srcAddrP, convControl, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTileData, typename SrcTileData>
__tf__ __aicore__ void TMovToFb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src) {
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
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
    constexpr int BURST_LEN_UNIT = 64;
    uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;
    uint16_t srcGap = 0;
    uint16_t dstGap = 0;

    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode>
__aicore__ PTO_INLINE constexpr uint8_t GetDualDstCtl()
{
    if constexpr (mode == L0cToUBMode::DualModeSplitM) {
        return 1;
    } else if constexpr (mode == L0cToUBMode::DualModeSplitN) {
        return 2;
    }
    return 0;
}

__aicore__ PTO_INLINE void SetLoop3Para()
{
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode, QuantMode_t QuantPre>
__tf__ __aicore__ void TMovL0cToUB(typename DstTileData::TileDType __out__ dst,
    typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr bool subBlockId = (mode == L0cToUBMode::SingleModeUB1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTileData, SrcTileData, mode>();
    constexpr bool NZ2NDEn = (DstTileData::isRowMajor & DstTileData::SFractal == SLayout::NoneBox);

    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);

    constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                (DstTileData::SFractalSize == 1024)
                                ? 2 * C0_SIZE_BYTE / sizeof(dstType)
                                : C0_SIZE_BYTE / sizeof(dstType);
    
    constexpr uint32_t dstStride = NZ2NDEn ? DstTileData::Cols : (DstTileData::Rows * c0Size);

    if constexpr (NZ2NDEn) {
        SetLoop3Para();
    }

    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)(src);
    copy_matrix_cc_to_ub(dstAddr,
        srcData,
        0,
        validCol,
        validRow,
        dstStride,
        SrcTileData::Rows,
        dualDstCtl,
        subBlockId,
        0,
        0,
        QuantPre,
        0,
        channelSplitEnable,
        NZ2NDEn,
        0,
        0,
        false,
        false,
        0,
        false,
        false,
        false,
        false,
        false,
        false);
}

template <typename DstTileData, typename SrcTileData>
__aicore__ PTO_INLINE constexpr void CommonCheck()
{
    static_assert(is_textract_supported_type<typename DstTileData::DType>,
        "Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
        half, bfloat16_t, float");
    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TMov: Destination and Source tile data types must be the same.");

    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                    (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                    "TMov: SrcTile Invalid Fractal.");
}

template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType, bool isQuant>
__aicore__ PTO_INLINE void CheckTMovL0cToUBValid()
{
    static_assert((SrcTileData::Loc == Location::Acc), "Source location only support Acc.");
    static_assert((DstTileData::Loc == Location::Vec), "Destination location only support Vec.");
    static_assert((!SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor),
        "Src fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    static_assert(((std::is_same<SrcType, float>::value) || (std::is_same<SrcType, int32_t>::value)),
        "Src data type only support float or int32_t.");
    if constexpr (isQuant) {
        if constexpr (std::is_same<SrcType, float>::value) {
            static_assert((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value) ||
                              (std::is_same<DstType, hifloat8_t>::value) || (std::is_same<DstType, half>::value) ||
                              (std::is_same<DstType, bfloat16_t>::value),
                "The output data type must be restricted to int8_t/uint8_t/hifloat/bfloat8_t/half/bfloat16_t.");
        } else if constexpr (std::is_same<SrcType, int32_t>::value) {
            static_assert((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value) ||
                              (std::is_same<DstType, half>::value) || (std::is_same<DstType, bfloat16_t>::value),
                "The output data type must be restricted to int8_t/uint8_t/half/bfloat16_t.");
        }
    } else {
        if constexpr (std::is_same<SrcType, float>::value) {
            static_assert((std::is_same<DstType, half>::value) || (std::is_same<DstType, bfloat16_t>::value) ||
                              (std::is_same<DstType, float>::value),
                "The output data type must be restricted to half/bfloat16_t/float.");
        } else if constexpr (std::is_same<SrcType, int32_t>::value) {
            static_assert((std::is_same<DstType, int32_t>::value),
                "The output data type must be restricted to int32_t.");
        }
    }
    static_assert(((DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                    (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor)),
        "Only nz2nz and nz2nd are supported Currently.");
    if constexpr (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) {
        constexpr uint16_t dstStride = DstTileData::Cols;
        static_assert(((dstStride * sizeof(DstType) % C0_SIZE_BYTE == 0) && (dstStride > 0)),
            "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd.");
    }
    if constexpr (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) {
        constexpr bool channelSplitEnable = (std::is_same_v<typename DstTileData::DType, float>) &&
                                            (DstTileData::SFractalSize == 512);
        constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                (DstTileData::SFractalSize == 1024)
                                ? 2 * C0_SIZE_BYTE / sizeof(DstType)
                                : C0_SIZE_BYTE / sizeof(DstType);
        constexpr uint32_t dstStride = DstTileData::Rows * c0Size;
        static_assert(((dstStride * sizeof(DstType) % C0_SIZE_BYTE == 0) && (dstStride > 0)),
                    "Dst Tile Cols * sizeof(DstType) must be multiples of 32 and not 0 when nz2nz.");
    }
}

template <typename DstTileData, typename SrcTileData>
__aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::Loc == Location::Mat) {
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
            "TMov: The shape of destination and source tile must be the same.");
        if constexpr (DstTileData::Loc == Location::Bias) {
            TMovToBt<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == Location::Scaling) {
            TMovToFb<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == Location::Left) {
            CommonCheck<DstTileData, SrcTileData>();
            static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal.");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        } else if constexpr (DstTileData::Loc == Location::Right) {
            CommonCheck<DstTileData, SrcTileData>();
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal.");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        }
    } else if constexpr (SrcTileData::Loc == Location::Acc) {
        if constexpr (DstTileData::Loc == Location::Vec) {
            CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
            constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
            uint16_t m = src.GetValidRow();
            uint16_t n = src.GetValidCol();
            TMovL0cToUB<DstTileData, SrcTileData, L0cToUBMode::SingleModeUB0, quantPre>(dst.data(), src.data(), m, n);
        }
    }
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode>
__aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(((mode == L0cToUBMode::SingleModeUB0) || (mode == L0cToUBMode::SingleModeUB1)) ||
                    (std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value),
        "Quant is not support in dual Dst Mode.");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    TMovL0cToUB<DstTileData, SrcTileData, mode, quantPre>(dst.data(), src.data(), m, n);
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
__aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    static_assert((mode == L0cToUBMode::SingleModeUB0) || (mode == L0cToUBMode::SingleModeUB1),
        "Quant is not support in dual Dst Mode.");
    CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    TMovL0cToUB<DstTileData, SrcTileData, mode, quantPre>(dst.data(), src.data(), m, n);
}

template <typename FpTileData>
__tf__ __aicore__ PTO_INLINE void SetFPC(typename FpTileData::TileDType __in__ fp)
{
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData,
    L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
__aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    static_assert((mode == L0cToUBMode::SingleModeUB0) || (mode == L0cToUBMode::SingleModeUB1),
        "Quant is not support in dual Dst Mode.");
    CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert(FpTileData::Loc == Location::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTileData>(fp.data());
    TMovL0cToUB<DstTileData, SrcTileData, mode, quantPre>(dst.data(), src.data(), m, n);
}
}  // namespace pto
#endif