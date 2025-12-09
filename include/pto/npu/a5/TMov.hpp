/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_HPP
#define TMOV_HPP
#include "TExtract.hpp"
#include "TCopy.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
__tf__ __aicore__ void TMovToBt(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using DstType = typename DstTileData::DType;
    using SrcType = typename SrcTileData::DType;
    static_assert((std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>) ||
                      (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float>) ||
                      (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) ||
                      (std::is_same_v<SrcType, bfloat16_t> && std::is_same_v<DstType, float>),
        "Incorrect data type.");

    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr const int BIAS_TABLE_UNIT = 64;
    constexpr const int BIAS_TABLE_SIZE = 4096;
    constexpr const int ONE_ROW = 1;

    static_assert(srcRow == ONE_ROW, "TMov: When Location is Bias, row must be 1.");
    static_assert(dstCol * sizeof(DstType) % BIAS_TABLE_UNIT == 0,
        "TMov: When Location is Bias, col * sizeof(Dtype) must be aligned to 64.");
    static_assert(dstCol * sizeof(DstType) <= BIAS_TABLE_SIZE,
        "TMov: The memory occupation of BiasTile exceeds 4.0KB bias table size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(src);
    uint64_t dstAddrP = (uint64_t)dst;

    bool convControl = false;
    constexpr uint16_t burstNum = 1;
    constexpr const int BURST_LEN_UNIT_SHIFT = 5; // BURST_LEN_UNIT = 32;
    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) >> BURST_LEN_UNIT_SHIFT;
    constexpr uint16_t srcGap = 0;
    constexpr uint16_t dstGap = 0;

    if constexpr (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) {
        convControl = true;
    }
    copy_cbuf_to_bt(dstAddrP, srcAddrP, convControl, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTileData, typename SrcTileData>
__tf__ __aicore__ void TMovToFb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr const int FIXPIPE_BUFFER_UNIT = 128;
    constexpr const int FIXPIPE_BUFFER_SIZE = 4096;
    constexpr const int ONE_ROW = 1;

    static_assert(srcRow == ONE_ROW, "TMov: When Location is Scaling, row must be 1.");
    static_assert(dstCol * sizeof(DstType) % FIXPIPE_BUFFER_UNIT == 0,
        "TMov: When Location is Scaling, col * sizeof(Dtype) must be aligned to 128.");
    static_assert(dstCol * sizeof(DstType) <= FIXPIPE_BUFFER_SIZE,
        "TMov: The memory occupation of FbTile exceeds 4.0KB fixpipe buffer size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(src);
    __fbuf__ DstType *dstAddrP = (__fbuf__ DstType *)(dst);

    constexpr uint16_t burstNum = 1;
    constexpr int BURST_LEN_UNIT_SHIFT = 6;  //BURST_LEN_UNIT = 64;
    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) >> BURST_LEN_UNIT_SHIFT;
    constexpr uint16_t srcGap = 0;
    constexpr uint16_t dstGap = 0;

    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode, QuantMode_t quantPre>
__aicore__ PTO_INLINE constexpr uint8_t GetDualDstCtl()
{
    if constexpr (mode == L0cToUBMode::DualModeSplitM || mode == L0cToUBMode::DualModeSplitN) {
        static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.");
        static_assert((!(!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox)),
            "Dual Dst Mode is not support in nz2dn.");
        return ((mode == L0cToUBMode::DualModeSplitM) ? 1 : 2);
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

template <typename DstTileData, typename SrcTileData>
__aicore__ PTO_INLINE constexpr uint32_t GetTmovL0cToUBDstStride()
{
    if constexpr (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) {
        return DstTileData::Cols;
    } else if constexpr (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) {
        return DstTileData::Rows;
    }
    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);
    constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (DstTileData::SFractalSize == 1024) ?
                                    2 * C0_SIZE_BYTE / sizeof(typename DstTileData::DType) :
                                    C0_SIZE_BYTE / sizeof(typename DstTileData::DType);
    return DstTileData::Rows * c0Size;
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode, QuantMode_t quantPre>
__tf__ __aicore__ void TMovL0cToUB(typename DstTileData::TileDType __out__ dst,
    typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr bool subBlockId = (mode == L0cToUBMode::SingleModeUB1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTileData, SrcTileData, mode, quantPre>();
    constexpr bool enableNz2Nd = (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovL0cToUBDstStride<DstTileData, SrcTileData>();

    if constexpr (enableNz2Nd) {
        SetLoop3Para();
    } else if constexpr (enableNz2Dn) {
        SetLoop3Para();
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    }

    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)(src);
    copy_matrix_cc_to_ub(dstAddr, srcData, 0, validCol, validRow, dstStride, SrcTileData::Rows, dualDstCtl, subBlockId,
        0, 0, quantPre, 0, channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0, false, false, false, false, false,
        enableNz2Dn);
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
            static_assert(
                (std::is_same<DstType, int32_t>::value), "The output data type must be restricted to int32_t.");
        }
    }
    static_assert(((DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                      (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) ||
                      (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor)),
        "Only support nz2nz, nz2nd or nz2dn.");
    constexpr uint32_t dstStride = GetTmovL0cToUBDstStride<DstTileData, SrcTileData>();
    static_assert(((dstStride * sizeof(DstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
        "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(DstType) must be multiples of 32 and not 0 when nz2nz.");
}

template <typename DstTileData, typename SrcTileData>
__aicore__ void TMovToVec(DstTileData &dst, SrcTileData &src) {
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename SrcTileData::DType);
    constexpr unsigned dstStride = DstTileData::RowStride;
    constexpr unsigned srcStride = SrcTileData::RowStride;
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TCopy<DstTileData, SrcTileData, blockSizeElem, srcStride, dstStride>(dst.data(), src.data(), validRow, validCol);
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
            CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType,
                false>();
            constexpr QuantMode_t quantPre =
                GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
            uint16_t m = src.GetValidRow();
            uint16_t n = src.GetValidCol();
            TMovL0cToUB<DstTileData, SrcTileData, L0cToUBMode::SingleModeUB0, quantPre>(dst.data(), src.data(), m, n);
        }
    } else if constexpr (SrcTileData::Loc == Location::Vec && DstTileData::Loc == Location::Vec) {
        TMovToVec<DstTileData, SrcTileData>(dst, src);
    }
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode>
__aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    TMovL0cToUB<DstTileData, SrcTileData, mode, quantPre>(dst.data(), src.data(), m, n);
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
__aicore__ void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((mode == L0cToUBMode::SingleModeUB0) || (mode == L0cToUBMode::SingleModeUB1),
        "Quant is not support in dual Dst Mode.");
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
    CheckTMovL0cToUBValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((mode == L0cToUBMode::SingleModeUB0) || (mode == L0cToUBMode::SingleModeUB1),
        "Quant is not support in dual Dst Mode.");
    static_assert(FpTileData::Loc == Location::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTileData>(fp.data());
    TMovL0cToUB<DstTileData, SrcTileData, mode, quantPre>(dst.data(), src.data(), m, n);
}
} // namespace pto
#endif