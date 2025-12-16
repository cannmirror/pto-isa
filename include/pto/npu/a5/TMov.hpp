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
#include "TPartAdd.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TMovToBt(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
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

    static_assert(srcRow == ONE_ROW, "TMov: When TileType is Bias, row must be 1.");
    static_assert(dstCol * sizeof(DstType) % BIAS_TABLE_UNIT == 0,
        "TMov: When TileType is Bias, col * sizeof(Dtype) must be aligned to 64.");
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
__tf__ AICORE void TMovToFb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr const int FIXPIPE_BUFFER_UNIT = 128;
    constexpr const int FIXPIPE_BUFFER_SIZE = 4096;
    constexpr const int ONE_ROW = 1;

    static_assert(srcRow == ONE_ROW, "TMov: When TileType is Scaling, row must be 1.");
    static_assert(dstCol * sizeof(DstType) % FIXPIPE_BUFFER_UNIT == 0,
        "TMov: When TileType is Scaling, col * sizeof(Dtype) must be aligned to 128.");
    static_assert(dstCol * sizeof(DstType) <= FIXPIPE_BUFFER_SIZE,
        "TMov: The memory occupation of FbTile exceeds 4.0KB fixpipe buffer size.");

    __cbuf__ SrcType *srcAddrP = (__cbuf__ SrcType *)(src);
    __fbuf__ DstType *dstAddrP = (__fbuf__ DstType *)(dst);

    constexpr uint16_t burstNum = 1;
    constexpr int BURST_LEN_UNIT_SHIFT = 6; // BURST_LEN_UNIT = 64;
    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) >> BURST_LEN_UNIT_SHIFT;
    constexpr uint16_t srcGap = 0;
    constexpr uint16_t dstGap = 0;

    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, QuantMode_t quantPre>
PTO_INTERNAL constexpr uint8_t GetDualDstCtl()
{
    if constexpr (mode == AccToVecMode::DualModeSplitM || mode == AccToVecMode::DualModeSplitN) {
        static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.");
        static_assert((!(!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox)),
            "Dual Dst Mode is not support in nz2dn.");
        return ((mode == AccToVecMode::DualModeSplitM) ? 1 : 2);
    }
    return 0;
}

PTO_INTERNAL void SetLoop3Para()
{
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL constexpr uint32_t GetTmovAccDstStride()
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

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToCb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr bool enableNz2Nd = (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTileData, SrcTileData>();

    if constexpr (enableNz2Nd || enableNz2Dn) {
        SetLoop3Para();
    }
    if constexpr (enableNz2Dn) {
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    }

    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)(src);

    copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, validCol, validRow, dstStride, SrcTileData::Rows, 0, 0, 0, QuantPre,
        reluMode, channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0, false, false, false, false, false, enableNz2Dn);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, QuantMode_t quantPre, ReluPreMode reluMode>
__tf__ AICORE void TMovCcToUb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
    uint16_t validRow, uint16_t validCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr bool subBlockId = (mode == AccToVecMode::SingleModeVec1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTileData, SrcTileData, mode, quantPre>();
    constexpr bool enableNz2Nd = (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTileData, SrcTileData>();

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
        0, 0, quantPre, reluMode, channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0, false, false, false, false,
        false, enableNz2Dn);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL constexpr void CommonCheck()
{
    static_assert(is_textract_supported_type<typename DstTileData::DType>,
        "TMov: Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
            half, bfloat16_t, float, float4_e2m1x2_t, float4_e1m2x2_t");
    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TMov: Destination and Source tile data types must be the same.");

    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
        "TMov: SrcTile Invalid Fractal.");
}

template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType, bool isQuant = false>
PTO_INTERNAL void CheckTMovAccValid()
{
    static_assert((SrcTileData::Loc == TileType::Acc), "Source TileType only support Acc.");
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
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTileData, SrcTileData>();
    static_assert(((dstStride * sizeof(DstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
        "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(DstType) must be multiples of 32 and not 0 when nz2nz.");
}

template <typename T, typename DstTileData, typename SrcTileData>
AICORE void TMovToVecNd2Nz(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, uint32_t validRow, uint32_t validCol)
{
    static_assert((std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) ||
        (std::is_same<T, float>::value), "Dst and src must be float, half or bfloat16_t.");
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t srcByteSize = srcRow * srcCol * sizeof(T);
    constexpr int32_t dstByteSize = DstTileData::Rows * DstTileData::Cols * sizeof(T);
    static_assert((srcByteSize % CUBE_BLOCK_SIZE == 0) && (dstByteSize >= srcByteSize),
        "SrcTile bytes size must be 512B align and dstTile greater than or equal to srcTile.");

    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr bool isOptForConflict = (dstByteSize >= (srcRow + 1) * srcCol * sizeof(T)) ? true : false;
    uint32_t blockStride = isOptForConflict ? ((validRow + 1) * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE :
        (validRow * C0_SIZE_BYTE) / BLOCK_BYTE_SIZE;
    uint32_t repeatStride = 1;
    __VEC_SCOPE__
    {
        RegTensor<T> vreg;
        MaskReg preg;
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                uint32_t count = ((j + 1) * elementsPerRepeat >= validCol ?
                    (validCol - j * elementsPerRepeat) : elementsPerRepeat);
                preg = CreatePredicate<T>(count);
                vlds(vreg, srcPtr, i * SrcTileData::RowStride + j * count, NORM);
                vsstb(vreg, dstPtr, (blockStride << 16u) | (repeatStride &0xFFFFU), preg, POST_UPDATE);
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TMovToVec(DstTileData &dst, SrcTileData &src) {
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename SrcTileData::DType);
    constexpr unsigned dstStride = DstTileData::RowStride;
    constexpr unsigned srcStride = SrcTileData::RowStride;
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    if constexpr ((SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) &&
        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor))) {
        TMovToVecNd2Nz<typename DstTileData::DType, DstTileData, SrcTileData>(
            (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst.data()),
            (__ubuf__ typename SrcTileData::DType *)__cce_get_tile_ptr(src.data()), validRow, validCol);
    } else {
        TPartCopyInstr<typename DstTileData::DType, DstTileData, SrcTileData, blockSizeElem, dstStride, srcStride>(
            (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst.data()),
            (__ubuf__ typename SrcTileData::DType *)__cce_get_tile_ptr(src.data()),
            validRow, validCol, 0);
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (SrcTileData::Loc == TileType::Mat) {
        static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
            "TMov: The shape of destination and source tile must be the same.");
        if constexpr (DstTileData::Loc == TileType::Bias) {
            TMovToBt<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == TileType::Scaling) {
            TMovToFb<DstTileData, SrcTileData>(dst.data(), src.data());
        } else if constexpr (DstTileData::Loc == TileType::Left) {
            CommonCheck<DstTileData, SrcTileData>();
            static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal.");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        } else if constexpr (DstTileData::Loc == TileType::Right) {
            CommonCheck<DstTileData, SrcTileData>();
            static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                "TMov: DstTile Invalid Fractal.");
            if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
                TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
            } else {
                TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
            }
        }
    } else if constexpr (SrcTileData::Loc == TileType::Acc) {
        CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        if constexpr (DstTileData::Loc == TileType::Vec) {
            TMovCcToUb<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), m, n);
        } else if constexpr (DstTileData::Loc == TileType::Mat) {
            TMovCcToCb<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), m, n);
        }
    } else if constexpr (SrcTileData::Loc == TileType::Vec) {
        if constexpr (DstTileData::Loc == TileType::Vec) {
            TMovToVec<DstTileData, SrcTileData>(dst, src);
        } else if constexpr(DstTileData::Loc == TileType::Mat) {
            CommonCheck<DstTileData, SrcTileData>();
            TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, src.GetValidRow(),
                src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    if constexpr (DstTileData::Loc == TileType::Vec) {
        TMovCcToUb<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), m, n);
    } else if constexpr (DstTileData::Loc == TileType::Mat) {
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    static_assert((DstTileData::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TMovCcToUb<DstTileData, SrcTileData, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr (DstTileData::Loc == TileType::Vec) {
        TMovCcToUb<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), m, n);
    } else if constexpr (DstTileData::Loc == TileType::Mat) {
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
        "Quant is not support in dual Dst Mode.");
    static_assert((DstTileData::Loc == TileType::Vec), "Destination location only support Vec.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TMovCcToUb<DstTileData, SrcTileData, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}

template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPC(typename FpTileData::TileDType __in__ fp)
{
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data());
    if constexpr (DstTileData::Loc == TileType::Vec) {
        TMovCcToUb<DstTileData, SrcTileData, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), m, n);
    } else if constexpr (DstTileData::Loc == TileType::Mat) {
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), m, n);
    }
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu>
AICORE void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((mode == AccToVecMode::SingleModeVec0) || (mode == AccToVecMode::SingleModeVec1),
        "Quant is not support in dual Dst Mode.");
    static_assert((DstTileData::Loc == TileType::Vec), "Destination location only support Vec.");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    uint16_t m = src.GetValidRow();
    uint16_t n = src.GetValidCol();
    SetFPC<FpTileData>(fp.data());
    TMovCcToUb<DstTileData, SrcTileData, mode, quantPre, reluMode>(dst.data(), src.data(), m, n);
}
} // namespace pto
#endif