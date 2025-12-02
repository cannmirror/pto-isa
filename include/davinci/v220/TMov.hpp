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
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr const int BURST_LEN_UNIT = 64;
    __cbuf__ SrcType* srcAddrP = (__cbuf__ SrcType*)(src);
    uint64_t dstAddrP = (uint64_t)dst;

    uint16_t convControl = 0;
    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;

    if constexpr (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) {
        convControl = 1;
    }
    copy_cbuf_to_bt(dstAddrP, srcAddrP, convControl, (uint16_t)1, burstLen, (uint16_t)0, (uint16_t)0);
}

template <typename DstTileData, typename SrcTileData>
__tf__ __aicore__ void TMovToFb(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr const int BURST_LEN_UNIT = 128;
    constexpr const int RELU_BIT = 16;
    __cbuf__ SrcType* srcAddrP = (__cbuf__ SrcType*)(src);
    __fbuf__ DstType* dstAddr = (__fbuf__ DstType*)(dst);
    constexpr bool isRelu = 0;
    __fbuf__ DstType* dstAddrP = (__fbuf__ DstType*)(dstAddr || (isRelu << RELU_BIT));

    constexpr uint16_t burstLen = srcRow * srcCol * sizeof(SrcType) / BURST_LEN_UNIT;
    copy_cbuf_to_fbuf(dstAddrP, srcAddrP, (uint16_t)1, burstLen, (uint16_t)0, (uint16_t)0);
}

template <typename DstTileData, typename SrcTileData>
__aicore__ PTO_INLINE void TMovCheckValid()
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    static_assert((SrcTileData::Rows == DstTileData::Rows) && ((SrcTileData::Cols == DstTileData::Cols)),
                  "TMov: The shape of src needs to be the same as that of dst.");
    static_assert((SrcTileData::Loc == Location::Mat &&
                      (DstTileData::Loc == Location::Left || DstTileData::Loc == Location::Right ||
                       DstTileData::Loc == Location::Bias || DstTileData::Loc == Location::Scaling))||
                       (DstTileData::Loc == Location::Vec && SrcTileData::Loc == Location::Vec),
                  "TMov: Invalid Location.");
    if constexpr (DstTileData::Loc == Location::Left) {
        static_assert(std::is_same<DstType, SrcType>::value,
                      "TMov: Destination and Source tile data types must be the same.");
        static_assert(std::is_same<DstType, int8_t>::value || std::is_same<DstType, half>::value ||
                          std::is_same<DstType, bfloat16_t>::value || std::is_same<DstType, float>::value,
                      "TMov: Invalid data type.");
        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                          (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                      "TMov: SrcTile Invalid Fractal.");
        static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                      "TMov: LeftTile Invalid Fractal.");
    } else if constexpr (DstTileData::Loc == Location::Right) {
        static_assert(std::is_same<DstType, SrcType>::value,
                      "TMov: Destination and Source tile data types must be the same.");
        static_assert(std::is_same<DstType, int8_t>::value || std::is_same<DstType, half>::value ||
                          std::is_same<DstType, bfloat16_t>::value || std::is_same<DstType, float>::value,
                      "TMov: Invalid data type.");
        static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                          (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                      "TMov: SrcTile Invalid Fractal.");
        static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                      "TMov: RightTile Invalid Fractal.");
    } else if constexpr (DstTileData::Loc == Location::Bias) {
        // check dataType
        if constexpr (std::is_same<SrcType, int32_t>::value || std::is_same<SrcType, float>::value) {
            static_assert(std::is_same<DstType, SrcType>::value,
                          "TMov: Destination and Source tile data types must be the same.");
        } else if constexpr (std::is_same<SrcType, half>::value) {
            static_assert(std::is_same<DstType, float>::value,
                          "TMov: When Source tile data types is half, dst tile data types must be float");
        }
        // check shape
        static_assert(SrcTileData::Rows == 1, "TMov: When Location is Bias, row must be 1");
        // check alignment
        static_assert(SrcTileData::Cols * sizeof(SrcType) % 64 == 0,
                      "TMov: When Location is Bias, col * sizeof(srcDType) must be aligned to 64");
    } else if constexpr (DstTileData::Loc == Location::Scaling) {
        // check dataType
        static_assert(std::is_same<DstType, SrcType>::value,
                      "TMov: Destination and Source tile data types must be the same.");
        static_assert(std::is_same<DstType, uint64_t>::value, "TMov: Invalid data type.");
        // check shape
        static_assert(SrcTileData::Rows == 1, "TMov: When Location is Scaling, row must be 1");
        // check alignment
        static_assert(SrcTileData::Cols * sizeof(SrcType) % 128 == 0,
                      "TMov: When Location is Scaling, col * sizeof(srcType) must be aligned to 128");
    }
}

template <typename DstTileData, typename SrcTileData>
__aicore__ void TMovToVec(DstTileData &dst, SrcTileData &src) {
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename SrcTileData::DType);
    constexpr unsigned srcStride = SrcTileData::RowStride;
    constexpr unsigned dstStride = DstTileData::RowStride;
    uint64_t validSrcRow = src.GetValidRow();
    uint64_t validSrcCol = src.GetValidCol();
    uint64_t validDstRow = dst.GetValidRow();
    uint64_t validDstCol = dst.GetValidCol();
    uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
    uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;
    TCopy<DstTileData, SrcTileData, blockSizeElem, srcStride, dstStride>(dst.data(), src.data(), validRow, validCol);
}

template <typename DstTileData, typename SrcTileData>
__aicore__ void TMOV_IMPL(DstTileData& dst, SrcTileData& src)
{
    TMovCheckValid<DstTileData, SrcTileData>();
    if constexpr (SrcTileData::Loc == Location::Mat && DstTileData::Loc == Location::Left) {
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    } else if constexpr (SrcTileData::Loc == Location::Mat && DstTileData::Loc == Location::Right) {
        if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    } else if constexpr (SrcTileData::Loc == Location::Mat && DstTileData::Loc == Location::Bias) {
        TMovToBt<DstTileData, SrcTileData>(dst.data(), src.data());
    } else if constexpr (SrcTileData::Loc == Location::Mat && DstTileData::Loc == Location::Scaling) {
        TMovToFb<DstTileData, SrcTileData>(dst.data(), src.data());
    } else if constexpr (SrcTileData::Loc == Location::Vec && DstTileData::Loc == Location::Vec) {
        TMovToVec<DstTileData, SrcTileData>(dst, src);
    }
}
}  // namespace pto
#endif  //TMOV_HPP
