/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP

namespace pto {
template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src,
    uint16_t nBurst, uint32_t lenBurst, uint32_t gmGap, uint32_t ubGap)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 8) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmNd2nd(typename GlobalData::DType *dstAddr,
    __ubuf__ typename TileData::DType *srcAddr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    uint16_t nBurst = gShape3;
    uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
    uint32_t gmGap = (gStride3 - gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = ((TileData::Cols - validCol) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;

    int64_t srcStride2 = gShape3 * TileData::Cols;
    int64_t srcStride1 = gShape2 * srcStride2;
    int64_t srcStride0 = gShape1 * srcStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * gStride0;
        int64_t srcAddr0 = i * srcStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * gStride1;
            int64_t srcAddr1 = j * srcStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                dstGlobalAddr = dstAddr + dstAddr0 + dstAddr1 + k * gStride2;
                srcTileAddr = srcAddr + srcAddr0 + srcAddr1 + k * srcStride2;
                TStoreUb2gmInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmDn2dn(typename GlobalData::DType *dstAddr,
    __ubuf__ typename TileData::DType *srcAddr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    uint16_t nBurst = gShape4;
    uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
    uint32_t gmGap = (gStride4 - gShape3) * sizeof(typename TileData::DType);
    uint32_t ubGap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    int64_t srcStride2 = TileData::Rows * gShape4;
    int64_t srcStride1 = gShape2 * srcStride2;
    int64_t srcStride0 = gShape1 * srcStride1;

    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * srcStride0;
        int64_t dstAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * srcStride1;
            int64_t dstAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                dstGlobalAddr = dstAddr + dstAddr0 + dstAddr1 + k * gStride2;
                srcTileAddr = srcAddr + srcAddr0 + srcAddr1 + k * srcStride2;
                TStoreUb2gmInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmNz2nz(typename GlobalData::DType *dstAddr,
    __ubuf__ typename TileData::DType *srcAddr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
        "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = TileData::Rows - validRow;

    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;

    int64_t tileStride = TileData::Rows * gShape1 * gShape4;
    for (uint32_t i = 0; i < gShape0; i++) {
        dstGlobalAddr = dstAddr + i * gStride0;
        srcTileAddr = srcAddr + i * tileStride;
        TStoreUb2gmInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
    }
}

template <typename GlobalData, typename TileData, AtomicType atomicType = AtomicType::AtomicNone>
__tf__ AICORE void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename TileData::DType *srcAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;

    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreUb2gmNd2nd<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreUb2gmDn2dn<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        TStoreUb2gmNz2nz<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetCastPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if constexpr (std::is_same<DstType, __gm__ bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::QF322B8_PRE;
        } else if constexpr ((std::is_same<DstType, __gm__ half>::value)) {
            quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr ((std::is_same<DstType, __gm__ bfloat16_t>::value)) {
            quantPre = QuantMode_t::QF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr ((std::is_same<DstType, __gm__ half>::value)) {
            quantPre = QuantMode_t::DEQF16;
        } else if constexpr ((std::is_same<DstType, __gm__ bfloat16_t>::value)) {
            quantPre = QuantMode_t::QS322BF16_PRE;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::VQF322B8_PRE;
        } else if constexpr ((std::is_same<DstType, __gm__ half>::value)) {
            quantPre = QuantMode_t::VQF322F16_PRE;
        } else if constexpr ((std::is_same<DstType, __gm__ bfloat16_t>::value)) {
            quantPre = QuantMode_t::VQF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr ((std::is_same<DstType, __gm__ half>::value)) {
            quantPre = QuantMode_t::VDEQF16;
        } else if constexpr ((std::is_same<DstType, __gm__ bfloat16_t>::value)) {
            quantPre = QuantMode_t::VQS322BF16_PRE;
        }
    }
    return quantPre;
}

template <typename T>
PTO_INTERNAL void SetAtomicAdd()
{
    static_assert((std::is_same<T, __gm__ half>::value) || (std::is_same<T, __gm__ float>::value) ||
                      (std::is_same<T, __gm__ int16_t>::value) || (std::is_same<T, __gm__ int32_t>::value) ||
                      (std::is_same<T, __gm__ int8_t>::value) || (std::is_same<T, __gm__ bfloat16_t>::value),
        "Dst and src must be half / float / int16_t / int32_t / int8_t / bfloat16_t.");
    atomic_type_t atomicType = atomic_type_t::ATOMIC_NONE;
    if constexpr (std::is_same<T, __gm__ float>::value) {
        atomicType = atomic_type_t::ATOMIC_F32;
    } else if constexpr (std::is_same<T, __gm__ half>::value) {
        atomicType = atomic_type_t::ATOMIC_F16;
    } else if constexpr (std::is_same<T, __gm__ int16_t>::value) {
        atomicType = atomic_type_t::ATOMIC_S16;
    } else if constexpr (std::is_same<T, __gm__ int32_t>::value) {
        atomicType = atomic_type_t::ATOMIC_S32;
    } else if constexpr (std::is_same<T, __gm__ int8_t>::value) {
        atomicType = atomic_type_t::ATOMIC_S8;
    } else if constexpr (std::is_same<T, __gm__ bfloat16_t>::value) {
        atomicType = atomic_type_t::ATOMIC_BF16;
    }
    set_st_atomic_cfg(atomicType | (atomic_op_t::ATOMIC_SUM << 3));
}

PTO_INTERNAL void SetAtomicNone()
{
    set_st_atomic_cfg(atomic_type_t::ATOMIC_NONE, atomic_op_t::ATOMIC_SUM);
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreAccND(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    constexpr uint16_t srcStride = TileData::Rows;
    uint32_t dstD = gStride3;

    uint16_t ndNum = validCol / gShape4;
    constexpr uint8_t c0 = 16;
    uint16_t srcNdStride = TileData::Rows * gShape4 * c0;
    uint16_t dstNdStride = gStride2;

    constexpr uint8_t unitFlagCtrl = 0;
    constexpr uint8_t nz2ndEn = 1;

    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    uint64_t xmReg = 0;
    xmReg = ((nSize & 0xfff) << 4) | (static_cast<uint64_t>(mSize & 0xffff) << 16) |
            (static_cast<uint64_t>(dstD & 0xffffffff) << 32);
    uint64_t xtReg = 0;
    xtReg = srcStride | (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |
            (static_cast<uint64_t>(quantPre & 0x1f) << 34) | (static_cast<uint64_t>(nz2ndEn & 0x1) << 43);
    uint64_t ndParaSPR = 0;
    ndParaSPR = ndNum | (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) |
                (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);
    set_nd_para(ndParaSPR);
    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreAccNZ(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
     typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __cc__ typename TileData::DType *srcTileAddr = srcAddr;
    uint16_t mSize = validRow;
    uint16_t nSize = gShape1 * gShape4;

    constexpr uint16_t srcStride = TileData::Rows;
    uint32_t dstStride = gShape2 * gShape3 * 2;

    constexpr uint8_t unitFlagCtrl = 0;
    uint8_t channelSplitEn = 0;
    if (std::is_same_v<typename TileData::DType, float> && std::is_same_v<typename GlobalData::DType, __gm__ float>) {
        if (gShape4 == 8) {
            dstStride >>= 1;
            channelSplitEn = 1;
        }
    }
    if (std::is_same_v<typename GlobalData::DType, __gm__ half> ||
        std::is_same_v<typename GlobalData::DType, __gm__ bfloat16_t>) {
        dstStride >>= 1;
    }
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();

    uint64_t xmReg = 0;
    xmReg = (static_cast<uint64_t>(nSize & 0xfff) << 4) | (static_cast<uint64_t>(mSize) << 16) |
            (static_cast<uint64_t>(dstStride & 0xffffffff) << 32);
    uint64_t xtReg = 0;
    xtReg = srcStride | (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |
            (static_cast<uint64_t>(quantPre & 0x1f) << 34) | (static_cast<uint64_t>(channelSplitEn & 0x1) << 42);

    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    for (uint32_t i = 0; i < gShape0; ++i) {
        dstGlobalAddr = dstAddr + i * gStride0;
        srcTileAddr = srcAddr + i * tileStride;
        copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
    }
}

template <typename GlobalData, typename TileData>
__tf__ AICORE void TStoreAcc(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    __cc__ typename TileData::DType *srcAddr = (__cc__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;

    if constexpr (GlobalData::layout == pto::Layout::ND) {
        TStoreAccND<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NZ) {
        TStoreAccNZ<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckStaticVec()
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/bfloat16_t/float!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Source dtype must be same with dst dtype!");
    static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
        "Src and dst layout must be same, only support ND/DN/NZ!");
    if constexpr (std::is_same_v<typename TileData::DType, int64_t> ||
                  std::is_same_v<typename TileData::DType, uint64_t>) {
        static_assert((GlobalData::layout == pto::Layout::ND &&
        (TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox)) ||
        (GlobalData::layout == pto::Layout::DN && (!TileData::isRowMajor && TileData::SFractal == SLayout::NoneBox)),
            "TSTORE(GlobalTensor, VecTile) only support ND2ND/DN2DN for b64!");
    }
}

template <typename TileData, typename GlobalData, bool isQuant>
PTO_INTERNAL void CheckStaticAcc()
{
    static_assert((GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::NZ),
        "The output data layout must be ND or NZ.");
    static_assert(std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, float>,
        "The input data type must be restricted to int32_t/float!");
    if constexpr (!isQuant) {
        static_assert(std::is_same_v<typename GlobalData::DType, __gm__ int32_t> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ float> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ half> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ bfloat16_t>,
            "The output data type must be restricted to int32_t/float/half/bfloat16_t!");
    }
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert((GlobalData::layout == pto::Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
                      (GlobalData::layout == pto::Layout::NZ && TileData::Rows >= 1 && TileData::Rows <= 65535 &&
                          TileData::Cols % 16 == 0),
        "When GlobalData is ND format, the range of Rows is [1, 8192]."
        "When GlobalData is NZ format, the range of Rows is [1, 65535] and Cols "
        "must be an integer multiple of 16.");
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
AICORE void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    static_assert(TileData::Loc == pto::TileType::Vec || TileData::Loc == pto::TileType::Acc,
        "Source TileType only suport Vec/Acc!");
    PTO_ASSERT(dst.GetShape(0) > 0 && dst.GetShape(1) > 0 && dst.GetShape(2) > 0 && dst.GetShape(3) > 0 &&
                   dst.GetShape(4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
        "The shape of src and dst must be greater than 0!");
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        CheckStaticVec<TileData, GlobalData>();
        TStore<GlobalData, TileData>(dst.data(), src.data(), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
            dst.GetShape(3), dst.GetShape(4), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3),
            dst.GetStride(4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Acc) {
        CheckStaticAcc<TileData, GlobalData, false>();
        if constexpr (atomicType == AtomicType::AtomicAdd) {
            SetAtomicAdd<typename GlobalData::DType>();
        }
        TStoreAcc<GlobalData, TileData>(dst.data(), src.data(), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
            dst.GetShape(3), dst.GetShape(4), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3),
            dst.GetStride(4), src.GetValidRow(), src.GetValidCol());
        if constexpr (atomicType == AtomicType::AtomicAdd) {
            SetAtomicNone();
        }
    }
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
{}

}  // namespace pto
#endif
