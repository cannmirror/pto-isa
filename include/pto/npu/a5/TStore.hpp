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
#include "pto/common.hpp"

namespace pto {
template <typename SrcType, typename DstType>
__aicore__ PTO_INLINE constexpr QuantMode_t GetCastPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, __gm__ bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::F322F16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
__aicore__ PTO_INLINE constexpr QuantMode_t GetScalarPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::QF322B8_PRE;
        } else if constexpr (std::is_same<DstType, __gm__ hifloat8_t>::value) {
            quantPre = QuantMode_t::QF322HIF8_PRE;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (std::is_same<DstType, __gm__ bfloat16_t>::value) {
            quantPre = QuantMode_t::QF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::DEQF16;
        } else if constexpr (std::is_same<DstType, __gm__ bfloat16_t>::value) {
            quantPre = QuantMode_t::QS322BF16_PRE;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
__aicore__ PTO_INLINE constexpr QuantMode_t GetVectorPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::VQF322B8_PRE;
        } else if constexpr (std::is_same<DstType, __gm__ hifloat8_t>::value) {
            quantPre = QuantMode_t::VQF322HIF8_PRE;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::VQF322F16_PRE;
        } else if constexpr (std::is_same<DstType, __gm__ bfloat16_t>::value) {
            quantPre = QuantMode_t::VQF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, __gm__ int8_t>::value) || (std::is_same<DstType, __gm__ uint8_t>::value)) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr (std::is_same<DstType, __gm__ half>::value) {
            quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (std::is_same<DstType, __gm__ bfloat16_t>::value) {
            quantPre = QuantMode_t::VQS322BF16_PRE;
        }
    }
    return quantPre;
}

template <typename T>
__aicore__ PTO_INLINE void SetAtomicAdd()
{
    static_assert((std::is_same_v<T, __gm__ half>) || (std::is_same_v<T, __gm__ float>) ||
                      (std::is_same_v<T, __gm__ int16_t>) || (std::is_same_v<T, __gm__ int32_t>) ||
                      (std::is_same_v<T, __gm__ int8_t>) || (std::is_same_v<T, __gm__ bfloat16_t>),
        "Dst and src must be half / float / int16_t / int32_t / int8_t / bfloat16_t");
    atomic_type_t atomicType = atomic_type_t::ATOMIC_NONE;
    if constexpr (std::is_same_v<T, __gm__ float>) {
        atomicType = atomic_type_t::ATOMIC_F32;
    } else if constexpr (std::is_same_v<T, __gm__ half>) {
        atomicType = atomic_type_t::ATOMIC_F16;
    } else if constexpr (std::is_same_v<T, __gm__ int16_t>) {
        atomicType = atomic_type_t::ATOMIC_S16;
    } else if constexpr (std::is_same_v<T, __gm__ int32_t>) {
        atomicType = atomic_type_t::ATOMIC_S32;
    } else if constexpr (std::is_same_v<T, __gm__ int8_t>) {
        atomicType = atomic_type_t::ATOMIC_S8;
    } else if constexpr (std::is_same_v<T, __gm__ bfloat16_t>) {
        atomicType = atomic_type_t::ATOMIC_BF16;
    }
    set_st_atomic_cfg(atomicType | (atomic_op_t::ATOMIC_SUM << 3));
}

template <typename TileData, typename GlobalData, bool isQuant>
__aicore__ PTO_INLINE void CheckStaticAcc()
{
    static_assert(std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, float>,
        "The input data type must be restricted to int32_t/float!");
    static_assert((GlobalData::layout == pto::Layout::ND) || (GlobalData::layout == pto::Layout::NZ),
        "TSTORE(GlobalTensor, AccTile) only support NZ2ND/NZ2NZ.");
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert((GlobalData::layout == pto::Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
                      (GlobalData::layout == pto::Layout::NZ && TileData::Rows >= 1 && TileData::Rows <= 65535 &&
                          TileData::Cols % 16 == 0),
        "When GlobalData is ND format, the range of Rows is [1, 8192]."
        "When GlobalData is NZ format, the range of Rows is [1, 65535] and Cols"
        "must be an integer multiple of 16.");
    if constexpr (!isQuant) {
        static_assert(std::is_same_v<typename GlobalData::DType, __gm__ int32_t> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ float> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ half> ||
                          std::is_same_v<typename GlobalData::DType, __gm__ bfloat16_t>,
            "The output data type must be restricted to int32_t/float/half/bfloat16_t!");
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void CheckStaticVec()
{
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Source dtype must be same with dst dtype!");
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float> ||
            std::is_same_v<typename TileData::DType, float8_e4m3_t> ||
            std::is_same_v<typename TileData::DType, float8_e5m2_t> ||
            std::is_same_v<typename TileData::DType, hifloat8_t> ||
            std::is_same_v<typename TileData::DType, float4_e1m2x2_t> ||
            std::is_same_v<typename TileData::DType, float4_e2m1x2_t>,
        "Data type must be "
        "int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/bfloat16_t/float/float8_e4m3_t/"
        "float8_e5m2_t/hifloat8_t/float4_e1m2x2_t/float4_e2m1x2_t!");
    static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
        "Src and dst layout must be same!");
    static_assert(
        ((GlobalData::layout == pto::Layout::ND) && (TileData::Cols * sizeof(typename TileData::DType) % 32 == 0)) ||
        ((GlobalData::layout == pto::Layout::DN) && (TileData::Rows * sizeof(typename TileData::DType) % 32 == 0)) ||
        (GlobalData::layout == pto::Layout::NZ));
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant>
__aicore__ PTO_INLINE void TStoreAccND(typename GlobalData::DType *dstGlobalAddr,
    __cc__ typename TileData::DType *srcTileAddr, int gShape3, int gShape4, int gStride2, int gStride3, int validRow,
    int validCol)
{
    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    constexpr uint16_t srcStride = TileData::Rows;
    uint32_t dstD = gStride3;

    uint16_t ndNum = validCol / gShape4;
    constexpr uint16_t c0 = 16;
    uint16_t srcNdStride = TileData::Rows * gShape4 * c0;
    uint16_t dstNdStride = gStride2;
    constexpr uint8_t unitFlagCtrl = 0;
    constexpr uint8_t nz2ndEn = 1;

    uint64_t xmReg = 0;
    xmReg = ((nSize & 0xfff) << 4) | (static_cast<uint64_t>(mSize & 0xffff) << 16) |
            (static_cast<uint64_t>(dstD & 0xffffffff) << 32);
    uint64_t xtReg = 0;
    xtReg = srcStride | (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |
            (static_cast<uint64_t>(quantPre & 0x1f) << 34) | (static_cast<uint64_t>(nz2ndEn & 0x1) << 43);
    uint64_t config = ndNum | (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) |
                      (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);
    set_loop3_para(config);
    copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant>
__aicore__ PTO_INLINE void TStoreAccNZ(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
    typename GlobalData::DType *dstGlobalAddr, __cc__ typename TileData::DType *srcTileAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int validRow, int validCol)
{
    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    constexpr uint16_t srcStride = TileData::Rows;
    constexpr uint8_t unitFlagCtrl = 0;
    uint8_t channelSplitEn = 0;

    constexpr uint16_t cubeBlock = 16;
    uint16_t c0Size = 16;
    if constexpr (sizeof(typename TileData::DType) == 1) {
        c0Size = 32;
    } else if constexpr (std::is_same_v<typename TileData::DType, float> &&
                         std::is_same_v<typename GlobalData::DType, __gm__ float>) {
        if (gShape4 == 8) {
            c0Size = 8;
            channelSplitEn = 1;
        }
    }
    uint32_t dstStride = (gShape2 * gShape3 + cubeBlock - 1) / cubeBlock * cubeBlock * c0Size;

    uint64_t xtReg = 0;
    xtReg = srcStride | (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) |
            (static_cast<uint64_t>(quantPre & 0x1f) << 34) | (static_cast<uint64_t>(channelSplitEn & 0x1) << 42);
    uint64_t xmReg = 0;
    xmReg = ((static_cast<uint64_t>(nSize & 0xfff) << 4) | (static_cast<uint64_t>(mSize & 0xffff) << 16) |
             (static_cast<uint64_t>(dstStride & 0xffffffff) << 32));

    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    for (uint32_t i = 0; i < gShape0; ++i) {
        dstGlobalAddr = dstAddr + i * gStride0;
        srcTileAddr = srcAddr + i * tileStride;
        copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
    }
}

template <typename GlobalData, typename TileData, typename FpTileData, QuantMode_t quantPre = QuantMode_t::NoQuant>
__tf__ __aicore__ void TStoreAccFp(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
    typename FpTileData::TileDType __in__ fp, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cc__ typename TileData::DType *srcAddr = (__cc__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        __cc__ typename TileData::DType *srcTileAddr = srcAddr;
        typename GlobalData::DType *dstGlobalAddr = dstAddr;
        TStoreAccNZ<GlobalData, TileData, quantPre>(dstAddr, srcAddr, dstGlobalAddr, srcTileAddr, gShape0, gShape1,
            gShape2, gShape3, gShape4, gStride0, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::ND) {
        TStoreAccND<GlobalData, TileData, quantPre>(
            dstAddr, srcAddr, gShape3, gShape4, gStride2, gStride3, validRow, validCol);
    }
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant>
__tf__ __aicore__ void TStoreAcc(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    __cc__ typename TileData::DType *srcAddr = (__cc__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;
    if constexpr (GlobalData::layout == pto::Layout::ND) {
        TStoreAccND<GlobalData, TileData, quantPre>(
            dstAddr, srcAddr, gShape3, gShape4, gStride2, gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NZ) {
        __cc__ typename TileData::DType *srcTileAddr = srcAddr;
        typename GlobalData::DType *dstGlobalAddr = dstAddr;
        TStoreAccNZ<GlobalData, TileData, quantPre>(dstAddr, srcAddr, dstGlobalAddr, srcTileAddr, gShape0, gShape1,
            gShape2, gShape3, gShape4, gStride0, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TStoreInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src,
    uint16_t nBurst, uint32_t lenBurst, uint64_t burstDstStride, uint32_t burstSrcStride)
{
    copy_ubuf_to_gm_align_v2(dst, src, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
}

template <typename GlobalData, typename TileData>
__aicore__ PTO_INLINE void TStoreVecND(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    uint32_t loop1SrcStride = GetByteSize<typename TileData::DType>(gShape3 * TileData::Cols);
    uint32_t loop1DstStride = GetByteSize<typename TileData::DType>(gStride2);
    uint32_t loop2SrcStride = GetByteSize<typename TileData::DType>(gShape2 * gShape3 * TileData::Cols);
    uint32_t loop2DstStride = GetByteSize<typename TileData::DType>(gStride1);

    uint64_t loopSizeConfig = 0;
    uint64_t loop1Size = gShape2 & 0x1FFFFF;
    loopSizeConfig |= loop1Size;
    uint64_t loop2Size = (static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21;
    loopSizeConfig |= loop2Size;
    set_loop_size_ubtoout(loopSizeConfig);

    uint64_t loop1Config = 0;
    loop1Config |= ((uint64_t)loop1SrcStride) << 40;
    loop1Config |= (uint64_t)loop1DstStride;
    set_loop1_stride_ubtoout(loop1Config);
    uint64_t loop2Config = 0;
    loop2Config |= ((uint64_t)loop2SrcStride) << 40;
    loop2Config |= (uint64_t)loop2DstStride;
    set_loop2_stride_ubtoout(loop2Config);
    uint64_t srcStride0 = gShape1 * gShape2 * gShape3 * TileData::Cols;
    if constexpr (std::is_same<typename TileData::DType, float4_e1m2x2_t>::value ||
                  std::is_same<typename TileData::DType, float4_e2m1x2_t>::value) {
        srcStride0 = srcStride0 >> 1; // fp4 srcAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 dstAddr offset need divide 2 as use b8 to move
    }
    uint16_t nBurst = gShape3;

    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validCol);
    uint64_t burstDstStride = GetByteSize<typename TileData::DType>(gStride3);
    uint32_t burstSrcStride = GetByteSize<typename TileData::DType>(TileData::Cols);
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * srcStride0;
        TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}
template <typename GlobalData, typename TileData>
__aicore__ PTO_INLINE void TStoreVecDN(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    uint32_t loop1SrcStride = GetByteSize<typename TileData::DType>(TileData::Rows * gShape4);
    uint32_t loop1DstStride = GetByteSize<typename TileData::DType>(gStride2);
    uint32_t loop2SrcStride = GetByteSize<typename TileData::DType>(gShape2 * TileData::Rows * gShape4);
    uint32_t loop2DstStride = GetByteSize<typename TileData::DType>(gStride1);

    uint64_t loop1Config = 0;
    loop1Config |= ((uint64_t)loop1SrcStride) << 40;
    loop1Config |= (uint64_t)loop1DstStride;
    set_loop1_stride_ubtoout(loop1Config);
    uint64_t loop2Config = 0;
    loop2Config |= ((uint64_t)loop2SrcStride) << 40;
    loop2Config |= (uint64_t)loop2DstStride;
    set_loop2_stride_ubtoout(loop2Config);

    uint64_t loopSizeConfig = 0;
    uint64_t loop1Size = gShape2 & 0x1FFFFF;
    loopSizeConfig |= loop1Size;
    uint64_t loop2Size = (static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21;
    loopSizeConfig |= loop2Size;
    set_loop_size_ubtoout(loopSizeConfig);

    uint64_t srcStride0 = gShape1 * gShape2 * gShape4 * TileData::Rows;
    uint16_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validRow);
    uint64_t burstDstStride = GetByteSize<typename TileData::DType>(gStride4);
    uint32_t burstSrcStride = GetByteSize<typename TileData::DType>(TileData::Rows);
    if constexpr (std::is_same<typename TileData::DType, float4_e1m2x2_t>::value ||
                  std::is_same<typename TileData::DType, float4_e2m1x2_t>::value) {
        srcStride0 = srcStride0 >> 1; // fp4 srcAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 dstAddr offset need divide 2 as use b8 to move
    }

    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * srcStride0;
        TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}

template <typename GlobalData, typename TileData>
__aicore__ PTO_INLINE void TStoreVecNZ(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint64_t burstDstStride = GetByteSize<typename TileData::DType>(gStride1);
    uint32_t burstSrcStride = TileData::Rows * C0_SIZE_BYTE;
    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    if constexpr (std::is_same<typename TileData::DType, float4_e1m2x2_t>::value ||
                  std::is_same<typename TileData::DType, float4_e2m1x2_t>::value) {
        tileStride = tileStride >> 1; // fp4 srcAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 dstAddr offset need divide 2 as use b8 to move
    }
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * tileStride;
        TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}
template <typename GlobalData, typename TileData>
__tf__ __aicore__ void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename TileData::DType *srcAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;

    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreVecND<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreVecDN<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        TStoreVecNZ<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
            gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    static_assert(TileData::Loc == pto::Location::Vec || TileData::Loc == pto::Location::Acc,
        "Source location only suport Vec/Acc!");
    if constexpr (TileData::Loc == pto::Location::Vec) {
        CheckStaticVec<TileData, GlobalData>();

        TStore<GlobalData, TileData>(dst.data(), src.data(), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
            dst.GetShape(3), dst.GetShape(4), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3),
            dst.GetStride(4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (TileData::Loc == pto::Location::Acc) {
        using L0cT = typename TileData::DType;
        using DstT = typename GlobalData::DType;
        CheckStaticAcc<TileData, GlobalData, false>();

        if constexpr (atomicType == AtomicType::AtomicAdd) {
            SetAtomicAdd<DstT>();
        }
        constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
        TStoreAcc<GlobalData, TileData, quantPre>(dst.data(), src.data(), dst.GetShape(0), dst.GetShape(1),
            dst.GetShape(2), dst.GetShape(3), dst.GetShape(4), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2),
            dst.GetStride(3), dst.GetStride(4), src.GetValidRow(), src.GetValidCol());
        if constexpr (atomicType == AtomicType::AtomicAdd) {
            set_st_atomic_cfg(atomic_type_t::ATOMIC_NONE, atomic_op_t::ATOMIC_SUM);
        }
    }
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{
    using L0cT = typename TileData::DType;
    using DstT = typename GlobalData::DType;
    CheckStaticAcc<TileData, GlobalData, true>();

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<DstT>();
    }
    constexpr QuantMode_t quantPre = GetScalarPreQuantModeGm<L0cT, DstT>();
    set_quant_pre(preQuantScalar);
    TStoreAcc<GlobalData, TileData, quantPre>(dst.data(), src.data(), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
        dst.GetShape(3), dst.GetShape(4), dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3),
        dst.GetStride(4), src.GetValidRow(), src.GetValidCol());
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_st_atomic_cfg(atomic_type_t::ATOMIC_NONE, atomic_op_t::ATOMIC_SUM);
    }
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
{
    using L0cT = typename TileData::DType;
    using DstT = typename GlobalData::DType;
    CheckStaticAcc<TileData, GlobalData, true>();

    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<DstT>();
    }
    constexpr QuantMode_t quantPre = GetVectorPreQuantModeGm<L0cT, DstT>();
    TStoreAccFp<GlobalData, TileData, FpTileData, quantPre>(dst.data(), src.data(), fp.data(), dst.GetShape(0),
        dst.GetShape(1), dst.GetShape(2), dst.GetShape(3), dst.GetShape(4), dst.GetStride(0), dst.GetStride(1),
        dst.GetStride(2), dst.GetStride(3), dst.GetStride(4), src.GetValidRow(), src.GetValidCol());
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_st_atomic_cfg(atomic_type_t::ATOMIC_NONE, atomic_op_t::ATOMIC_SUM);
    }
}
} // namespace pto
#endif
