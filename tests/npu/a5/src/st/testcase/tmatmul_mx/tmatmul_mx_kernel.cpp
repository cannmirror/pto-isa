/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T>
AICORE inline constexpr T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
AICORE inline constexpr T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
AICORE inline void DynGM2L1(__cbuf__ T *dst, __gm__ T *src, unsigned TShape0, unsigned TShape1)
{
    if (std::is_same<T, float4_e2m1x2_t>::value || std::is_same<T, float4_e1m2x2_t>::value) {
        uint32_t lenBurst = TShape0 * TShape1 * sizeof(T) / 2;
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t *)dst, (__gm__ uint8_t *)src, 0, 1, lenBurst, 0, 0, 0, 0, 0, 0);
    } else {
        uint32_t lenBurst = TShape0 * TShape1 * sizeof(T);
        copy_gm_to_cbuf_align_v2(
            dst, src, 0 /*sid*/, 1 /*nBrust*/, lenBurst, 0, 0, 0, 0, 0 /*srcStride*/, 0 /*dstStride*/);
    }
}

template <typename T>
AICORE inline void DynGM2L1MXA(__cbuf__ T *dst, __gm__ T *src, unsigned TShape0, unsigned TShape1, unsigned iter)
{
    uint32_t lenBurst = 16 * TShape1 * sizeof(T);
    uint32_t nBurst = CeilDiv<uint32_t>(TShape0, 16);
    uint32_t srcStride = lenBurst * iter;

    copy_gm_to_cbuf_align_v2(dst, src, 0 /*sid*/, nBurst, lenBurst, 0, 0, 0, 0, srcStride, lenBurst /*dstStride*/);
}

template <typename T>
AICORE inline void DynGM2L1MXB(__cbuf__ T *dst, __gm__ T *src, unsigned TShape0, unsigned TShape1, unsigned iter)
{
    uint32_t lenBurst = TShape0 * 16 * sizeof(T);
    uint32_t nBurst = CeilDiv<uint32_t>(TShape1, 16);
    uint32_t srcStride = lenBurst * iter;

    copy_gm_to_cbuf_align_v2(dst, src, 0 /*sid*/, nBurst, lenBurst, 0, 0, 0, 0, srcStride, lenBurst /*dstStride*/);
}

template <typename scaleType>
AICORE inline void DynL12L0AMX(__ca__ scaleType *dst, __cbuf__ scaleType *src, unsigned dstM, unsigned dstK)
{
    uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16;
    constexpr int c0Size = 2;
    uint8_t mStep = CeilDiv<uint8_t>(dstK, c0Size);
    uint8_t kStep = CeilDiv<uint8_t>(dstM, 16);
    uint8_t srcStride = CeilDiv<uint8_t>(dstM, 16);
    uint8_t dstStride = CeilDiv<uint8_t>(dstM, 16);
    load_cbuf_to_ca_mx(mxDstAddr, static_cast<__cbuf__ void *>(src), 0, 0, mStep, kStep, srcStride, dstStride);
}

template <typename scaleType>
AICORE inline void DynL12L0BMX(__cb__ scaleType *dst, __cbuf__ scaleType *src, unsigned dstK, unsigned dstN)
{
    uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16;
    constexpr int c0Size = 2;
    uint8_t kStep = CeilDiv<uint8_t>(dstK, c0Size);
    uint8_t nStep = CeilDiv<uint8_t>(dstN, 16);
    uint8_t srcStride = CeilDiv<uint8_t>(dstN, 16);
    uint8_t dstStride = CeilDiv<uint8_t>(dstN, 16);
    load_cbuf_to_cb_mx(mxDstAddr, static_cast<__cbuf__ void *>(src), 0, 0, kStep, nStep, srcStride, dstStride);
}

template <typename OutType, typename AType, typename BType, typename BiasType, int M, int K, int N, int validM,
    int validK, int validN, bool isBias, bool isFp4>
__global__ AICORE void RunTMATMULMX(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ uint8_t *src2,
    __gm__ uint8_t *src3, __gm__ BiasType *src4)
{
    constexpr uint8_t kMX = CeilDiv(K, 32);
    constexpr int kAlign = CeilAlign(validK, 64);

    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, validM, kAlign>,
        pto::Stride<1 * validM * kAlign, 1 * validM * kAlign, validM * kAlign, kAlign, 1>>;
    using GlobalDataSrc1 = GlobalTensor<BType, pto::Shape<1, 1, 1, kAlign, validN>,
        pto::Stride<1 * kAlign * validN, 1 * kAlign * validN, kAlign * validN, validN, 1>>;
    using GlobalDataSrc2 =
        GlobalTensor<uint8_t, pto::Shape<1, 1, 1, M, kMX>, pto::Stride<M * kMX, M * kMX, M * kMX, kMX, 1>>;
    using GlobalDataSrc3 =
        GlobalTensor<uint8_t, pto::Shape<1, 1, 1, kMX, N>, pto::Stride<N * kMX, N * kMX, N * kMX, N, 1>>;
    using GlobalDataSrc4 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataSrc4 src4Global(src4);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, kAlign, BLayout::ColMajor, validM, kAlign, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, kAlign, N, BLayout::ColMajor, kAlign, validN, SLayout::RowMajor, 512>;

    using TileScaleAData =
        Tile<TileType::Mat, uint8_t, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileScaleBData =
        Tile<TileType::Mat, uint8_t, K, N, BLayout::ColMajor, validK, validN, SLayout::ColMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<AType, M, kAlign, validM, kAlign>;
    using RightTile = TileRight<BType, kAlign, N, kAlign, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleTile;
    TileScaleBData bScaleTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * K);
    TASSIGN(aScaleTile, M * K + K * N);
    TASSIGN(bScaleTile, M * K + K * N + M * kMX);
    TASSIGN(biasDataTile, M * K + K * N + M * kMX + N * kMX);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    __cbuf__ uint8_t *srcAScaleAddr = aScaleTile.data();
    __cbuf__ uint8_t *srcBScaleAddr = bScaleTile.data();

    __ca__ uint8_t *amx = (__ca__ uint8_t *)(aTile.data());
    __cb__ uint8_t *bmx = (__cb__ uint8_t *)(bTile.data());

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    DynGM2L1<uint8_t>(srcAScaleAddr, src2, M, kMX);
    DynGM2L1<uint8_t>(srcBScaleAddr, src3, kMX, N);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src4Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    DynL12L0AMX<uint8_t>(amx, srcAScaleAddr, M, kMX);
    DynL12L0BMX<uint8_t>(bmx, srcBScaleAddr, kMX, N);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    if constexpr (isBias) {
        TMATMUL_MX(cTile, aTile, amx, bTile, bmx, biasTile);
    } else {
        TMATMUL_MX(cTile, aTile, amx, bTile, bmx);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename OutType, typename AType, typename BType, typename BiasType, int M, int K, int N, int validM,
    int validK, int validN, bool isBias, bool isFp4>
__global__ AICORE void RunTMATMULMX_SPLIT_K(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1,
    __gm__ uint8_t *src2, __gm__ uint8_t *src3, __gm__ BiasType *src4)
{
    constexpr int BASEK = 64;
    constexpr int BASEKMX = CeilDiv(BASEK, 32);

    using GlobalDataSrc0 = GlobalTensor<AType, pto::Shape<1, 1, 1, validM, BASEK>,
        pto::Stride<1 * validM * K, 1 * validM * K, validM * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<BType, pto::Shape<1, 1, 1, BASEK, validN>,
        pto::Stride<1 * K * validN, 1 * K * validN, K * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<uint8_t, pto::Shape<1, 1, 1, validM, BASEKMX>,
        pto::Stride<validM * BASEKMX, validM * BASEKMX, validM * BASEKMX, BASEKMX, 1>>;
    using GlobalDataSrc3 = GlobalTensor<uint8_t, pto::Shape<1, 1, 1, BASEKMX, validN>,
        pto::Stride<validN * BASEKMX, validN * BASEKMX, validN * BASEKMX, validN, 1>>;

    using GlobalDataOut = GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc4 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    GlobalDataSrc4 src4Global(src4);

    using TileMatAData = Tile<TileType::Mat, AType, M, BASEK, BLayout::ColMajor, validM, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::RowMajor, 512>;

    using TileScaleAData =
        Tile<TileType::Mat, uint8_t, M, BASEK, BLayout::RowMajor, validM, BASEK, SLayout::RowMajor, 512>;
    using TileScaleBData =
        Tile<TileType::Mat, uint8_t, BASEK, N, BLayout::ColMajor, BASEK, validN, SLayout::ColMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<AType, M, BASEK, validM, BASEK>;
    using RightTile = TileRight<BType, BASEK, N, BASEK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, BiasType, 1, N, BLayout::RowMajor, 1, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileScaleAData aScaleTile;
    TileScaleBData bScaleTile;
    TileBiasData biasDataTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, M * BASEK);
    TASSIGN(aScaleTile, M * BASEK + N * BASEK);
    TASSIGN(bScaleTile, M * BASEK + N * BASEK + M * BASEKMX);
    TASSIGN(biasDataTile, M * BASEK + N * BASEK + M * BASEKMX + N * BASEKMX);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;

    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    __cbuf__ uint8_t *srcAScaleAddr = aScaleTile.data();
    __cbuf__ uint8_t *srcBScaleAddr = bScaleTile.data();

    __ca__ uint8_t *amx = (__ca__ uint8_t *)(aTile.data());
    __cb__ uint8_t *bmx = (__cb__ uint8_t *)(bTile.data());

    constexpr int iter = K / BASEK;
    for (int i = 0; i < iter; i++) {
        const int offsetA = (!isFp4) ? (i * BASEK) : (i * BASEK / 2);
        const int offsetB = (!isFp4) ? (validN * i * BASEK) : (validN * i * BASEK / 2);
        GlobalDataSrc0 src0Global(src0 + offsetA);
        GlobalDataSrc1 src1Global(src1 + offsetB);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        const int offsetAMX = i * BASEKMX * 16;
        const int offsetBMX = 16 * i * BASEKMX;

        DynGM2L1MXA<uint8_t>(srcAScaleAddr, src2 + offsetAMX, M, BASEKMX, iter);
        DynGM2L1MXB<uint8_t>(srcBScaleAddr, src3 + offsetBMX, BASEKMX, N, iter);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src4Global);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);
        
        DynL12L0AMX<uint8_t>(amx, srcAScaleAddr, M, BASEKMX);
        DynL12L0BMX<uint8_t>(bmx, srcBScaleAddr, BASEKMX, N);

        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_MX(cTile, aTile, amx, bTile, bmx, biasTile);
            } else {
                TMATMUL_MX(cTile, aTile, amx, bTile, bmx);
            }
        } else {
            TMATMUL_MX(cTile, cTile, aTile, amx, bTile, bmx);
        }
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL_MX(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULMX<float, float8_e5m2_t, float8_e5m2_t, float, 128, 64, 64, 128, 64, 64, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float, 128, 128, 64, 127, 72, 64, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e5m2_t, float, 128, 128, 64, 128, 110, 63, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e2m1x2_t, float, 128, 64, 64, 128, 64, 64, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMULMX<float, float4_e1m2x2_t, float4_e2m1x2_t, float, 128, 64, 64, 117, 64, 60, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float, 128, 128, 64, 128, 118, 64, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 7) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float, 128, 64, 64, 115, 64, 30, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 8) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float, 16, 64, 32, 16, 32, 16, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 9) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e5m2_t, float, 16, 64, 64, 10, 50, 54, false, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    } else if constexpr (tilingKey == 10) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e2m1x2_t, float, 16, 64, 64, 4, 30, 8, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                reinterpret_cast<float4_e2m1x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), nullptr);
    }
}

template void LaunchTMATMUL_MX<1>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<2>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<3>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<4>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<5>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<6>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<7>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<8>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<9>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);
template void LaunchTMATMUL_MX<10>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, void *stream);

template <int32_t tilingKey>
void LaunchTMATMUL_MX_BIAS(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULMX<float, float8_e5m2_t, float8_e4m3_t, float, 128, 64, 32, 115, 64, 30, true, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 2) {
        RunTMATMULMX<float, float8_e4m3_t, float8_e4m3_t, float, 208, 192, 96, 200, 192, 95, true, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 3) {
        RunTMATMULMX<float, float4_e2m1x2_t, float4_e1m2x2_t, float, 48, 128, 64, 35, 128, 56, true, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e2m1x2_t *>(src0),
                reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 4) {
        RunTMATMULMX_SPLIT_K<float, float4_e1m2x2_t, float4_e1m2x2_t, float, 48, 128, 64, 47, 128, 62, true, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 5) {
        RunTMATMULMX_SPLIT_K<float, float8_e4m3_t, float8_e5m2_t, float, 64, 128, 64, 64, 65, 64, true, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), reinterpret_cast<float *>(src4));
    } else if constexpr (tilingKey == 6) {
        RunTMATMULMX<float, float4_e1m2x2_t, float4_e1m2x2_t, float, 16, 64, 64, 1, 64, 62, true, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float4_e1m2x2_t *>(src0),
                reinterpret_cast<float4_e1m2x2_t *>(src1), reinterpret_cast<uint8_t *>(src2),
                reinterpret_cast<uint8_t *>(src3), reinterpret_cast<float *>(src4));
    }
}

template void LaunchTMATMUL_MX_BIAS<1>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<2>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<3>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<4>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<5>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream);
template void LaunchTMATMUL_MX_BIAS<6>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3, uint8_t *src4, void *stream);