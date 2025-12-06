/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

using namespace pto;

template <typename outType, typename aType, typename bType, typename biasType, int M, int K, int N, int validM,
    int validK, int validN, bool isBias>
__global__ __aicore__ void RunTMATMUL(
    __gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ biasType *src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<outType, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<Location::Mat, biasType, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<aType, M, K, validM, validK>;
    using RightTile = TileRight<bType, K, N, validK, validN>;
    using AccTile = TileAcc<outType, M, N, validM, validN>;
    using BiasTile = Tile<Location::Bias, outType, 1, N, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

template <typename outType, typename aType, typename bType, typename biasType, int M, int K, int N, bool isBias>
__global__ __aicore__ void RunTMATMUL_SPLIT_K(
    __gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ biasType *src2)
{
    constexpr int BASEM = 128;
    constexpr int BASEK = 64;
    constexpr int BASEN = 64;
    using GlobalDataSrc0 =
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, BASEK>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<bType, pto::Shape<1, 1, 1, BASEK, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut =
        GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, aType, BASEM, BASEK, BLayout::ColMajor, M, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, BASEK, BASEN, BLayout::ColMajor, BASEK, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<Location::Mat, biasType, 1, BASEN, BLayout::RowMajor, 1, BASEN>;

    using LeftTile = TileLeft<aType, BASEM, BASEK, M, BASEK>;
    using RightTile = TileRight<bType, BASEK, BASEN, BASEK, N>;
    using AccTile = TileAcc<outType, BASEM, BASEN, M, N>;
    using BiasTile = Tile<Location::Bias, outType, 1, BASEN, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    constexpr int iter = K / BASEK;
    for (int i = 0; i < iter; i++) { // baseK = 64
        /*************************************TLOAD****************************************/
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + i * BASEK * N);
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);
        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**********************************TMOV && TEXTRACT**********************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);
        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TMATMUL(cTile, aTile, bTile);
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
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
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<float, half, half, float, 128, 128, 64, 127, 128, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int8_t, 128, 128, 64, 128, 127, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL_SPLIT_K<float, half, half, float, 127, 128, 61, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), nullptr);
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<float, float, float, float, 128, 128, 64, 127, 127, 63, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), nullptr);
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, float, 128, 128, 64, 128, 128, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                reinterpret_cast<bfloat16_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<float, float8_e4m3_t, float8_e4m3_t, float, 128, 128, 64, 128, 128, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<float, float8_e4m3_t, float8_e5m2_t, float, 128, 128, 64, 128, 128, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL<float, float8_e5m2_t, float8_e4m3_t, float, 128, 128, 64, 128, 128, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 9) {
        RunTMATMUL<float, float8_e5m2_t, float8_e5m2_t, float, 128, 128, 64, 128, 128, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), nullptr);
    } else if constexpr (tilingKey == 10) {
        RunTMATMUL<float, hifloat8_t, hifloat8_t, float, 128, 128, 64, 128, 128, 64, false>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<hifloat8_t *>(src0),
                reinterpret_cast<hifloat8_t *>(src1), nullptr);
    }
}

template void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<float, half, half, half, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<float, half, half, bfloat16_t, 128, 128, 64, 128, 127, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                reinterpret_cast<half *>(src1), reinterpret_cast<bfloat16_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<float, bfloat16_t, bfloat16_t, bfloat16_t, 128, 128, 64, 128, 128, 63, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                reinterpret_cast<bfloat16_t *>(src1), reinterpret_cast<bfloat16_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL_SPLIT_K<float, half, half, float, 127, 128, 63, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<float, float, float, float, 128, 128, 64, 127, 128, 63, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                reinterpret_cast<float *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<float, float8_e4m3_t, float8_e4m3_t, float, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL<float, float8_e4m3_t, float8_e5m2_t, float, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 9) {
        RunTMATMUL<float, float8_e5m2_t, float8_e4m3_t, float, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                reinterpret_cast<float8_e4m3_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 10) {
        RunTMATMUL<float, float8_e5m2_t, float8_e5m2_t, float, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float8_e5m2_t *>(src0),
                reinterpret_cast<float8_e5m2_t *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 11) {
        RunTMATMUL<float, hifloat8_t, hifloat8_t, float, 128, 128, 64, 128, 128, 64, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<hifloat8_t *>(src0),
                reinterpret_cast<hifloat8_t *>(src1), reinterpret_cast<float *>(src2));
    }
}

template void LaunchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
