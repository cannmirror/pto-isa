/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, typename U, typename S, typename B, int M, int K, int N, int validM, int validK, int validN,
    bool isBias>
AICORE inline void RunTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2) {
    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 =
        GlobalTensor<B, pto::Shape<1, 1, 1, 1, validN>, pto::Stride<validN, validN, validN, validN, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<U, M, K, validM, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, N>;

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

    /******************************TLOAD*****************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    if constexpr (isBias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int M, int K, int N, int validM, int validK, int validN,
    bool isBias>
AICORE inline void RunTMATMULSplitK(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2) {
    constexpr int BASEK = 32;
    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, BASEK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, BASEK, validN>,
        pto::Stride<1 * BASEK * validN, 1 * BASEK * validN, BASEK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataOut dstGlobal(out);

    using GlobalDataSrc2 = GlobalTensor<B, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, U, M, BASEK, BLayout::ColMajor, M, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, BASEK, N, BLayout::ColMajor, BASEK, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, B, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<U, M, BASEK, validM, BASEK>;
    using RightTile = TileRight<S, BASEK, N, BASEK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;
    using BiasTile = Tile<TileType::Bias, B, 1, N, BLayout::RowMajor, 1, validN>;

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
    for (int i = 0; i < iter; i++) {
        GlobalDataSrc0 src0Global(src0 + i * BASEK);
        GlobalDataSrc1 src1Global(src1 + validN * i * BASEK);

        /******************************TLOAD*****************************/
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
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
                TMATMUL(cTile, aTile, bTile); // L0C清空
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile); // L0C不清空
        }
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);

    out = dstGlobal.data();
}

extern "C" __global__ AICORE void LaunchTMATMUL_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    RunTMATMUL<float, half, half, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0), reinterpret_cast<__gm__ half *>(src1), nullptr);
}

extern "C" __global__ AICORE void LaunchTMATMUL_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    RunTMATMUL<int32_t, int8_t, int8_t, int32_t, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0), reinterpret_cast<__gm__ int8_t *>(src1), nullptr);
}

extern "C" __global__ AICORE void LaunchTMATMUL_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t validM = 127;
    constexpr uint32_t validN = 63;
    constexpr uint32_t validK = 63;

    RunTMATMULSplitK<float, half, half, float, M, K, N, validM, validK, validN, false>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), nullptr);
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_1(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    RunTMATMUL<float, half, half, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0), reinterpret_cast<__gm__ half *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_2(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t validN = 63;

    RunTMATMUL<float, half, half, float, M, K, N, M, K, validN, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0), reinterpret_cast<__gm__ half *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_3(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 16;
    constexpr uint32_t N = 16;
    constexpr uint32_t K = 16;
    constexpr uint32_t validM = 15;
    constexpr uint32_t validN = 15;

    RunTMATMUL<float, float, float, float, M, K, N, validM, K, validN, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0), reinterpret_cast<__gm__ float *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_4(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t validM = 127;
    constexpr uint32_t validN = 63;
    constexpr uint32_t validK = 127;

    RunTMATMUL<int32_t, int8_t, int8_t, int32_t, M, K, N, validM, validK, validN, true>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), reinterpret_cast<__gm__ int32_t *>(src2));
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_5(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    RunTMATMUL<float, bfloat16_t, bfloat16_t, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0), reinterpret_cast<__gm__ bfloat16_t *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_6(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    RunTMATMUL<int32_t, int8_t, int8_t, int32_t, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0), reinterpret_cast<__gm__ int8_t *>(src1),
        reinterpret_cast<__gm__ int32_t *>(src2));
}

extern "C" __global__ AICORE void LaunchTMATMULBIAS_7(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1, __gm__ uint8_t *src2) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    RunTMATMULSplitK<int32_t, int8_t, int8_t, int32_t, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0), reinterpret_cast<__gm__ int8_t *>(src1),
        reinterpret_cast<__gm__ int32_t *>(src2));
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    if constexpr (tilingKey == 1) {
        LaunchTMATMUL_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        LaunchTMATMUL_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        LaunchTMATMUL_3<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream) {
    if constexpr (tilingKey == 1) {
        LaunchTMATMULBIAS_1<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 2) {
        LaunchTMATMULBIAS_2<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 3) {
        LaunchTMATMULBIAS_3<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 4) {
        LaunchTMATMULBIAS_4<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 5) {
        LaunchTMATMULBIAS_5<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 6) {
        LaunchTMATMULBIAS_6<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 7) {
        LaunchTMATMULBIAS_7<<<1, nullptr, stream>>>(out, src0, src1, src2);
    }
}

template void LaunchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMATMULBIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);