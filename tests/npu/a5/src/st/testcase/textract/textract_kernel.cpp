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

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t< isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t< isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>, pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t< isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t< isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, mValid, kValid>;
    using RightTile = TileRight<S, kValid, nValid, kValid, nValid>;
    using AccTile = TileAcc<T, mValid, nValid, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexM, uint16_t indexK, uint16_t indexN,  bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT_DYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int k, int n)
{
    constexpr int mValid = M - indexM;
    constexpr int kValid = K - indexK;
    constexpr int nValid = N - indexN;

    using GlobalDataSrc0 = std::conditional_t< isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t< isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;

    using DynShape3Dim5 = pto::Shape<1, 1, 1, mValid, nValid>;
    using DynSTrid3Dim5 = pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShape3Dim5, DynSTrid3Dim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t< isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t< isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, -1, -1>;
    using RightTile = TileRight<S, kValid, nValid, kValid, -1>;
    using AccTile = TileAcc<T, mValid, nValid, -1, nValid>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    int validM = m - indexM;
    int validK = k - indexK;
    int validN = n - indexN;

    LeftTile aTile(validM, validK);
    RightTile bTile(validN);
    AccTile cTile(validM);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, half, half, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 48;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, float, float, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexK = 16;
    constexpr uint16_t indexN = 16;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, half, half, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexK = 32;
    constexpr uint16_t indexN = 16;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, float, float, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexK = 64;
    constexpr uint16_t indexN = 32;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 64;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTEXTRACT<float, half, half, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 128;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 32;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTEXTRACT<float, float, float, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 32;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, float8_e4m3_t, float8_e4m3_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 32;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;
    runTEXTRACT<float, float8_e5m2_t, float8_e5m2_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 32;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;
    runTEXTRACT<float, hifloat8_t, hifloat8_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 32;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexK = 0;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT_DYNAMIC<int32_t, int8_t, int8_t, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, K, N);
}
extern "C" __global__ __aicore__ void launchTEXTRACT_15(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 48;
    constexpr uint32_t N = 96;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexK = 16;
    constexpr uint16_t indexN = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT_DYNAMIC<float, half, half, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, K, N);
}
extern "C" __global__ __aicore__ void launchTEXTRACT_16(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 48;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexK = 32;
    constexpr uint16_t indexN = 16;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT_DYNAMIC<float, float, float, M, K, N, indexM, indexK, indexN, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1), M, K, N);
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTEXTRACT_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTEXTRACT_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTEXTRACT_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTEXTRACT_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTEXTRACT_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTEXTRACT_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTEXTRACT_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTEXTRACT_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTEXTRACT_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTEXTRACT_10<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 11) {
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 12) {
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 13) {
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 14) {
        launchTEXTRACT_14<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 15) {
        launchTEXTRACT_15<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 16) {
        launchTEXTRACT_16<<<1, nullptr, stream>>>(out, src0, src1);
    } 
}

template void launchTEXTRACT<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=1 的版本
template void launchTEXTRACT<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<15>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<16>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 

template <typename T, typename U, typename S, int M, int K, int N, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = std::conditional_t< isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t< isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t< isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t< isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTMOV_DYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int k, int n)
{
    using GlobalDataSrc0 = std::conditional_t< isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t< isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>>;
        
    using DynShape3Dim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTrid3Dim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;
    using GlobalDataOut = GlobalTensor<T, DynShape3Dim5, DynSTrid3Dim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t< isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t< isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, M, K, -1, -1>;
    using RightTile = TileRight<S, K, N, K, -1>;
    using AccTile = TileAcc<T, M, N, -1, N>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile(m, k);
    RightTile bTile(n);
    AccTile cTile(m);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, bool isAtranspose, bool isBtranspose, int targetM, int targetK, int targetN>
__aicore__ inline void runTMOV_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = std::conditional_t< isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>, pto::Stride<1*targetM*targetK, 1*targetM*targetK, targetM*targetK, 1, targetM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>, pto::Stride<1*targetM*targetK, 1*targetM*targetK, targetM*targetK, targetK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t< isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>, pto::Stride<1*targetK*targetN, 1*targetK*targetN, targetK*targetN, targetN, 1>, Layout::ND>,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>, pto::Stride<1*targetK*targetN, 1*targetK*targetN, targetK*targetN, 1, targetK>, Layout::DN>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t< isAtranspose,
        Tile<Location::Mat, U, targetM, targetK, BLayout::RowMajor, targetM, targetK, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, targetM, targetK, BLayout::ColMajor, targetM, targetK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t< isBtranspose,
        Tile<Location::Mat, S, targetK, targetN, BLayout::ColMajor, targetK, targetN, SLayout::RowMajor, 512>,
        Tile<Location::Mat, S, targetK, targetN, BLayout::RowMajor, targetK, targetN, SLayout::ColMajor, 512>>;

    using LeftTile = TileLeft<U, targetM, targetK, targetM, K>;
    using RightTile = TileRight<S, targetK, targetN, K, targetN>;
    using AccTile = TileAcc<T, targetM, targetN, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTMOV_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTMOV<float, half, half, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 48;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTMOV<float, float, float, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTMOV<int32_t, int8_t, int8_t, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTMOV<float, bfloat16_t, bfloat16_t, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTMOV<float, float8_e4m3_t, float8_e4m3_t, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;
    runTMOV<float, float8_e5m2_t, float8_e5m2_t, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTMOV<float, hifloat8_t, hifloat8_t, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTMOV_DYNAMIC<int32_t, int8_t, int8_t, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, K, N);
}
extern "C" __global__ __aicore__ void launchTMOV_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTMOV_DYNAMIC<float, half, half, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, K, N);
}
extern "C" __global__ __aicore__ void launchTMOV_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;
    runTMOV_DYNAMIC<float, float, float, M, K, N, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1), M, K, N);
}
extern "C" __global__ __aicore__ void launchTMOV_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t K = 40;
    constexpr uint32_t N = 66;

    constexpr uint32_t targetM = 96;
    constexpr uint32_t targetK = 64;
    constexpr uint32_t targetN = 96;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, K, N, isAtranspose, isBtranspose, targetM, targetK, targetN>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t K = 40;
    constexpr uint32_t N = 66;

    constexpr uint32_t targetM = 80;
    constexpr uint32_t targetK = 48;
    constexpr uint32_t targetN = 80;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTMOV_UNALIGN<float, half, half, M, K, N, isAtranspose, isBtranspose, targetM, targetK, targetN>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t K = 40;
    constexpr uint32_t N = 66;

    constexpr uint32_t targetM = 80;
    constexpr uint32_t targetK = 48;
    constexpr uint32_t targetN = 80;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    runTMOV_UNALIGN<float, float, float, M, K, N, isAtranspose, isBtranspose, targetM, targetK, targetN>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

template <int32_t tilingKey>
void launchTMOV(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMOV_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMOV_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMOV_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMOV_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTMOV_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTMOV_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTMOV_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTMOV_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTMOV_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTMOV_10<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTMOV_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTMOV_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTMOV_13<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTMOV<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=1 的版本
template void launchTMOV<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTMOV<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 