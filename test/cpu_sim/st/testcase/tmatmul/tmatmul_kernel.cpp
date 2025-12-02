/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <common/pto_tileop.hpp>
#include <common/tile_tensor_impl.hpp>
#include <common/constants.hpp>

using namespace pto;

template <typename T, typename U, typename S, int M, int K, int N>
__aicore__ inline void runTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut  = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    int offset = 0;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // L1上都是大n小z
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

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

    /******************************TLOAD*****************************/
    TLOAD(aMatTile,src0Global);
    TLOAD(bMatTile,src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /********************************TSTORE****************************/
    TSTORE(dstGlobal,cTile);

    out = dstGlobal.data();
}   

template <typename T, typename U, typename S, int M, int K, int N>
__aicore__ inline void runTMATMUL_SPLIT_K(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, uint32_t numRepeats)
{
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut  = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // L1上都是大n小z
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

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

    for(uint32_t i=0; i<numRepeats; i++) {
        /******************************TLOAD*****************************/
        GlobalDataSrc0 src0Global(src0+i*M*K);
        GlobalDataSrc1 src1Global(src1+i*K*N);

        TLOAD(aMatTile,src0Global);
        TLOAD(bMatTile,src1Global);

        /**************************TMOV && TEXTRACT**************************/
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        if (i == 0) {
            TMATMUL(cTile, aTile, bTile); 
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
    }

    /********************************TSTORE****************************/
    TSTORE(dstGlobal,cTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTMATMUL_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<int32_t, int8_t, int8_t, M, K, N>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMATMUL_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, half, half, M, K, N>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMATMUL_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t repeats = 5;

    runTMATMUL_SPLIT_K<float, half, half, M, K, N>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        repeats);
}
extern "C" __global__ __aicore__ void launchTMATMUL_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 16;    
    constexpr uint32_t N = 32;

    runTMATMUL<float, float, float, M, K, N>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

template <int32_t tilingKey>
void launchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMATMUL_1(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMATMUL_2(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMATMUL_3(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMATMUL_4(out, src0, src1);
    }
}

template void launchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 tilingKey=1
template void launchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 tilingKey=2
template void launchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 tilingKey=3
template void launchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 tilingKey=3
