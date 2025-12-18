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
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, typename U, typename S, int M, int K, int N, uint32_t baseM, uint32_t baseK, uint32_t baseN>
AICORE inline void ProcessKIteration(uint32_t kIter, __gm__ U *currentSrc0, __gm__ S *currentSrc1,
    Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor> aMatTile[2], 
    Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor> bMatTile[2], 
    TileLeft<U, baseM, baseK, baseM, baseK> aTile[2], TileRight<S, baseK, baseN, baseK, baseN> bTile[2], 
    TileAcc<T, baseM, baseN, baseM, baseN> &cTile)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK>;
    using NDsingleCoreShapeA = BaseShape2D<U, M, K>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA>;

    using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, K, N, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    int cur = kIter % 2;
    GlobalDataSrcA gmA(currentSrc0 + kIter * baseK);
    GlobalDataSrcB gmB(currentSrc1 + kIter * baseK);

    wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);
    TLOAD(aMatTile[cur], gmA);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TLOAD(bMatTile[cur], gmB);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    
    wait_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile[cur], aMatTile[cur]);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(bTile[cur], bMatTile[cur]);
    set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);

    set_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
    wait_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
    if (kIter == 0) {
        TMATMUL(cTile, aTile[cur], bTile[cur]);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile[cur], bTile[cur]);
    }
    set_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
}

template <typename T, typename U, typename S, int M, int K, int N, uint32_t singleCoreM, uint32_t singleCoreK, 
    uint32_t singleCoreN, uint16_t baseM, uint16_t baseK, uint16_t baseN>
AICORE inline void runGEMMBASIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    // get gm offset of each core
    constexpr uint32_t mIter = M / singleCoreM;
    uint32_t mIterIdx = get_block_idx() % mIter; // get current launch core idx
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t currentGmOffsetA = mIterIdx * singleCoreM * K;
    uint64_t currentGmOffsetB = nIterIdx * K * singleCoreN;
    uint64_t currentGmOffsetC = mIterIdx * singleCoreM * N + nIterIdx * singleCoreN;
    __gm__ U *currentSrc0 = src0 + currentGmOffsetA;
    __gm__ S *currentSrc1 = src1 + currentGmOffsetB;
    __gm__ T *currentDst = out + currentGmOffsetC;

    using NDValidShapeC = TileShape2D<T, baseM, baseN>;
    using NDWholeShapeC = BaseShape2D<T, M, N>;
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC>;

    using TileMatAData = Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatBData = Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;

    TileMatAData aMatTile[2];
    TileMatBData bMatTile[2];
    TASSIGN(aMatTile[0], 0x0); // L1 split to 4 buffer: pingA, pongA, pingB, pongB 
    TASSIGN(aMatTile[1], 0x0 + baseM * baseK * sizeof(U)); // pongA
    TASSIGN(bMatTile[0], 0x0 + baseM * baseK * 2 * sizeof(U)); // pingB
    TASSIGN(bMatTile[1], 0x0 + baseM * baseK * 2 * sizeof(U) + baseK * baseN * sizeof(U)); // pongB

    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    LeftTile aTile[2];
    RightTile bTile[2];
    ResTile cTile;
    TASSIGN(aTile[0], 0x0);         // L0A ping
    TASSIGN(aTile[1], 0x0 + baseM * baseK  * sizeof(U)); // L0A pong
    TASSIGN(bTile[0], 0x0);         // L0B ping
    TASSIGN(bTile[1], 0x0 + baseK * baseN  * sizeof(U)); // L0B pong
    TASSIGN(cTile, 0x0);

    constexpr uint32_t kLoop = singleCoreK / baseK;

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (uint32_t kIter = 0; kIter < kLoop; kIter++) {
        ProcessKIteration<T, U, S, M, K, N, baseM, baseK, baseN>(kIter, currentSrc0, currentSrc1, 
            aMatTile, bMatTile, aTile, bTile, cTile);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    GlobalDataOut dstGlobal(currentDst);
    TSTORE(dstGlobal, cTile);
}

template <typename T, uint32_t M, uint32_t K, uint32_t N, uint32_t singleCoreM, uint32_t singleCoreK,
    uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN>
__global__ AICORE void GemmBasic(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    runGEMMBASIC<float, half, half, M, K, N, singleCoreM, singleCoreK, singleCoreN, baseM,
        baseK, baseN>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0), reinterpret_cast<__gm__ half *>(src1));
}

template <typename T>
void launchGEMMBASIC(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    constexpr uint32_t blockDim = 24;
    constexpr uint32_t M = 512;
    constexpr uint32_t K = 2048;
    constexpr uint32_t N = 1536;
    constexpr uint32_t singleCoreM = 128;
    constexpr uint32_t singleCoreK = 2048;
    constexpr uint32_t singleCoreN = 256;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t baseN = 256;
    GemmBasic<T, M, K, N, singleCoreM, singleCoreK, singleCoreN, baseM, baseK,
        baseN><<<blockDim, nullptr, stream>>>(out, src0, src1);
}

template void launchGEMMBASIC<uint16_t>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
