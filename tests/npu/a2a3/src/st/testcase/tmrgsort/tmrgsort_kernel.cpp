/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/tile_tensor_impl.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <pto/npu/a2a3/TMrgSort.hpp>
#include <pto/npu/a2a3/TLoad.hpp>
#include <pto/npu/a2a3/TStore.hpp>
#include <pto/npu/a2a3/TAssign.hpp>
#include <iostream>

using namespace std;
using namespace pto;

#define EXHAUSTED 1

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
__global__ __aicore__ void runTMrgsort(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2, __gm__ T* src3)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_ * LISTNUM>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_ * LISTNUM, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using TmpGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<Location::Vec, T, 1, kTCols_ * LISTNUM, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(1, kTCols_);
    TileData src1Tile(1, kTCols_src1);
    TileData src2Tile(1, kTCols_src2);
    TileData src3Tile(1, kTCols_src3);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols_*LISTNUM);
    uint32_t src1Addr = 1 * kTCols_ * sizeof(T);
    uint32_t src2Addr = src1Addr + kTCols_src1 * sizeof(T);
    uint32_t src3Addr = src2Addr + kTCols_src2 * sizeof(T);
    uint32_t dstAddr = src3Addr + kTCols_src3 * sizeof(T);
    uint32_t tmpAddr = dstAddr + TOPK * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + src1Addr);
    TASSIGN(src2Tile, 0x0 + src2Addr);
    TASSIGN(src3Tile, 0x0 + src3Addr);
    TASSIGN(dstTile, 0x0 + dstAddr);
    TASSIGN(tmpTile, 0x0 + tmpAddr);

    for (unsigned row = 0; row < kTRows_; row ++) {
        int dataOffset = 0; // 0 * 128
        GlobalData src0Global(src0 + dataOffset);
        GlobalData src1Global(src1 + dataOffset);
        GlobalData src2Global(src2 + dataOffset);
        GlobalData src3Global(src3 + dataOffset);
        DstGlobalData dstGlobal(out + dataOffset*LISTNUM);

        // TSTORE(dstGlobal, src0Tile)
        MrgSortExecutedNumList executedNumList;
        // 4
        if constexpr (LISTNUM == 4) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            TLOAD(src2Tile, src2Global);
            TLOAD(src3Tile, src3Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, 0>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
        }
        // 3
        if constexpr (LISTNUM == 3) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            TLOAD(src2Tile, src2Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, 0>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
        }
        // 2
        if constexpr (LISTNUM == 2) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, 0>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, dstTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
__global__ __aicore__ void runTMrgsortExhausted(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* src2,
                                                __gm__ T* src3)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_ * LISTNUM>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_ * LISTNUM, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using TmpGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<Location::Vec, T, 1, kTCols_ * LISTNUM, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(1, kTCols_);
    TileData src1Tile(1, kTCols_src1);
    TileData src2Tile(1, kTCols_src2);
    TileData src3Tile(1, kTCols_src3);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols_*LISTNUM);
    uint32_t src1Addr = 1 * kTCols_ * sizeof(T);
    uint32_t src2Addr = src1Addr + kTCols_src1 * sizeof(T);
    uint32_t src3Addr = src2Addr + kTCols_src2 * sizeof(T);
    uint32_t dstAddr = src3Addr + kTCols_src3 * sizeof(T);
    uint32_t tmpAddr = dstAddr + TOPK * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + src1Addr);
    TASSIGN(src2Tile, 0x0 + src2Addr);
    TASSIGN(src3Tile, 0x0 + src3Addr);
    TASSIGN(dstTile, 0x0 + dstAddr);
    TASSIGN(tmpTile, 0x0 + tmpAddr);

    int offset = 0; // (block_idx / 4) * （64 * 16) + (block_idx % 4) * 16;
    for (unsigned row = 0; row < kTRows_; row ++) {
        // ############################ round1
        int dataOffset = row * kGCols_;
        GlobalData src00Global(src0 + dataOffset);
        GlobalData src10Global(src1 + dataOffset);
        GlobalData src20Global(src2 + dataOffset);
        GlobalData src30Global(src3 + dataOffset);
        DstGlobalData dst0Global(out + dataOffset*LISTNUM);

        MrgSortExecutedNumList executedNumList;
        // 4
        if constexpr (LISTNUM == 4) {
            TLOAD(src0Tile, src00Global);
            TLOAD(src1Tile, src10Global);
            TLOAD(src2Tile, src20Global);
            TLOAD(src3Tile, src30Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
        }
        // 3
        if constexpr (LISTNUM == 3) {
            TLOAD(src0Tile, src00Global);
            TLOAD(src1Tile, src10Global);
            TLOAD(src2Tile, src20Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
        }
        // 2
        if constexpr (LISTNUM == 2) {
            TLOAD(src0Tile, src00Global);
            TLOAD(src1Tile, src10Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dst0Global, dstTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
__global__ __aicore__ void runTMrgsort_single(__gm__ T *out, __gm__ T *src0) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(1, kTCols_);
    DstTileData dstTile(1, kTCols_);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0 + kTCols_ * sizeof(T));

    int offset = 0;
    GlobalData src0Global(src0 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMRGSORT<DstTileData, TileData>(dstTile, src0Tile, blockLen);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}

template <int kTCols_>
__aicore__ PTO_INLINE int32_t fillMrgArray(int32_t* mrgArray, int blockLen) {
    int32_t arrayCount = 0;
    int32_t tmpInner = kTCols_;
    for (int32_t i = blockLen; i >= 64; i /= 4) {
        int32_t count;
        for (count = 0; count < tmpInner / i; count++) {
            mrgArray[arrayCount++] = i;
        }
        tmpInner -= count * i;
    }
    return arrayCount;
}

template <typename GlobalData, typename DstGlobalData, typename DstTileData, typename TmpTileData, typename T,
    int kTCols_, int topk>
__aicore__ PTO_INLINE void sortTailBlock(
    DstGlobalData &dstGlobal, DstTileData &dstTile, __gm__ T *src, __ubuf__ T *srcAddr, int blockLen)
{
    TmpTileData tmp1Tile(1, kTCols_);
    TASSIGN(tmp1Tile, 0x0 + (kTCols_ * 2 + topk) * sizeof(T));
    
    int32_t mrgArray[15] = {0};
    int32_t arrayCount = fillMrgArray<kTCols_>(mrgArray, blockLen);
    uint16_t mrgSortedLen = 0;
    GlobalData srcGlobal(src);
    MrgSortExecutedNumList executedNumList;
    for (int32_t i = 0; i < arrayCount - 1; ++i) {
        using Src0TileData = Tile<Location::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
        using Src1TileData = Tile<Location::Vec, T, 1, topk, BLayout::RowMajor, -1, -1>;
        mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
        uint64_t tmpMrgSortedLen = mrgSortedLen;
        uint64_t tmpMrgArray = mrgArray[i + 1];
        if (tmpMrgSortedLen > topk) {
            tmpMrgSortedLen = topk;
        }
        if (tmpMrgArray > topk) {
            tmpMrgArray = topk;
        }
        Src0TileData src0Tile(1, tmpMrgSortedLen);
        Src1TileData src1Tile(1, tmpMrgArray);
        TASSIGN(src0Tile, 0x0 + (kTCols_ * 3 + topk) * sizeof(T));
        TASSIGN(src1Tile, 0x0 + (kTCols_ * 3 + topk + tmpMrgSortedLen) * sizeof(T));
        copy_ubuf_to_ubuf(src0Tile.data(), (__ubuf__ void *)srcAddr, 0, 1,
            (tmpMrgSortedLen * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE, 0, 0);
        pipe_barrier(PIPE_V);
        copy_ubuf_to_ubuf(src1Tile.data(), (__ubuf__ void *)(srcAddr + mrgSortedLen), 0, 1,
            (tmpMrgArray * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE, 0, 0);
        pipe_barrier(PIPE_V);
        TMRGSORT<DstTileData, TmpTileData, Src0TileData, Src1TileData, 0>(
            dstTile, executedNumList, tmp1Tile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
        copy_ubuf_to_ubuf((__ubuf__ void *)srcAddr, dstTile.data(), 0, 1,
            (topk * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE, 0, 0);
        pipe_barrier(PIPE_V);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
__global__ __aicore__ void runTMrgsort_topk(__gm__ T *out, __gm__ T *src)
{
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstGlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, topk, 1>>;
    using TmpGlobalData = GlobalTensor<T, Shape<1, 1, 1, kGRows_, kGCols_>, pto::Stride<1, 1, 1, kGCols_, 1>>;
    using DstTileData = Tile<Location::Vec, T, kTRows_, topk, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<Location::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(1, kTCols_);
    DstTileData dstTile(1, topk);
    TmpTileData tmpTile(1, kTCols_);
    uint32_t dstAddr = kTCols_ * sizeof(T);
    uint32_t tmpAddr = dstAddr + topk * sizeof(T);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + dstAddr);
    TASSIGN(tmpTile, 0x0 + tmpAddr);

    GlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    uint32_t blockLen = 64;

    // 每4个合并，计算整块
    TLOAD(srcTile, srcGlobal);
    for (; blockLen * 4 <= kTCols_; blockLen *= 4) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMRGSORT<TmpTileData, TileData>(tmpTile, srcTile, blockLen);
        pipe_barrier(PIPE_V);
        uint16_t cols = kTCols_ / (blockLen * 4) * (blockLen * 4);
        copy_ubuf_to_ubuf(
            srcTile.data(), tmpTile.data(), 0, 1, (cols * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE, 0, 0);
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    // 合并尾块
    if (blockLen < kTCols_) {
        sortTailBlock<GlobalData, DstGlobalData, DstTileData, TmpTileData, T, kTCols_, topk>(
            dstGlobal, dstTile, src, srcTile.data(), blockLen);
    } else {
        copy_ubuf_to_ubuf(
            dstTile.data(), tmpTile.data(), 0, 1, (topk * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE, 0, 0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, dstTile);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
void LanchTMrgsortMulti(float* out, float* src0, float* src1, float* src2, float* src3, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        runTMrgsort<half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, kTCols_src1 * TYPE_COEF,
                    kTCols_src2 * TYPE_COEF, kTCols_src3 * TYPE_COEF, TOPK * TYPE_COEF, LISTNUM>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0),
                                     reinterpret_cast<half*>(src1), reinterpret_cast<half*>(src2),
                                     reinterpret_cast<half*>(src3));
    } else {
        runTMrgsort<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM>
            <<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM>
void LanchTMrgsortExhausted(float* out, float* src0, float* src1, float* src2, float* src3, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        runTMrgsortExhausted<half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, kTCols_src1 * TYPE_COEF,
                             kTCols_src2 * TYPE_COEF, kTCols_src3 * TYPE_COEF, TOPK * TYPE_COEF, LISTNUM>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0),
                                     reinterpret_cast<half*>(src1), reinterpret_cast<half*>(src2),
                                     reinterpret_cast<half*>(src3));
    } else {
        runTMrgsortExhausted<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK,
                             LISTNUM><<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
void LanchTMrgsortSingle(float* out, float* src, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        runTMrgsort_single<half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, blockLen * TYPE_COEF>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src));
    } else {
        runTMrgsort_single<T, kGRows_, kGCols_, kTRows_, kTCols_, blockLen><<<1, nullptr, stream>>>(out, src);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
void LanchTMrgsortTopK(float* out, float* src, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
        runTMrgsort_topk<half, kGRows_, kGCols_ * TYPE_COEF, kTRows_, kTCols_ * TYPE_COEF, topk * TYPE_COEF>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src));
    } else {
        runTMrgsort_topk<T, kGRows_, kGCols_, kTRows_, kTCols_, topk><<<1, nullptr, stream>>>(out, src);
    }
}

// multi case
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 128, 512, 4>(float* out, float* src0, float* src1,
                                                                               float* src2, float* src3, void* stream);
template void LanchTMrgsortMulti<uint16_t, 1, 128, 1, 128, 128, 128, 128, 512, 4>(float* out, float* src0, float* src1,
                                                                                  float* src2, float* src3,
                                                                                  void* stream);
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 64, 448, 4>(float* out, float* src0, float* src1,
                                                                              float* src2, float* src3, void* stream);
template void LanchTMrgsortMulti<float, 1, 128, 1, 128, 128, 64, 0, 128, 3>(float* out, float* src0, float* src1,
                                                                            float* src2, float* src3, void* stream);
// multi exhausted case
// 上板时，耗尽模式在排序停止时，实际排序长度之后的位置内存值不保证有效性
template void LanchTMrgsortExhausted<float, 1, 64, 1, 64, 64, 0, 0, 128, 2>(float* out, float* src0, float* src1,
                                                                            float* src2, float* src3, void* stream);
template void LanchTMrgsortExhausted<uint16_t, 1, 256, 1, 256, 256, 256, 0, 768, 3>(float* out, float* src0,
                                                                                    float* src1, float* src2,
                                                                                    float* src3, void* stream);
// single case
template void LanchTMrgsortSingle<float, 1, 256, 1, 256, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<float, 1, 320, 1, 256, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<float, 1, 512, 1, 512, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<float, 1, 640, 1, 512, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 256, 1, 256, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 320, 1, 256, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 512, 1, 512, 64>(float* out, float* src, void* stream);
template void LanchTMrgsortSingle<uint16_t, 1, 1024, 1, 1024, 256>(float* out, float* src, void* stream);

// topk case
template void LanchTMrgsortTopK<float, 1, 2048, 1, 2048, 1024>(float* out, float* src, void* stream);
template void LanchTMrgsortTopK<float, 1, 2048, 1, 2048, 2048>(float* out, float* src, void* stream);
template void LanchTMrgsortTopK<float, 1, 1280, 1, 1280, 512>(float* out, float* src, void* stream);
template void LanchTMrgsortTopK<uint16_t, 1, 2048, 1, 2048, 1024>(float* out, float* src, void* stream);
template void LanchTMrgsortTopK<uint16_t, 1, 2048, 1, 2048, 2048>(float* out, float* src, void* stream);
template void LanchTMrgsortTopK<uint16_t, 1, 1280, 1, 1280, 512>(float* out, float* src, void* stream);