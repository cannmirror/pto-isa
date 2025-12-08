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

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ PTO_INLINE void runTTRANS(__gm__ T __out__ *out, __gm__ T __in__ *src, int vRows, int vCols) {
    using DynShapeSrc = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideSrc = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeSrc, DynStrideSrc>;

    using DynShapeDst = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideDst = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDst, DynStrideDst>;

    constexpr int kTCols_aligned = (kTCols_ * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int kTRows_aligned = (kTRows_ * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE / sizeof(T);
    using TileDataSrc = Tile<Location::Vec, T, kTRows_, kTCols_aligned, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<Location::Vec, T, kTCols_, kTRows_aligned, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(vCols, vRows);

    constexpr uint32_t alignedSrcTileSize = (kTRows_ * kTCols_aligned * sizeof(T) + 0x1FF) / 0x200 * 0x200;
    constexpr uint32_t alignedDstTileSize = (kTCols_ * kTRows_aligned * sizeof(T) + 0x1FF) / 0x200 * 0x200;
    static_assert(alignedSrcTileSize + alignedDstTileSize <= TMP_UB_OFFSET);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, alignedSrcTileSize);

    GlobalDataSrc srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, kGCols_, kGRows_));
    GlobalDataDst dstGlobal(out, pto::Shape(1, 1, 1, vCols, vRows), pto::Stride(1, 1, 1, kGRows_, kGCols_));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TTRANS(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ __aicore__ void launchTTRANS_1(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    constexpr int KG_ROWS    = 16;
    constexpr int KG_COLS    = 8;
    constexpr int KT_ROWS    = 16;
    constexpr int KT_COLS    = 8;
    constexpr int VALID_ROWS = 16;
    constexpr int VALID_COLS = 8;
    typedef float IN_DTYPE;

    runTTRANS<IN_DTYPE, KG_ROWS, KG_COLS, KT_ROWS, KT_COLS>(reinterpret_cast<__gm__ IN_DTYPE*>(out),
                                                            reinterpret_cast<__gm__ IN_DTYPE*>(src), 
                                                            VALID_ROWS, VALID_COLS);
}

extern "C" __global__ __aicore__ void launchTTRANS_2(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    constexpr int KG_ROWS    = 16;
    constexpr int KG_COLS    = 16;
    constexpr int KT_ROWS    = 16;
    constexpr int KT_COLS    = 16;
    constexpr int VALID_ROWS = 16;
    constexpr int VALID_COLS = 16;

    typedef half IN_DTYPE;
    runTTRANS<IN_DTYPE, KG_ROWS, KG_COLS, KT_ROWS, KT_COLS>(reinterpret_cast<__gm__ IN_DTYPE*>(out),
                                                            reinterpret_cast<__gm__ IN_DTYPE*>(src), 
                                                            VALID_ROWS, VALID_COLS);
}

extern "C" __global__ __aicore__ void launchTTRANS_3(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    constexpr int KG_ROWS    = 32;
    constexpr int KG_COLS    = 32;
    constexpr int KT_ROWS    = 32;
    constexpr int KT_COLS    = 32;
    constexpr int VALID_ROWS = 32;
    constexpr int VALID_COLS = 32;
    typedef int8_t IN_DTYPE;
    runTTRANS<IN_DTYPE, KG_ROWS, KG_COLS, KT_ROWS, KT_COLS>(reinterpret_cast<__gm__ IN_DTYPE*>(out),
                                                            reinterpret_cast<__gm__ IN_DTYPE*>(src), 
                                                            VALID_ROWS, VALID_COLS);
}

extern "C" __global__ __aicore__ void launchTTRANS_11(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    constexpr int KG_ROWS    = 32;
    constexpr int KG_COLS    = 16;
    constexpr int KT_ROWS    = 32;
    constexpr int KT_COLS    = 16;
    constexpr int VALID_ROWS = 31;
    constexpr int VALID_COLS = 15;

    typedef float IN_DTYPE;
    runTTRANS<IN_DTYPE, KG_ROWS, KG_COLS, KT_ROWS, KT_COLS>(reinterpret_cast<__gm__ IN_DTYPE*>(out),
                                                            reinterpret_cast<__gm__ IN_DTYPE*>(src), 
                                                            VALID_ROWS, VALID_COLS);
}

extern "C" __global__ __aicore__ void launchTTRANS_12(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    constexpr int KG_ROWS    = 32;
    constexpr int KG_COLS    = 32;
    constexpr int KT_ROWS    = 32;
    constexpr int KT_COLS    = 33;
    constexpr int VALID_ROWS = 31;
    constexpr int VALID_COLS = 31;
    typedef half IN_DTYPE;
    runTTRANS<IN_DTYPE, KG_ROWS, KG_COLS, KT_ROWS, KT_COLS>(reinterpret_cast<__gm__ IN_DTYPE*>(out),
                                                            reinterpret_cast<__gm__ IN_DTYPE*>(src), 
                                                            VALID_ROWS, VALID_COLS);
}

extern "C" __global__ __aicore__ void launchTTRANS_13(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    constexpr int KG_ROWS    = 64;
    constexpr int KG_COLS    = 64;
    constexpr int KT_ROWS    = 64;
    constexpr int KT_COLS    = 64;
    constexpr int VALID_ROWS = 22;
    constexpr int VALID_COLS = 63;
    typedef int8_t IN_DTYPE;
    runTTRANS<IN_DTYPE, KG_ROWS, KG_COLS, KT_ROWS, KT_COLS>(reinterpret_cast<__gm__ IN_DTYPE*>(out),
                                                            reinterpret_cast<__gm__ IN_DTYPE*>(src), 
                                                            VALID_ROWS, VALID_COLS);
}

template <int32_t tilingKey>
void launchTTRANS_demo(uint8_t *out, uint8_t *src, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTTRANS_1<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 2) {
        launchTTRANS_2<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 3) {
        launchTTRANS_3<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 11) {
        launchTTRANS_11<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 12) {
        launchTTRANS_12<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 13) {
        launchTTRANS_13<<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTTRANS_demo<1>(uint8_t *out, uint8_t *src, void *stream);
template void launchTTRANS_demo<2>(uint8_t *out, uint8_t *src, void *stream);
template void launchTTRANS_demo<3>(uint8_t *out, uint8_t *src, void *stream);
template void launchTTRANS_demo<11>(uint8_t *out, uint8_t *src, void *stream);
template void launchTTRANS_demo<12>(uint8_t *out, uint8_t *src, void *stream);
template void launchTTRANS_demo<13>(uint8_t *out, uint8_t *src, void *stream);
