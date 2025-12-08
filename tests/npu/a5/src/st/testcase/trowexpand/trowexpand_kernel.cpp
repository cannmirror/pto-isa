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
#include <pto/npu/a5/TMrgSort.hpp>
#include <pto/npu/a5/TLoad.hpp>
#include <pto/npu/a5/TStore.hpp>
#include <pto/npu/a5/TAssign.hpp>
#include <iostream>

using namespace std;
using namespace pto;


template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
__aicore__ inline void runROWEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src) {

    using DynShapeDim5 = Shape<1, 1, 1, rows, srcCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, srcCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, rows, srcCols, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, rows, dstCols>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCols, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, rows, dstCols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(rows, 1);
    DstTileData dstTile(rows, dstValidCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0xF000); // UB最大到0x40000

    int offset = 0;
    GlobalData srcGlobal(src + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);   // gm to ub
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPAND(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    // A5：ub to gm
    // copy_ubuf_to_gm_align_v2(dstGlobal.data(), dstTile.data(), 0 /*sid*/, SINGLE_ROW , SINGLE_COL * sizeof(T), 0 /*l2 cache ctrl*/, SINGLE_COL * sizeof(T), SINGLE_COL * sizeof(T));
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}


extern "C" __global__ __aicore__ void launchTROWEXPANDcase0(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runROWEXPAND<half, 16, 16, 512, 512> (
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTROWEXPANDcase1(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runROWEXPAND<int8_t, 16, 32, 256, 256> (
        reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTROWEXPANDcase2(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runROWEXPAND<float, 16, 8, 128, 128> (
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTROWEXPANDcase3(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runROWEXPAND<half, 16, 16, 511, 512> (
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTROWEXPANDcase4(__gm__ uint8_t *out, __gm__ uint8_t *src)
{

    runROWEXPAND<int8_t, 16, 32, 100, 256> (
        reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}
extern "C" __global__ __aicore__ void launchTROWEXPANDcase5(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runROWEXPAND<float, 16, 8, 127, 128> (
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}


template <int32_t tilingKey>
void launchTROWEXPAND(uint8_t *out, uint8_t *src, void* stream){
    cout << "launchTROWEXPAND start!" << endl;
    if constexpr (tilingKey == 0) {
        launchTROWEXPANDcase0<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 1) {
        launchTROWEXPANDcase1<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 2) {
        launchTROWEXPANDcase2<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 3) {
        launchTROWEXPANDcase3<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 4) {
        launchTROWEXPANDcase4<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == 5) {
        launchTROWEXPANDcase5<<<1, nullptr, stream>>>(out, src);
    }
    cout << "launchTROWEXPAND end!" << endl;
}

template void launchTROWEXPAND<0>(uint8_t *out, uint8_t *src, void *stream);  // 实例化 Key=0 的版本
template void launchTROWEXPAND<1>(uint8_t *out, uint8_t *src, void *stream);  // 实例化 Key=1 的版本
template void launchTROWEXPAND<2>(uint8_t *out, uint8_t *src, void *stream);  // 实例化 Key=2 的版本
template void launchTROWEXPAND<3>(uint8_t *out, uint8_t *src, void *stream);  // 实例化 Key=3 的版本
template void launchTROWEXPAND<4>(uint8_t *out, uint8_t *src, void *stream);  // 实例化 Key=4 的版本
template void launchTROWEXPAND<5>(uint8_t *out, uint8_t *src, void *stream);  // 实例化 Key=5 的版本
