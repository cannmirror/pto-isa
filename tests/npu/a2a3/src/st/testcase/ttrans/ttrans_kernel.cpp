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
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTTRANS(__gm__ T __out__ *out, __gm__ T __in__ *src, int vRows, int vCols) {
    using DynShapeSrc = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideSrc = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeSrc, DynStrideSrc>;

    using DynShapeDst = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideDst = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDst, DynStrideDst>;

    constexpr int kTCols_aligned = (kTCols_ * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int kTRows_aligned = (kTRows_ * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE / sizeof(T);
    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, kTCols_aligned, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, kTCols_, kTRows_aligned, BLayout::RowMajor, -1, -1>;

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

template <typename T, int tRows, int tCols, int vRows, int vCols>
void LaunchTTRANS(T *out, T *src, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> ){
        runTTRANS<half, tRows, tCols, vRows, vCols><<<1, nullptr, stream>>>((half*)(out), (half*)(src), vRows, vCols);
    }else {
        runTTRANS<T, tRows, tCols, vRows, vCols><<<1, nullptr, stream>>>(out, src, vRows, vCols);
    }
}

template void LaunchTTRANS<float, 16, 8, 16, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<aclFloat16, 16, 16, 16, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<uint8_t, 32, 32, 32, 32>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<float, 32, 16, 31, 15>(float *out, float *src, void *stream);
template void LaunchTTRANS<aclFloat16, 32, 32, 31, 31>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<uint8_t, 64, 64, 22, 63>(uint8_t *out, uint8_t *src, void *stream);