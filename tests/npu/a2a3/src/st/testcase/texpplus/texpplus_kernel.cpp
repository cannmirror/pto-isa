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

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol, bool isInPlace = false>
__global__ AICORE void runTExp( __gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(srcTileRow, srcTileCol));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));

    using srcTileData = Tile<TileType::Vec, T, srcTileRow, srcTileCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);
    if constexpr(isInPlace) {
        TASSIGN(dstTile, 0x0);
    } else {
        TASSIGN(dstTile, 0x20000);
        TLOAD(dstTile, dstGlobal);
    }

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TEXP(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool isInPlace = false>
void LaunchTExp(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTExp<half, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, isInPlace>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src));
    else
        runTExp<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, isInPlace>
            <<<1, nullptr, stream>>>(out, src);
}

template void LaunchTExp<float, 64, 64, 64, 128, 64, 64, true>(float *out, float *src, void *stream);
template void LaunchTExp<float, 128, 64, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTExp<aclFloat16, 64, 64, 128, 128, 64, 64, true>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTExp<aclFloat16, 64, 256, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);