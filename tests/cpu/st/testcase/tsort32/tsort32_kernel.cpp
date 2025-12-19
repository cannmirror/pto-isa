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

using namespace pto;

template <int kCols>
AICORE void runTSORT32(__gm__ float __out__ *outVal, __gm__ uint32_t __out__ *outIdx, __gm__ float __in__ *src)
{
    using ShapeT = Shape<1, 1, 1, 1, kCols>;
    using StrideT = Stride<1, 1, 1, kCols, 1>;
    using GlobalF = GlobalTensor<float, ShapeT, StrideT>;
    using GlobalI = GlobalTensor<uint32_t, ShapeT, StrideT>;

    using TileF = Tile<TileType::Vec, float, 1, kCols, BLayout::RowMajor, -1, -1>;
    using TileI = Tile<TileType::Vec, uint32_t, 1, kCols, BLayout::RowMajor, -1, -1>;
    TileF srcTile(1, kCols);
    TileF dstTile(1, kCols);
    TileI idxTile(1, kCols);

    GlobalF srcGlobal(src);
    GlobalF dstGlobal(outVal);
    GlobalI idxGlobal(outIdx);

    TLOAD(srcTile, srcGlobal);
    TSORT32(dstTile, srcTile, idxTile);
    TSTORE(dstGlobal, dstTile);
    TSTORE(idxGlobal, idxTile);
    outVal = dstGlobal.data();
    outIdx = idxGlobal.data();
}

template <int kCols>
void LaunchTSORT32(float *outVal, uint32_t *outIdx, float *src, void *stream)
{
    (void)stream;
    runTSORT32<kCols>(outVal, outIdx, src);
}

template void LaunchTSORT32<32>(float *outVal, uint32_t *outIdx, float *src, void *stream);

