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

template <int kRows, int kCols>
AICORE void runTSEL(__gm__ float __out__ *out, __gm__ uint8_t __in__ *mask, __gm__ float __in__ *src0,
                   __gm__ float __in__ *src1)
{
    using ShapeMat = Shape<1, 1, 1, kRows, kCols>;
    using StrideMat = Stride<1, 1, 1, kCols, 1>;
    using GlobalMat = GlobalTensor<float, ShapeMat, StrideMat>;

    constexpr int kMaskColsAligned = 32;
    constexpr int kMaskColsValid = (kCols + 7) / 8;
    using ShapeMask = Shape<1, 1, 1, kRows, kMaskColsValid>;
    using StrideMask = Stride<1, 1, 1, kMaskColsValid, 1>;
    using GlobalMask = GlobalTensor<uint8_t, ShapeMask, StrideMask>;

    using TileMat = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileMask = Tile<TileType::Vec, uint8_t, kRows, kMaskColsAligned, BLayout::RowMajor, -1, -1>;

    TileMat src0Tile(kRows, kCols);
    TileMat src1Tile(kRows, kCols);
    TileMat dstTile(kRows, kCols);
    TileMask maskTile(kRows, kMaskColsValid);

    GlobalMat src0Global(src0);
    GlobalMat src1Global(src1);
    GlobalMat dstGlobal(out);
    GlobalMask maskGlobal(mask);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(maskTile, maskGlobal);
    TSEL(dstTile, maskTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
void LaunchTSEL(float *out, uint8_t *mask, float *src0, float *src1, void *stream)
{
    (void)stream;
    runTSEL<kRows, kCols>(out, mask, src0, src1);
}

template void LaunchTSEL<2, 32>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
