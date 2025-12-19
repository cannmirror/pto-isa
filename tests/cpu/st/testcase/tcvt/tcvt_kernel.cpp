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

template <typename SrcT, typename DstT, int kRows, int kCols>
AICORE void runTCVT(__gm__ DstT __out__ *out, __gm__ SrcT __in__ *src, RoundMode mode)
{
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStridDim5Src = Stride<1, 1, 1, kCols, 1>;
    using DynStridDim5Dst = Stride<1, 1, 1, kCols, 1>;
    using GlobalSrc = GlobalTensor<SrcT, DynShapeDim5, DynStridDim5Src>;
    using GlobalDst = GlobalTensor<DstT, DynShapeDim5, DynStridDim5Dst>;

    using TileSrc = Tile<TileType::Vec, SrcT, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileDst = Tile<TileType::Vec, DstT, kRows, kCols, BLayout::RowMajor, -1, -1>;

    TileSrc srcTile(kRows, kCols);
    TileDst dstTile(kRows, kCols);

    GlobalSrc srcGlobal(src);
    GlobalDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TCVT(dstTile, srcTile, mode);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols>
void LaunchTCVT(int32_t *out, float *src, RoundMode mode, void *stream)
{
    (void)stream;
    runTCVT<float, int32_t, kRows, kCols>(out, src, mode);
}

template void LaunchTCVT<64, 64>(int32_t *out, float *src, RoundMode mode, void *stream);

