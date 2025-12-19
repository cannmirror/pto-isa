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

template <typename T, int kRows, int kCols>
AICORE void runTCOLSUM(__gm__ T __out__ *out, __gm__ T __in__ *src, bool isBinary)
{
    using ShapeSrc = Shape<1, 1, 1, kRows, kCols>;
    using StrideSrc = Stride<1, 1, 1, kCols, 1>;
    using GlobalSrc = GlobalTensor<T, ShapeSrc, StrideSrc>;

    using ShapeDst = Shape<1, 1, 1, 1, kCols>;
    using StrideDst = Stride<1, 1, 1, kCols, 1>;
    using GlobalDst = GlobalTensor<T, ShapeDst, StrideDst>;

    using SrcTileT = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using DstTileT = Tile<TileType::Vec, T, 1, kCols, BLayout::RowMajor, -1, -1>;
    using TmpTileT = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;

    SrcTileT srcTile(kRows, kCols);
    DstTileT dstTile(1, kCols);
    TmpTileT tmpTile(kRows, kCols);

    GlobalSrc srcGlobal(src);
    GlobalDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TCOLSUM(dstTile, srcTile, tmpTile, isBinary);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kRows, int kCols>
void LaunchTCOLSUM(T *out, T *src, bool isBinary, void *stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLSUM<half, kRows, kCols>((half *)(out), (half *)(src), isBinary);
    } else {
        runTCOLSUM<T, kRows, kCols>(out, src, isBinary);
    }
}

template void LaunchTCOLSUM<float, 64, 64>(float *out, float *src, bool isBinary, void *stream);
