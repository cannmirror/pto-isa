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
AICORE void runTCMPS(__gm__ T __out__ *out, __gm__ T __in__ *src0, T scalar, pto::CmpMode mode)
{
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStridDim5 = Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using TileData = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kRows, kCols);
    TileData dstTile(kRows, kCols);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TCMPS(dstTile, src0Tile, scalar, mode);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kRows, int kCols>
void LaunchTCMPS(T *out, T *src0, T scalar, pto::CmpMode mode, void *stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCMPS<half, kRows, kCols>((half *)(out), (half *)(src0), static_cast<half>(scalar), mode);
    } else {
        runTCMPS<T, kRows, kCols>(out, src0, scalar, mode);
    }
}

template void LaunchTCMPS<float, 64, 64>(float *out, float *src0, float scalar, pto::CmpMode mode, void *stream);

