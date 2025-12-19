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

using namespace pto;

template <typename T, int kGRowsD_, int kGColsD_, int kGRowsS0_, int kGColsS0_, int kGRowsS1_, int kGColsS1_,
          int kTRowsD_, int kTColsD_, int kTRowsS0_, int kTColsS0_, int kTRowsS1_, int kTColsS1_, int DstValidRow,
          int DstValidCol, int Src0ValidRow, int Src0ValidCol, int Src1ValidRow, int Src1ValidCol>
__global__ AICORE void runTPartAdd( __gm__ T __out__ *out, __gm__ T __in__ *src0,  __gm__ T __in__ *src1)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData src0Global(src0, DynDim2Shape(kGRowsS0_, kGColsS0_), DynDim2Stride(kGRowsS0_, kGColsS0_));
    GlobalData src1Global(src1, DynDim2Shape(kGRowsS1_, kGColsS1_), DynDim2Stride(kGRowsS1_, kGColsS1_));
    GlobalData dstGlobal(out, DynDim2Shape(kGRowsD_, kGColsD_), DynDim2Stride(kGRowsD_, kGColsD_));

    using TileDataSrc0 = Tile<TileType::Vec, T, kTRowsS0_, kTColsS0_, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, kTRowsS1_, kTColsS1_, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, kTRowsD_, kTColsD_, BLayout::RowMajor, -1, -1>;
    TileDataSrc0 src0Tile(Src0ValidRow, Src0ValidCol);
    TileDataSrc1 src1Tile(Src1ValidRow, Src1ValidCol);
    TileDataDst dstTile(DstValidRow, DstValidCol);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, kTRowsS0_ * kTColsS0_ * sizeof(T));
    TASSIGN(dstTile, kTRowsS0_ * kTColsS0_ * sizeof(T) + kTRowsS1_ * kTColsS1_ * sizeof(T));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TPARTADD(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRowsD_, int kGColsD_, int kGRowsS0_, int kGColsS0_, int kGRowsS1_, int kGColsS1_,
          int kTRowsD_, int kTColsD_, int kTRowsS0_, int kTColsS0_, int kTRowsS1_, int kTColsS1_, int DstValidRow,
          int DstValidCol, int Src0ValidRow, int Src0ValidCol, int Src1ValidRow, int Src1ValidCol>
void LaunchTPartAdd(T *out, T *src0, T *src1, aclrtStream stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> ) {
        runTPartAdd<half, kGRowsD_, kGColsD_, kGRowsS0_, kGColsS0_, kGRowsS1_, kGColsS1_, kTRowsD_, kTColsD_, kTRowsS0_,
                    kTColsS0_, kTRowsS1_, kTColsS1_, DstValidRow, DstValidCol, Src0ValidRow, Src0ValidCol, Src1ValidRow,
                    Src1ValidCol>((half*)(out), (half*)(src0), (half*)(src1));
    } else { 
        runTPartAdd<T, kGRowsD_, kGColsD_, kGRowsS0_, kGColsS0_, kGRowsS1_, kGColsS1_, kTRowsD_, kTColsD_, kTRowsS0_,
                    kTColsS0_, kTRowsS1_, kTColsS1_, DstValidRow, DstValidCol, Src0ValidRow, Src0ValidCol, Src1ValidRow,
                    Src1ValidCol>(out, src0, src1);
    }
}

template void LaunchTPartAdd<int16_t, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32>
    (int16_t *out, int16_t *src0, int16_t *src1, aclrtStream stream);
template void LaunchTPartAdd<aclFloat16, 22, 32, 22, 32, 16, 32, 22, 32, 22, 32, 16, 32, 22, 32, 22, 32, 16, 32>
    (aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, aclrtStream stream);
template void LaunchTPartAdd<float, 22, 40, 22, 40, 22, 32, 22, 40, 22, 40, 22, 32, 22, 40, 22, 40, 22, 32>
    (float *out, float *src0, float *src1, aclrtStream stream);
template void LaunchTPartAdd<int32_t, 22, 40, 22, 40, 8, 40, 22, 40, 22, 40, 8, 40, 22, 40, 22, 40, 8, 40>
    (int32_t *out, int32_t *src0, int32_t *src1, aclrtStream stream);
template void LaunchTPartAdd<float, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128>
    (float *out, float *src0, float *src1, aclrtStream stream);
template void LaunchTPartAdd<int16_t, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32, 16, 32, 16, 0, 16, 32>
    (int16_t *out, int16_t *src0, int16_t *src1, aclrtStream stream);
template void LaunchTPartAdd<aclFloat16, 16, 32, 16, 32, 16, 32, 16, 32, 8, 32, 16, 32, 16, 32, 0, 32, 16, 32>
    (aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, aclrtStream stream);
template void LaunchTPartAdd<float, 16, 32, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32, 16, 32, 16, 0>
    (float *out, float *src0, float *src1, aclrtStream stream);
template void LaunchTPartAdd<int32_t, 16, 32, 16, 32, 16, 32, 16, 32, 16, 32, 8, 32, 16, 32, 16, 32, 0, 32>
    (int32_t *out, int32_t *src0, int32_t *src1, aclrtStream stream);

