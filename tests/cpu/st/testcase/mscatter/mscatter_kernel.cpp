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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runMSCATTER(__gm__ T __in__ *src0, __gm__ T __out__ *out, __gm__ T __in__ *src1) {
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    
    using srcTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    
    srcTileData src0Tile(kTRows_, kTCols_);
    srcTileData src1Tile(kTRows_, kTCols_);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);

    std::fill(out, out + kTRows_*kTCols_, 0);

    GlobalData src0Global(src0, DynShapeDim5(kTRows_, kTCols_), 
            DynStridDim5(kTRows_, kTCols_));
    GlobalData src1Global(src1, DynShapeDim5(kTRows_, kTCols_), 
            DynStridDim5(kTRows_, kTCols_));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    MSCATTER(src0Tile, out, src1Tile); 
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchMSCATTER(T *src0, T *out, T *src1, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runMSCATTER<half, kGRows_, kGCols_, kTRows_, kTCols_>((half*)(src0), (half*)(out), (half*)(src1));
    else 
        runMSCATTER<T, kGRows_, kGCols_, kTRows_, kTCols_>(src0, out, src1);
}

template void LaunchMSCATTER<float, 64, 64, 64, 64>(float *src0, float *out, float *src1, void *stream);
template void LaunchMSCATTER<float, 16, 256, 16, 256>(float *src0, float *out, float *src1, void *stream);
