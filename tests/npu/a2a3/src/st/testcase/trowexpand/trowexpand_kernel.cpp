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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int rows, int src_col, int src_validCol, int dst_col, int dst_validCol>
__global__ AICORE void runTROWEXPAND( __gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData srcGlobal(src, DynShapeDim5(rows, src_validCol), DynStridDim5(rows, src_col));
    GlobalData dstGlobal(out, DynShapeDim5(rows, dst_validCol), DynStridDim5(rows, dst_col));
    using TileDataSrc = Tile<TileType::Vec, T, rows, src_col, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, rows, dst_validCol, BLayout::RowMajor, -1, -1>;


    TileDataSrc srcTile(rows, src_validCol);
    TileDataDst dstTile(rows, dst_validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x8000); // UB最大到0x40000


    TLOAD(srcTile, srcGlobal);   // gm to ub
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPAND(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}


template <typename T, int rows, int src_col, int src_validCol, int dst_col, int dst_validCol>
void launchTROWEXPAND(T *out, T *src, void* stream)
{
    cout << "launchTROWEXPAND start!" << endl;

    runTROWEXPAND<T, rows, src_col, src_validCol, dst_col, dst_validCol><<<1, nullptr, stream>>>(out, src);

    cout << "launchTROWEXPAND end!" << endl;
}

template void launchTROWEXPAND<uint16_t, 16, 16, 16, 512, 512>(uint16_t *out, uint16_t *src, void* stream);
template void launchTROWEXPAND<uint8_t, 16, 32, 32, 256, 256>(uint8_t *out, uint8_t *src, void* stream);
template void launchTROWEXPAND<uint32_t, 16, 8, 8, 128, 128>(uint32_t *out, uint32_t *src, void* stream);
template void launchTROWEXPAND<float, 16, 32, 32, 512, 512>(float *out, float *src, void* stream);
template void launchTROWEXPAND<uint16_t, 16, 16, 1, 255, 256>(uint16_t *out, uint16_t *src, void* stream);
