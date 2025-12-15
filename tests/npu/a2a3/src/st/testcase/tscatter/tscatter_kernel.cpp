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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template<typename T, typename I, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTScatter(__gm__ T *out, __gm__  T *src, __gm__ I *indexes) {
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using GlobalDataInd = GlobalTensor<I, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalDataInd indGlobal(indexes, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using indTileData = Tile<TileType::Vec, I, row, col, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    indTileData indTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(indTile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    TLOAD(dstTile, dstGlobal);

    TLOAD(srcTile, srcGlobal);

    TLOAD(indTile, indGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSCATTER(dstTile, srcTile, indTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTSCATTERCase1(__gm__ float *out, __gm__ float *src, __gm__ uint16_t *indexes)
{
    runTScatter<float, uint16_t, 32, 32, 64, 64>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, __gm__ uint16_t *indexes)
{
    runTScatter<half, uint16_t, 63, 63, 64, 64>((__gm__ half*)out, (__gm__ half*)src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase3(__gm__ int32_t *out, __gm__ int32_t *src, __gm__ uint16_t *indexes)
{
    runTScatter<int32_t, uint16_t, 31, 31, 128, 128>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase4(__gm__ int16_t *out, __gm__ int16_t *src, __gm__ uint16_t *indexes)
{
    runTScatter<int16_t, uint16_t, 15, 15, 192, 192>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase5(__gm__ float *out, __gm__ float *src, __gm__ uint16_t *indexes)
{
    runTScatter<float, uint16_t, 7, 7, 448, 448>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase6(__gm__ float *out, __gm__ float *src, __gm__ uint16_t *indexes)
{
    runTScatter<float, uint16_t, 256, 256, 16, 16>(out, src, indexes);
}

template <uint32_t caseId>
void launchTScatterTestCase(void *out, void *src, void *indexes, aclrtStream stream) {
    switch(caseId) {
        case 1: {
            launchTSCATTERCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, (uint16_t *)indexes);
            break;
        }
        case 2: {
            launchTSCATTERCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, (uint16_t *)indexes);
            break;
        }
        case 3: {
            launchTSCATTERCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, (uint16_t *)indexes);
            break;
        }
        case 4: {
            launchTSCATTERCase4<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, (uint16_t *)indexes);
            break;
        }
        case 5: {
            launchTSCATTERCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, (uint16_t *)indexes);
            break;
        }
        case 6: {
            launchTSCATTERCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, (uint16_t *)indexes);
            break;
        }
        default: {
        }
    }
}


template void launchTScatterTestCase<1>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<2>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<3>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<4>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<5>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<6>(void *out, void *src, void *indexes, aclrtStream stream);