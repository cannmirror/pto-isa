/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <common/pto_tileop.hpp>
#include <common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int row, int vaildRow, int srcCol, int srcVaildCol, int dstCol>
__aicore__ PTO_INLINE void runTRowSum(__gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynDim2Shape  = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(vaildRow, srcVaildCol), DynDim2Stride(row, srcCol));
    GlobalData dstGlobal(out, DynDim2Shape(vaildRow, dstCol), DynDim2Stride(row, dstCol));

    using srcTileData = Tile<Location::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<Location::Vec, T, row, 16, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(vaildRow, srcVaildCol);
    srcTileData tmpTile(vaildRow, srcVaildCol);
    dstTileData dstTile(vaildRow, dstCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x14000);
    TASSIGN(dstTile, 0x28000);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWSUM(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTROWSUMCase1(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWSUMCase2(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWSUMCase3(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWSUMCase4(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWSUMCase5(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWSUMCase6(__gm__ half *out, __gm__ half *src)
{
    runTRowSum<half, 256, 256, 16, 15, 1>(out, src);
}

template <uint32_t caseId>
void launchTROWSUMTestCase(void *out, void *src, aclrtStream stream) {
    switch (caseId) {
        case 1: {
            launchTROWSUMCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTROWSUMCase2<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTROWSUMCase3<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 4: {
            launchTROWSUMCase4<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 5: {
            launchTROWSUMCase5<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 6: {
            launchTROWSUMCase6<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWSUMTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<4>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<5>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<6>(void *out, void *src, aclrtStream stream);