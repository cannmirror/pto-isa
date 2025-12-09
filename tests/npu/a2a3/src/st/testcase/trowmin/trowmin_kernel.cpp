/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace pto;

template <typename T, int row, int vaildRow, int srcCol, int srcVaildCol, int dstCol>
PTO_INTERNAL void runTRowMin(__gm__ T *out, __gm__ T *src) {
  using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
  using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
  using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
  GlobalData srcGlobal(src, DynDim2Shape(vaildRow, srcVaildCol), DynDim2Stride(row, srcCol));
  GlobalData dstGlobal(out, DynDim2Shape(vaildRow, dstCol), DynDim2Stride(row, dstCol));

  using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
  using dstTileData = Tile<TileType::Vec, T, row, 16, BLayout::RowMajor, -1, -1>;
  srcTileData srcTile(vaildRow, srcVaildCol);
  srcTileData tmpTile(vaildRow, srcVaildCol);
  dstTileData dstTile(vaildRow, dstCol);
  TASSIGN(srcTile, 0x0);
  TASSIGN(dstTile, sizeof(T) * row * srcCol);
  TASSIGN(tmpTile, sizeof(T) * row * (srcCol + 16));

  // 搬运数据
  TLOAD(srcTile, srcGlobal);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWMIN(dstTile, srcTile, tmpTile);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTROWMINCase1(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase2(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase3(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase4(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase5(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase6(__gm__ half *out, __gm__ half *src) {
  runTRowMin<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase7(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 30, 30, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase8(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 30, 30, 216, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase9(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 30, 11, 216, 216, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase10(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 30, 11, 216, 24, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase11(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 238, 238, 40, 40, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase12(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 238, 238, 40, 16, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase13(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 238, 121, 40, 40, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase14(__gm__ float *out, __gm__ float *src) {
  runTRowMin<float, 238, 121, 40, 16, 1>(out, src);
}

template <uint32_t caseId>
void launchTROWMINTestCase(void *out, void *src, aclrtStream stream) {
  switch (caseId) {
  case 1: {
    launchTROWMINCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 2: {
    launchTROWMINCase2<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 3: {
    launchTROWMINCase3<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 4: {
    launchTROWMINCase4<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 5: {
    launchTROWMINCase5<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 6: {
    launchTROWMINCase6<<<1, nullptr, stream>>>((half *)out, (half *)src);
    break;
  }
  case 7: {
    launchTROWMINCase7<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 8: {
    launchTROWMINCase8<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 9: {
    launchTROWMINCase9<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 10: {
    launchTROWMINCase10<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 11: {
    launchTROWMINCase11<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 12: {
    launchTROWMINCase12<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 13: {
    launchTROWMINCase13<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 14: {
    launchTROWMINCase14<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  default: {
  }
  }
}

template void launchTROWMINTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<4>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<5>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<6>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<7>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<8>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<9>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<10>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<14>(void *out, void *src, aclrtStream stream);