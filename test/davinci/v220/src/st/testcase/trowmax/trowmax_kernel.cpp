#include <acl/acl.h>
#include <common/constants.hpp>
#include <common/pto_tileop.hpp>

using namespace std;
using namespace pto;

template <typename T, int row, int vaildRow, int srcCol, int srcVaildCol,
          int dstCol>
__aicore__ PTO_INLINE void runTRowMax(__gm__ T *out, __gm__ T *src) {
  using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
  using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
  using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
  GlobalData srcGlobal(src, DynDim2Shape(vaildRow, srcVaildCol),
                       DynDim2Stride(row, srcCol));
  GlobalData dstGlobal(out, DynDim2Shape(vaildRow, dstCol),
                       DynDim2Stride(row, dstCol));

  using srcTileData =
      Tile<Location::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
  using dstTileData =
      Tile<Location::Vec, T, row, 16, BLayout::RowMajor, -1, -1>;
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
  TROWMAX(dstTile, srcTile, tmpTile);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ __aicore__ void launchTROWMAXCase1(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase2(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase3(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase4(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase5(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase6(__gm__ half *out,
                                                         __gm__ half *src) {
  runTRowMax<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase7(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 30, 30, 216, 216, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase8(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 30, 30, 216, 24, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase9(__gm__ float *out,
                                                         __gm__ float *src) {
  runTRowMax<float, 30, 11, 216, 216, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase10(__gm__ float *out,
                                                          __gm__ float *src) {
  runTRowMax<float, 30, 11, 216, 24, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase11(__gm__ float *out,
                                                          __gm__ float *src) {
  runTRowMax<float, 238, 238, 40, 40, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase12(__gm__ float *out,
                                                          __gm__ float *src) {
  runTRowMax<float, 238, 238, 40, 16, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase13(__gm__ float *out,
                                                          __gm__ float *src) {
  runTRowMax<float, 238, 121, 40, 40, 1>(out, src);
}
extern "C" __global__ __aicore__ void launchTROWMAXCase14(__gm__ float *out,
                                                          __gm__ float *src) {
  runTRowMax<float, 238, 121, 40, 16, 1>(out, src);
}

template <uint32_t caseId>
void launchTROWMAXTestCase(void *out, void *src, aclrtStream stream) {
  switch (caseId) {
  case 1: {
    launchTROWMAXCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 2: {
    launchTROWMAXCase2<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 3: {
    launchTROWMAXCase3<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 4: {
    launchTROWMAXCase4<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 5: {
    launchTROWMAXCase5<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 6: {
    launchTROWMAXCase6<<<1, nullptr, stream>>>((half *)out, (half *)src);
    break;
  }
  case 7: {
    launchTROWMAXCase7<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 8: {
    launchTROWMAXCase8<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 9: {
    launchTROWMAXCase9<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 10: {
    launchTROWMAXCase10<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 11: {
    launchTROWMAXCase11<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 12: {
    launchTROWMAXCase12<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 13: {
    launchTROWMAXCase13<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  case 14: {
    launchTROWMAXCase14<<<1, nullptr, stream>>>((float *)out, (float *)src);
    break;
  }
  default: {
  }
  }
}

template void launchTROWMAXTestCase<1>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<2>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<3>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<4>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<5>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<6>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<7>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<8>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<9>(void *out, void *src,
                                       aclrtStream stream);
template void launchTROWMAXTestCase<10>(void *out, void *src,
                                        aclrtStream stream);
template void launchTROWMAXTestCase<11>(void *out, void *src,
                                        aclrtStream stream);
template void launchTROWMAXTestCase<12>(void *out, void *src,
                                        aclrtStream stream);
template void launchTROWMAXTestCase<13>(void *out, void *src,
                                        aclrtStream stream);
template void launchTROWMAXTestCase<14>(void *out, void *src,
                                        aclrtStream stream);