#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

using namespace std;
using namespace pto;

template <typename T, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
inline __aicore__ void runTCVT(__gm__ T __out__ *out, __gm__ S __in__ *src){

    using DynShapeDim4 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim4 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalDataSrc = GlobalTensor<S, DynShapeDim4, DynStridDim4>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    using TileDataSrc = Tile<Location::Vec, S, kTRows_, kTCols_, BLayout::RowMajor>;
    using TileDataDst = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;

    TASSIGN(srcTile, 0x0 + 0x400*block_idx);
    TASSIGN(dstTile, 0x20000 + 0x400*block_idx);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TCVT(dstTile, srcTile, RoundMode::CAST_RINT);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTCVT_1(__gm__ int *out, __gm__ float *src){
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t L = 128;
    runTCVT<int, float, M, N, K, L>(out, src);
}

extern "C" __global__ __aicore__ void launchTCVT_2(__gm__ float *out, __gm__ int *src){
    constexpr uint32_t M = 256;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 256;
    constexpr uint32_t L = 64;
    runTCVT<float, int, M, N, K, L>(out, src);
}

extern "C" __global__ __aicore__ void launchTCVT_3(__gm__ int16_t *out, __gm__ float *src){
    constexpr uint32_t M = 16;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 16;
    constexpr uint32_t L = 32;
    runTCVT<int16_t, float, M, N, K, L>(out, src);
}

extern "C" __global__ __aicore__ void launchTCVT_4(__gm__ int *out, __gm__ float *src){
    constexpr uint32_t M = 32;
    constexpr uint32_t N = 512;
    constexpr uint32_t K = 32;
    constexpr uint32_t L = 512;
    runTCVT<int, float, M, N, K, L>(out, src);
}

extern "C" __global__ __aicore__ void launchTCVT_5(__gm__ int16_t *out, __gm__ int *src){
    constexpr uint32_t M = 2;
    constexpr uint32_t N = 512;
    constexpr uint32_t K = 2;
    constexpr uint32_t L = 512;
    runTCVT<int16_t, int, M, N, K, L>(out, src);
}

extern "C" __global__ __aicore__ void launchTCVT_6(__gm__ float *out, __gm__ int *src){
    constexpr uint32_t M = 4;
    constexpr uint32_t N = 4096;
    constexpr uint32_t K = 4;
    constexpr uint32_t L = 4096;
    runTCVT<float, int, M, N, K, L>(out, src);
}

extern "C" __global__ __aicore__ void launchTCVT_7(__gm__ float *out, __gm__ int16_t *src){
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 64;
    constexpr uint32_t L = 64;
    runTCVT<float, int16_t, M, N, K, L>(out, src);
}

void launchTCVT1(int *out, float *src, void *stream){
    launchTCVT_1<<<1, nullptr, stream>>>(out, src);
}

void launchTCVT2(float *out, int *src, void *stream){
    launchTCVT_2<<<1, nullptr, stream>>>(out, src);
}

void launchTCVT3(int16_t *out, float *src, void *stream){
    launchTCVT_3<<<1, nullptr, stream>>>(out, src);
}

void launchTCVT4(int *out, float *src, void *stream){
    launchTCVT_4<<<1, nullptr, stream>>>(out, src);
}

void launchTCVT5(int16_t *out, int *src, void *stream){
    launchTCVT_5<<<1, nullptr, stream>>>(out, src);
}

void launchTCVT6(float *out, int *src, void *stream){
    launchTCVT_6<<<1, nullptr, stream>>>(out, src);
}

void launchTCVT7(float *out, int16_t *src, void *stream){
    launchTCVT_7<<<1, nullptr, stream>>>(out, src);
}