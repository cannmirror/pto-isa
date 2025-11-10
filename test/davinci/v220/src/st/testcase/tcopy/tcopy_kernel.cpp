#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/pto_instr_impl.hpp>
#include <common/constants.hpp>

using namespace pto;
typedef float IN_DTYPE;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ PTO_INLINE void runTCOPY( __gm__ T __out__ *out, __gm__ T __in__ *src) {
    if (block_idx > 0){return;}
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>;
    TileData srcTile;
    TileData dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x4000);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOPY<TileData, TileData>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}



extern "C" __global__ __aicore__ void launchTCOPY_1(__gm__ float *out, __gm__ float *src) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t K = 128;
        constexpr uint32_t L = 128;
    runTCOPY<float, M, N, K, L>(out, src); 
}



void launchTCOPY1(float *out, float *src, void *stream) {
    launchTCOPY_1<<<1, nullptr, stream>>>(out, src);
}
