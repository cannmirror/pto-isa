#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

using namespace std;
using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
inline __aicore__ void runTTRANS( __gm__ T __out__ *out, __gm__ T __in__ *src) {

    if (block_idx > 0) return;

    using DynShapeDim4 = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStridDim4 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim4, DynStridDim4>;



    constexpr uint16_t aligned_Rows = ( (kTRows_ * sizeof(T) + 31) / 32 ) * (32 / sizeof(T));
    constexpr uint16_t aligned_Cols = ( (kTCols_ * sizeof(T) + 31) / 32 ) * (32 / sizeof(T));

    using TileDataSrc = Tile<Location::Vec, T, kTRows_, aligned_Cols, BLayout::RowMajor>;
    using TileDataDst = Tile<Location::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;


    // TASSIGN(srcTile, 0x0 + 0x1000 * block_idx);
    // TASSIGN(dstTile, 0x8000 + 0x1000 * block_idx);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x20000);

    int offset = 0;

    GlobalData srcGlobal(src + offset,
                         pto::Shape(1, 1, 1, kGRows_, kGCols_),
                         pto::Stride(1, 1, 1, kGCols_, 1));

    GlobalData dstGlobal(out + offset,
                         pto::Shape(1, 1, 1, kGCols_, kGRows_),
                         pto::Stride(1, 1, 1, kGRows_, 1));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TTRANS(dstTile, srcTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}


extern "C" __global__ __aicore__ void launchTTRANS_1(__gm__ uint8_t *out, __gm__ uint8_t *src) {
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t L = 128;
    runTTRANS<float, M, N, K, L>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src)); 
}


template <int32_t tilingKey>
void launchTTRANS(uint8_t *out, uint8_t *src, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTTRANS_1<<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTTRANS<1>(uint8_t *out, uint8_t *src, void *stream);