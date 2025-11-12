#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ PTO_INLINE void runTRowsum_single(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ T __in__ *tmp) {
    using DynShapeDim4 = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStridDim4 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim4, DynStridDim4>;
    using srcTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(kTRows_, kTCols_);
    srcTileData tmpTile;
    dstTileData dstTile(kTRows_, 1);
    int tileSize = kTRows_ * kTCols_;

    TASSIGN(srcTile, tileSize * 2 * block_idx * sizeof(T));
    TASSIGN(dstTile, (tileSize + tileSize * 2 * block_idx) * sizeof(T));
    TASSIGN(tmpTile, 0);

    int offset = tileSize * block_idx;
    GlobalData srcGlobal(src + offset, Shape(1, 1, 1, kGRows_, kGCols_), pto::Stride(1, 1, 1, kGCols_, 1));
    GlobalData dstGlobal(out + offset, Shape(1, 1, 1, kGRows_, kGCols_), pto::Stride(1, 1, 1, kGCols_, 1));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWSUM(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTROWSUM_float(__gm__ float *out, __gm__ float *src0)
{
    runTRowsum_single<float, 16, 16, 16, 16>(out, src0, nullptr);
}

extern "C" __global__ __aicore__ void launchTROWSUM_half(__gm__ uint16_t *out, __gm__ uint16_t *src0)
{
    runTRowsum_single<uint16_t, 16, 16, 16, 16>(out, src0, nullptr);
}

void launchTROWSUM_demo_float(float *out, float *src0, aclrtStream stream){
    cout << "launch TROWSUM float start!" << endl;
    launchTROWSUM_float<<<1, nullptr, stream>>>(out, src0);
    cout << "launch TROWSUM float end!" << endl;
}

void launchTROWSUM_demo_half(uint16_t *out, uint16_t *src0, aclrtStream stream){
    cout << "launch TROWSUM half start!" << endl;
    launchTROWSUM_half<<<1, nullptr, stream>>>(out, src0);
    cout << "launch TROWSUM half end!" << endl;
}