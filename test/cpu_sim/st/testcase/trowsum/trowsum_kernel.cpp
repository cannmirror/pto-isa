#include <common/pto_tileop.hpp>
#include <common/tile_tensor_impl.hpp>
#include <common/constants.hpp>

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ inline void runTROWSUM(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<kGCols_, kGCols_, kGCols_, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData tmpTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    std::fill(dstTile.data(),dstTile.data()+kTRows_*kTCols_,0);
    
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWSUM(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTROWSUM(T *out, T *src, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> ) {
        runTROWSUM<half, kGRows_, kGCols_, kTRows_, kTCols_>( (half*)(out), (half*)src);
    } else {
        runTROWSUM<T, kGRows_, kGCols_, kTRows_, kTCols_>( out, src);
    }
}

template void LaunchTROWSUM<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWSUM<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);