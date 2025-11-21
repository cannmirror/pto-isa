#include <common/tile_tensor_impl.hpp>
#include <common/pto_tileop.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ __aicore__ void runTRsqrt( __gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData srcGlobal(src + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TRSQRT(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTRsqrt(T *out, T *src, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runTRsqrt<half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>((half*)(out),
                                                                                    (half*)(src));
    else 
        runTRsqrt<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTRsqrt<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTRsqrt<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);