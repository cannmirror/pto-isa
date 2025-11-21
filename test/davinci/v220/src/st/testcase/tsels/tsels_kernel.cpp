#include <common/tile_tensor_impl.hpp>
#include <common/pto_tileop.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ __aicore__ void runTSels(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1, uint8_t selectMode) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kTRows_, kTCols_);
    TileData src1Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(src1Tile, 0x4000 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSELS(dstTile, src0Tile, src1Tile, selectMode);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTSels(T *out, T *src0, T *src1, uint8_t selectMode, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runTSels<half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>((half*)(out),
                                                                                  (half*)(src0),
                                                                                  (half*)(src1),
                                                                                  selectMode);
    else 
        runTSels<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src0, src1, selectMode);
}

template void LaunchTSels<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<float, 16, 256, 16, 256>(float *out, float *src0, float *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<float, 2, 128, 2, 128>(float *out, float *src0, float *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<float, 2, 32, 2, 32>(float *out, float *src0, float *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<float, 2, 160, 2, 160>(float *out, float *src0, float *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<aclFloat16, 2, 128, 2, 128>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<aclFloat16, 2, 32, 2, 32>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, uint8_t selectMode, void *stream);
template void LaunchTSels<aclFloat16, 2, 160, 2, 160>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, uint8_t selectMode, void *stream);