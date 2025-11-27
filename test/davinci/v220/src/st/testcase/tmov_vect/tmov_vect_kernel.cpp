#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/pto_instr_impl.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template<typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ __aicore__ void runTMOV(__gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using SrcTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    
    SrcTileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, kTCols_);

    TASSIGN(srcTile, 0x0 + 0x400*block_idx);
    TASSIGN(dstTile, 0x20000 + 0x400*block_idx);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(dst);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV<DstTileData, SrcTileData>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}


template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTMOV(T *out, T *src, void *stream) {
    runTMOV<T, kGRows_, kGCols_, kTRows_, kTCols_>
        <<<1, nullptr, stream>>>(out, src);
}

template void launchTMOV<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void launchTMOV<float, 32, 32, 32, 32>(float *out, float *src, void *stream);
template void launchTMOV<float, 128, 128, 128, 128>(float *out, float *src, void *stream);
template void launchTMOV<float, 128, 32, 128, 32>(float *out, float *src, void *stream);
template void launchTMOV<float, 128, 64, 128, 64>(float *out, float *src, void *stream);
template void launchTMOV<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<aclFloat16, 32, 32, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<aclFloat16, 128, 128, 128, 128>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<aclFloat16, 128, 32, 128, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<aclFloat16, 128, 64, 128, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<uint8_t, 64, 64, 64, 64>(uint8_t *out, uint8_t *src, void *stream);
template void launchTMOV<uint8_t, 32, 32, 32, 32>(uint8_t *out, uint8_t *src, void *stream);
template void launchTMOV<uint8_t, 128, 128, 128, 128>(uint8_t *out, uint8_t *src, void *stream);
template void launchTMOV<uint8_t, 128, 32, 128, 32>(uint8_t *out, uint8_t *src, void *stream);
template void launchTMOV<uint8_t, 128, 64, 128, 64>(uint8_t *out, uint8_t *src, void *stream);
