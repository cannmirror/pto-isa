#include <common/tile_tensor_impl.hpp>
#include <common/pto_tileop.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, CmpMode cmpMode>
__global__ __aicore__ void runTCmps( __gm__ uint8_t __out__ *out, __gm__ T __in__ *src0,  T src1) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src0 = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using GlobalData_dst = GlobalTensor<uint8_t, DynShapeDim5, DynStridDim5>;
    using TileData_src0 = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<Location::Vec, uint8_t, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(kTRows_, kTCols_);
    TileData_dst dstTile(kTRows_, kTCols_);
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x20000 + 0x400 * block_idx);

    GlobalData_src0 src0Global(src0);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCMPS(dstTile, src0Tile, src1, cmpMode);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, CmpMode cmpMode>
void LaunchTCmps(uint8_t *out, T *src0, T src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> )
        runTCmps<half, kGRows_, kGCols_, kTRows_, kTCols_, cmpMode><<<1, nullptr, stream>>>((out),
                                                                                  (half*)(src0),
                                                                                  (half)(src1));
    else 
        runTCmps<T, kGRows_, kGCols_, kTRows_, kTCols_, cmpMode><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTCmps<float, 1, 64, 1, 64, CmpMode::EQ>(uint8_t *out, float *src0, float src1, void *stream);
template void LaunchTCmps<float, 8, 64, 8, 64, CmpMode::GT>(uint8_t *out, float *src0, float src1, void *stream);
template void LaunchTCmps<int32_t, 4, 64, 4, 64, CmpMode::NE>(uint8_t *out, int32_t *src0, int32_t src1, void *stream);
template void LaunchTCmps<int32_t, 128, 128, 64, 64, CmpMode::LT>(uint8_t *out, int32_t *src0, int32_t src1, void *stream);
template void LaunchTCmps<int32_t, 64, 64, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t src1, void *stream);
template void LaunchTCmps<int32_t, 16, 32, 16, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t src1, void *stream);
template void LaunchTCmps<float, 128, 128, 128, 128, CmpMode::LE>(uint8_t *out, float *src0, float src1, void *stream);
template void LaunchTCmps<int32_t, 77, 81, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t src1, void *stream);
template void LaunchTCmps<int32_t, 32, 32, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t src1, void *stream);