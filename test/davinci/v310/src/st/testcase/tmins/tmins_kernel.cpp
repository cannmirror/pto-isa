#include <common/tile_tensor_impl.hpp>
#include <common/pto_tileop.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int kPadValue_>
struct GenericDataSelector;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_>
struct GenericDataSelector<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, PAD_VALUE_NULL> {
    using DynShapeDim5 = Shape<1, 1, 1, kTRows_, kTCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
    using GlobalType = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileType = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, kTRows_, kTCols_, SLayout::NoneBox, 512, PadValue::Null>;
};

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_>
struct GenericDataSelector<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, PAD_VALUE_MAX> {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalType = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileType = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox, 512, PadValue::Max>;
};

template <typename T, typename ST, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int kPadValue_>
__global__ __aicore__ void runTMINS(__gm__ T __out__ *out, __gm__ T __in__ *src0, ST scalar) {
    using GDS = GenericDataSelector<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, kPadValue_>;
    using GlobalData = typename GDS::GlobalType;
    using TileData = typename GDS::TileType;
    TileData src0Tile;
    TileData dstTile;
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = 0;
    GlobalData src0Global(src0 + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMINS(dstTile, src0Tile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int kPadValue_>
void LaunchTMins(T *out, T *src0, T scalar, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runTMINS<half, uint16_t, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, kPadValue_><<<1, nullptr, stream>>>((half*)(out),
                                                                                                                 (half*)(src0),
                                                                                                                 scalar);
    else 
        runTMINS<T, T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, kPadValue_><<<1, nullptr, stream>>>(out, src0, scalar);
}

template void LaunchTMins<float, 64, 64, 32, 32, 64, 64, PAD_VALUE_NULL>(float *out, float *src0, float scalar, void *stream);
template void LaunchTMins<float, 128, 128, 64, 64, 128, 128, PAD_VALUE_NULL>(float *out, float *src0, float scalar, void *stream);
template void LaunchTMins<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(float *out, float *src0, float scalar, void *stream);
template void LaunchTMins<float, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(float *out, float *src0, float scalar, void *stream);
template void LaunchTMins<float, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(float *out, float *src0, float scalar, void *stream);
template void LaunchTMins<aclFloat16, 16, 200, 20, 224, 16, 200, PAD_VALUE_MAX>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 scalar, void *stream);
template void LaunchTMins<int32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(int32_t *out, int32_t *src0, int32_t scalar, void *stream);
template void LaunchTMins<uint32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(uint32_t *out, uint32_t *src0, uint32_t scalar, void *stream);
template void LaunchTMins<int16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(int16_t *out, int16_t *src0, int16_t scalar, void *stream);
// template void LaunchTMins<uint16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(uint16_t *out, uint16_t *src0, uint16_t scalar, void *stream);
template void LaunchTMins<int8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(int8_t *out, int8_t *src0, int8_t scalar, void *stream);
template void LaunchTMins<uint8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(uint8_t *out, uint8_t *src0, uint8_t scalar, void *stream);
template void LaunchTMins<aclFloat16, 16, 256, 16, 256, 16, 256, PAD_VALUE_NULL>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 scalar, void *stream);