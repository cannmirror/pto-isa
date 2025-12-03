#include <common/pto_tileop.hpp>
#include <common/tile_tensor_impl.hpp>
#include <common/constants.hpp>


using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ void runTROWMAX( __gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    
    using srcTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<Location::Vec, T, kTRows_, 16, BLayout::RowMajor, -1, -1>; 
    
    srcTileData srcTile(kTRows_, kTCols_);
    dstTileData dstTile(kTRows_, 1);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x11000);

    GlobalData srcGlobal(src, DynShapeDim5(kTRows_, kTCols_), 
            DynStridDim5(kTRows_, kTCols_));
    GlobalData dstGlobal(out, DynShapeDim5(kTRows_, 1),
            DynStridDim5(kTRows_, 1));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMAX(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTROWMAX(T *out, T *src, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        runTROWMAX<half, kGRows_, kGCols_, kTRows_, kTCols_>((half*)(out),
                                                          (half*)(src));
    else 
        runTROWMAX<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src);
}

template void LaunchTROWMAX<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTROWMAX<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWMAX<aclFloat16, 161, 161, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTROWMAX<float, 77, 81, 32, 16>(float *out, float *src, void *stream);
template void LaunchTROWMAX<float, 32, 32, 32, 16>(float *out, float *src, void *stream);