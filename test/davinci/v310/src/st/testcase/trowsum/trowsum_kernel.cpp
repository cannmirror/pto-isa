#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;

namespace TRowSumTest {
    template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
    __global__ __aicore__ void runTRowsum(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ T __in__ *tmp)
    {
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

    template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
    void launchTROWSUMTest(T *out, T *src, aclrtStream stream)
    {
        if constexpr (std::is_same<T, uint16_t>::value) {
            runTRowsum<half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(
                (half *)out, (half *)src, (half *)nullptr);
        } else {
            runTRowsum<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src, nullptr);
        }
    }

    constexpr int smallSize = 16;
    constexpr int bigSize666 = 666;
    constexpr int bigSizeAligned = 672;

    template void launchTROWSUMTest<float, smallSize, smallSize, smallSize, smallSize>(float *out,
        float *src, aclrtStream stream);
    template void launchTROWSUMTest<uint16_t, smallSize, smallSize, smallSize, smallSize>(uint16_t *out,
        uint16_t *src, aclrtStream stream);
    template void launchTROWSUMTest<float, bigSize666, bigSize666, bigSize666, bigSizeAligned>(float *out,
        float *src, aclrtStream stream);
};
