#include <common/pto_tileop.hpp>
#include <common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
__aicore__ PTO_INLINE void runTColSum(__gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynDim2Shape  = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalData dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using srcTileData = Tile<Location::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<Location::Vec, T, dstRow, col, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(srcValidRow, validCol);
    dstTileData dstTile(dstRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x20000);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLSUM(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTCOLSUMCase1(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 64, 64, 1, 64, 64>(out, src);
}

template <uint32_t caseId>
void launchTCOLSUMTestCase(void *out, void *src, aclrtStream stream) {
    switch (caseId) {
        case 1: {
            launchTCOLSUMCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLSUMTestCase<1>(void *out, void *src, aclrtStream stream);