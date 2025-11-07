#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;


template <typename Tsrc0, typename Tsrc1, int kGRows0_,int kGCols0_, int kGRows1_,int kGCols1_, int kTRows_, int kTCols_>
inline __aicore__ void runTGATHER(__gm__ Tsrc0 __out__ *out, __gm__ Tsrc0 __in__ *src0, __gm__ Tsrc1 __in__ *src1) {

    using DynShapDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_src0 = GlobalTensor<T, DynShapDim5_src0, DynStridDim5_src0>;
    using DynShapDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_src1 = GlobalTensor<T, DynShapDim5_src1, DynStridDim5_src1>;
    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>
    using DynStridDim5_dst = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_dst = GlobalTensor<Tsrc0, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows1_;
    constexpr int dst_col = kGCols1_;

    using TileData_src0 = Tile<Location::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<Location::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<Location::Vec, Tsrc0, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col);
    TileData_dst dstTile(dst_row, dst_col);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x20000);
    TASSIGN(dstTile, 0x40000);

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}



extern "C" __global__ __aicore__ void test_tgather_float(__gm__ float *out, __gm__ float *src0, __gm__ float *src1) {
    runTGATHER<float, int32_t, 32, 1024, 16, 64, 32, 1024>(out, src0, src1);
}

extern "C" __global__ __aicore__ void test_tgather_int32(__gm__ int32_t *out, __gm__ int32_t *src0, __gm__ int32_t *src1) {
    runTGATHER<int32_t, int32_t, 32, 512, 16, 256, 32, 512>(out, src0, src1);
}

extern "C" __global__ __aicore__ void test_tgather_half(__gm__ int16_t *out, __gm__ int16_t *src0, __gm__ int16_t *src1) {
    runTGATHER<int16_t, int16_t, 16, 1024, 16, 128, 16, 1024>(out, src0, src1);
}

extern "C" __global__ __aicore__ void test_tgather_int16(__gm__ int16_t *out, __gm__ int16_t *src0, __gm__ int16_t *src1) {
    runTGATHER<int16_t, int16_t, 32, 256, 32, 64, 32, 256>(out, src0, src1);
}

void launchTGATHER_demo_float(float *out, float *src0, int32_t *src1, aclrtStream *stream){
    cout<< "launch TGATHER float start!" << endl;
    test_tgather_float<<< 1, nullptr, stream>>>(out, src0, src1);
    cout<< "launch TGATHER float end!" << endl;
}

void launchTGATHER_demo_int32(int32_t *out, int32_t *src0, int32_t *src1, aclrtStream *stream){
    cout<< "launch TGATHER int32 start!" << endl;
    test_tgather_int32<<< 1, nullptr, stream>>>(out, src0, src1);
    cout<< "launch TGATHER int32 end!" << endl;
}

void launchTGATHER_demo_half(int16_t *out, int16_t *src0, int32_t *src1, aclrtStream *stream){
    cout<< "launch TGATHER half start!" << endl;
    test_tgather_half<<< 1, nullptr, stream>>>(out, src0, src1);
    cout<< "launch TGATHER half end!" << endl;
}

void launchTGATHER_demo_int16(int16_t *out, int16_t *src0, int32_t *src1, aclrtStream *stream){
    cout<< "launch TGATHER int16 start!" << endl;
    test_tgather_int16<<< 1, nullptr, stream>>>(out, src0, src1);
    cout<< "launch TGATHER int16 end!" << endl;
}