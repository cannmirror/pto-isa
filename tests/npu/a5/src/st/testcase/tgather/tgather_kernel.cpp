/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/tile_tensor_impl.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"
#include "tgather_common.h"

using namespace std;
using namespace pto;

template <typename Tsrc0, typename Tsrc1, int kGRows0_, int kGCols0_, int kGRows1_, int kGCols1_, int kTRows_,
    int kTCols_>
inline __aicore__ void runTGather(__gm__ Tsrc0 __out__ *out, __gm__ Tsrc0 __in__ *src0, __gm__ Tsrc1 __in__ *src1)
{
    using DynShapeDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_src0 = GlobalTensor<Tsrc0, DynShapeDim5_src0, DynStridDim5_src0>;
    using DynShapeDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_src1 = GlobalTensor<Tsrc1, DynShapeDim5_src1, DynStridDim5_src1>;
    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_dst = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_dst = GlobalTensor<Tsrc0, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_;
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows1_;
    constexpr int dst_col = kGCols1_;

    using TileData_src0 = Tile<Location::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<Location::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<Location::Vec, Tsrc0, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col);
    TileData_dst dstTile(dst_row, dst_col);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x20000);
    TASSIGN(dstTile, 0x28000);

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

extern "C" __global__ __aicore__ void test_tgather_float(__gm__ float *out, __gm__ float *src0, __gm__ int32_t *src1)
{
    runTGather<float, int32_t, 32, 1024, 16, 64, 32, 1024>(out, src0, src1);
}

extern "C" __global__ __aicore__ void test_tgather_int32(
    __gm__ int32_t *out, __gm__ int32_t *src0, __gm__ int32_t *src1)
{
    runTGather<int32_t, int32_t, 32, 512, 16, 256, 32, 512>(out, src0, src1);
}

extern "C" __global__ __aicore__ void test_tgather_half(__gm__ int16_t *out, __gm__ int16_t *src0, __gm__ int16_t *src1)
{
    runTGather<int16_t, int16_t, 16, 1024, 16, 128, 16, 1024>(out, src0, src1);
}

extern "C" __global__ __aicore__ void test_tgather_int16(
    __gm__ int16_t *out, __gm__ int16_t *src0, __gm__ int16_t *src1)
{
    runTGather<int16_t, int16_t, 32, 256, 32, 64, 32, 256>(out, src0, src1);
}

void launchTGATHER_demo_float(float *out, float *src0, int32_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER float start!" << endl;
    test_tgather_float<<<1, nullptr, stream>>>(out, src0, src1);
    cout << "launch TGATHER float end!" << endl;
}

void launchTGATHER_demo_int32(int32_t *out, int32_t *src0, int32_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER int32 start!" << endl;
    test_tgather_int32<<<1, nullptr, stream>>>(out, src0, src1);
    cout << "launch TGATHER int32 end!" << endl;
}

void launchTGATHER_demo_half(int16_t *out, int16_t *src0, int16_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER half start!" << endl;
    test_tgather_half<<<1, nullptr, stream>>>(out, src0, src1);
    cout << "launch TGATHER half end!" << endl;
}

void launchTGATHER_demo_int16(int16_t *out, int16_t *src0, int16_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER int16 start!" << endl;
    test_tgather_int16<<<1, nullptr, stream>>>(out, src0, src1);
    cout << "launch TGATHER int16 end!" << endl;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
__aicore__ void runTGATHER(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, (kTRows_ + 5), (kTCols_ + 32), BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + (kTRows_ + 5) * (kTCols_ + 32) * sizeof(T));

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, maskPattern>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
__aicore__ void runTGATHERDIFF(__gm__ dstT __out__ *out, __gm__ srcT __in__ *src)
{
    using DynShapeDim5 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using SrcGlobalData = GlobalTensor<srcT, DynShapeDim5, DynStridDim5>;
    using DstGlobalData = GlobalTensor<dstT, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, srcT, (kTRows_ + 5), (kTCols_ + 32), BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<Location::Vec, dstT, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + (kTRows_ + 5) * (kTCols_ + 32) * sizeof(dstT));

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, maskPattern>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTGATHER_31(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int8_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_32(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint8_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ uint8_t *>(out), reinterpret_cast<__gm__ uint8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_33(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int8_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_34(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint8_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ uint8_t *>(out), reinterpret_cast<__gm__ uint8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_35(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int8_t, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_36(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint8_t, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ uint8_t *>(out), reinterpret_cast<__gm__ uint8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_37(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int8_t, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ int8_t *>(out), reinterpret_cast<__gm__ int8_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_21(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint16_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ uint16_t *>(out), reinterpret_cast<__gm__ uint16_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_22(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint16_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ uint16_t *>(out), reinterpret_cast<__gm__ uint16_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_23(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int16_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ int16_t *>(out), reinterpret_cast<__gm__ int16_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_24(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int16_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ int16_t *>(out), reinterpret_cast<__gm__ int16_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_25(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint32_t, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ uint32_t *>(out), reinterpret_cast<__gm__ uint32_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_26(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int32_t, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int32_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_27(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int32_t, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int32_t *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_11(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_12(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_13(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_14(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_15(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_16(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_17(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_1(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0101_ROW, FLOAT_P0101_COL, FLOAT_P0101_ROW, FLOAT_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_2(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_3(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0001_ROW, FLOAT_P0001_COL, FLOAT_P0001_ROW, FLOAT_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_4(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0010_ROW, FLOAT_P0010_COL, FLOAT_P0010_ROW, FLOAT_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_5(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_6(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_7(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ __aicore__ void launchTGATHER_8(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHERDIFF<float, int32_t, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL,
        MaskPattern::P1010>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ float *>(src));
}

template <int32_t tilingKey>
void launchTGATHER_demo(uint8_t *out, uint8_t *src, void *stream)
{
    if constexpr (tilingKey == FP0101) {
        launchTGATHER_1<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FP1010) {
        launchTGATHER_2<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FP0001) {
        launchTGATHER_3<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FP0010) {
        launchTGATHER_4<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FP0100) {
        launchTGATHER_5<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FP1000) {
        launchTGATHER_6<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FP1111) {
        launchTGATHER_7<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == FPINT1010) {
        launchTGATHER_8<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP0101) {
        launchTGATHER_11<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP1010) {
        launchTGATHER_12<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP0001) {
        launchTGATHER_13<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP0010) {
        launchTGATHER_14<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP0100) {
        launchTGATHER_15<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP1000) {
        launchTGATHER_16<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == HP1111) {
        launchTGATHER_17<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == U16P0101) {
        launchTGATHER_21<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == U16P1010) {
        launchTGATHER_22<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == I16P0001) {
        launchTGATHER_23<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == I16P0010) {
        launchTGATHER_24<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == U32P0100) {
        launchTGATHER_25<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == I32P1000) {
        launchTGATHER_26<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == I32P1111) {
        launchTGATHER_27<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP0101) {
        launchTGATHER_31<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP1010) {
        launchTGATHER_32<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP0001) {
        launchTGATHER_33<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP0010) {
        launchTGATHER_34<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP0100) {
        launchTGATHER_35<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP1000) {
        launchTGATHER_36<<<1, nullptr, stream>>>(out, src);
    } else if constexpr (tilingKey == BP1111) {
        launchTGATHER_37<<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTGATHER_demo<FP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP1111>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FPINT1010>(uint8_t *out, uint8_t *src, void *stream);

template void launchTGATHER_demo<HP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP1111>(uint8_t *out, uint8_t *src, void *stream);

template void launchTGATHER_demo<U16P0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<U16P1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I16P0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I16P0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<U32P0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I32P1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I32P1111>(uint8_t *out, uint8_t *src, void *stream);

template void launchTGATHER_demo<BP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<BP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<BP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<BP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<BP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<BP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<BP1111>(uint8_t *out, uint8_t *src, void *stream);
