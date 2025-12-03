/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

template <typename GlobalData, typename TileData>
__aicore__ inline void TSTORE_MAT2GM(GlobalData &dst, TileData &src)
{
    __cbuf__ typename TileData::DType *srcAddr = (__cbuf__ typename TileData::DType *)src.data();
    typename GlobalData::DType *dstAddr = dst.data();

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);

    uint32_t validRow = src.GetValidRow();
    uint32_t validCol = src.GetValidCol();

    if constexpr (GlobalData::layout == pto::Layout::ND &&
        GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        uint16_t nBurst = validRow;
        uint16_t lenBurst = validCol / blockSizeElem;
        uint16_t l1Gap = (TileData::Cols - validCol) / blockSizeElem;
        uint16_t gmGap = 0;
        copy_cbuf_to_gm(dstAddr, srcAddr, (uint8_t)0, nBurst, lenBurst, l1Gap, gmGap);
    } else if constexpr (GlobalData::layout == pto::Layout::DN &&
        GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        uint16_t nBurst = validCol;
        uint16_t lenBurst = validRow / blockSizeElem;
        uint16_t l1Gap = (TileData::Rows - validRow) / blockSizeElem;
        uint16_t gmGap = 0;
        copy_cbuf_to_gm(dstAddr, srcAddr, (uint8_t)0, nBurst, lenBurst, l1Gap, gmGap);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        uint16_t nBurst = (validCol + blockSizeElem - 1) / blockSizeElem;
        uint16_t lenBurst = validRow;
        uint16_t l1Gap = TileData::Rows - validRow;
        uint16_t gmGap = 0;
        copy_cbuf_to_gm(dstAddr, srcAddr, (uint8_t)0, nBurst, lenBurst, l1Gap, gmGap);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ZN) {
        uint16_t nBurst = (validRow + blockSizeElem - 1) / blockSizeElem;
        uint16_t lenBurst = validCol;
        uint16_t l1Gap = TileData::Cols - validCol;
        uint16_t gmGap = 0;
        copy_cbuf_to_gm(dstAddr, srcAddr, (uint8_t)0, nBurst, lenBurst, l1Gap, gmGap);
    }
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__aicore__ inline void RunTLoadND2ND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        gWholeShape4,
        1};
    constexpr int blockSize = 32 / sizeof(T);
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validCol = gShape4;
    constexpr int Rows = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int Cols = (gShape4 + blockSize - 1) / blockSize * blockSize;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Mat, T, Rows, Cols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    int offset = 0;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_MAT2GM<GlobalData, TileData>(dstGlobal, srcTile);
    out = dstGlobal.data();
}
template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__aicore__ inline void RunTLoadDN2DN(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        1,
        gWholeShape3};

    constexpr int blockSize = 32 / sizeof(T);
    constexpr int Rows = (gShape3 + blockSize - 1) / blockSize * blockSize;
    constexpr int Cols = gShape0 * gShape1 * gShape2 * gShape4;
    constexpr int validRow = gShape3;
    constexpr int validCol = gShape0 * gShape1 * gShape2 * gShape4;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<Location::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1>;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    int offset = 0;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_MAT2GM(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__aicore__ inline void RunTLoadNZ2NZ(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        gWholeShape4,
        1};
    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using TileData = Tile<Location::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    int validRow = gShape2 * gShape3;
    int validCol = gShape0 * gShape1 * gShape4;
    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_MAT2GM(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__aicore__ inline void RunTLoadND2NZ(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        gWholeShape4,
        1};
    constexpr int c0_size = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int Rows = (gShape0 * gShape1 * gShape2 * gShape3 + 16 - 1) / 16 * 16;
    constexpr int Cols = (gShape4 + c0_size - 1) / c0_size * c0_size;

    using SrcDynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using SrcDynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, SrcDynShapeDim5, SrcDynStridDim5, Layout::ND>;
    using TileData = Tile<Location::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    int validRow = gShape0 * gShape1 * gShape2 * gShape3;
    int validCol = gShape4;
    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_MAT2GM(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__aicore__ inline void RunTLoadDN2ZN(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        1,
        gWholeShape3};
    constexpr int c0_size = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int Rows = (gShape3 + c0_size - 1) / c0_size * c0_size;
    constexpr int Cols = (gShape4 + 16 - 1) / 16 * 16;

    using SrcDynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using SrcDynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, SrcDynShapeDim5, SrcDynStridDim5, Layout::DN>;
    using TileData = Tile<Location::Mat, T, Rows, Cols, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;

    int validRow = gShape3;
    int validCol = gShape4;
    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_MAT2GM(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ __aicore__ void TLoadKernel(__gm__ T *out, __gm__ T *src)
{
    if constexpr (format == 0) {
        RunTLoadND2ND<T,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4>(out, src);
    } else if constexpr (format == 1) {
        RunTLoadDN2DN<T,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4>(out, src);
    } else if constexpr (format == 2) {
        RunTLoadNZ2NZ<T,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4>(out, src);
    } else if constexpr (format == 3) {
        RunTLoadND2NZ<T,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4>(out, src);
    } else if constexpr (format == 4) {
        RunTLoadDN2ZN<T,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4>(out, src);
    }
}

// format = 0: ND2ND
// foramt = 1: DN2DN
// format = 2: NZ2NZ
// format = 3: ND2NZ
template <int format, typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTLoad(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        TLoadKernel<bfloat16_t,
            format,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<bfloat16_t *>(src));
    } else {
        TLoadKernel<T,
            format,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4><<<1, nullptr, stream>>>(out, src);
    }
}

template void LaunchTLoad<0, float, 1, 1, 1, 3, 128, 3, 3, 3, 32, 128>(float *out, float *src, void *stream);
template void LaunchTLoad<0, int16_t, 2, 2, 1, 2, 32, 3, 3, 3, 111, 64>(int16_t *out, int16_t *src, void *stream);
template void LaunchTLoad<0, int8_t, 1, 2, 1, 11, 32, 1, 3, 2, 93, 32>(int8_t *out, int8_t *src, void *stream);
template void LaunchTLoad<1, float, 1, 1, 1, 128, 3, 3, 3, 3, 128, 32>(float *out, float *src, void *stream);
template void LaunchTLoad<1, int16_t, 2, 2, 1, 32, 2, 3, 3, 3, 64, 111>(int16_t *out, int16_t *src, void *stream);
template void LaunchTLoad<1, int8_t, 1, 2, 1, 32, 11, 1, 3, 2, 32, 93>(int8_t *out, int8_t *src, void *stream);
template void LaunchTLoad<2, float, 1, 5, 21, 16, 8, 1, 5, 21, 16, 8>(float *out, float *src, void *stream);
template void LaunchTLoad<2, int16_t, 2, 15, 11, 16, 16, 3, 23, 13, 16, 16>(int16_t *out, int16_t *src, void *stream);
template void LaunchTLoad<2, int8_t, 1, 16, 32, 16, 32, 1, 32, 32, 16, 32>(int8_t *out, int8_t *src, void *stream);
template void LaunchTLoad<3, float, 1, 1, 1, 49, 35, 1, 1, 1, 49, 35>(float *out, float *src, void *stream);
template void LaunchTLoad<3, int16_t, 1, 1, 1, 155, 250, 1, 1, 1, 752, 1000>(int16_t *out, int16_t *src, void *stream);
template void LaunchTLoad<3, int8_t, 1, 1, 1, 1023, 511, 1, 1, 1, 1024, 1024>(int8_t *out, int8_t *src, void *stream);
template void LaunchTLoad<3, uint16_t, 1, 1, 1, 1023, 51, 1, 1, 1, 1024, 1024>(
    uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<0, uint16_t, 1, 1, 1, 128, 128, 1, 1, 1, 256, 256>(
    uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<1, uint16_t, 1, 2, 2, 128, 311, 4, 3, 3, 256, 400>(
    uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<2, uint16_t, 2, 4, 5, 16, 16, 7, 7, 7, 16, 16>(uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<3, uint16_t, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<3, uint16_t, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16>(uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<3, uint16_t, 1, 1, 1, 256, 1024, 1, 1, 1, 256, 1024>(
    uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<0, int64_t, 1, 1, 1, 3, 128, 3, 3, 3, 32, 128>(int64_t *out, int64_t *src, void *stream);
template void LaunchTLoad<0, uint64_t, 2, 2, 1, 2, 32, 3, 3, 3, 111, 64>(uint64_t *out, uint64_t *src, void *stream);
template void LaunchTLoad<0, int64_t, 1, 2, 1, 11, 32, 1, 3, 2, 93, 32>(int64_t *out, int64_t *src, void *stream);
template void LaunchTLoad<1, uint64_t, 1, 1, 1, 128, 3, 3, 3, 3, 128, 32>(uint64_t *out, uint64_t *src, void *stream);
template void LaunchTLoad<1, int64_t, 2, 2, 1, 32, 2, 3, 3, 3, 64, 111>(int64_t *out, int64_t *src, void *stream);
template void LaunchTLoad<1, uint64_t, 1, 2, 1, 32, 11, 1, 3, 2, 32, 93>(uint64_t *out, uint64_t *src, void *stream);
template void LaunchTLoad<4, uint16_t, 1, 1, 1, 256, 1024, 1, 1, 1, 256, 1024>(
    uint16_t *out, uint16_t *src, void *stream);
template void LaunchTLoad<4, float, 1, 1, 1, 49, 35, 1, 1, 1, 49, 35>(float *out, float *src, void *stream);
template void LaunchTLoad<4, int16_t, 1, 1, 1, 155, 250, 1, 1, 1, 752, 1000>(int16_t *out, int16_t *src, void *stream);
template void LaunchTLoad<4, int8_t, 1, 1, 1, 1023, 511, 1, 1, 1, 1024, 1024>(int8_t *out, int8_t *src, void *stream);
