/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;

template <typename T, uint32_t Rows, uint32_t Cols>
AICORE void runTmovUb2l1( __gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    constexpr uint32_t c0Size = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));
    using OutShapeDim5 = pto::Shape<1, Cols / c0Size, 1, FRACTAL_NZ_ROW, c0Size>;
    using OutStridDim5 = pto::Stride<Cols / c0Size * 1 * FRACTAL_NZ_ROW * c0Size, 1 * FRACTAL_NZ_ROW * c0Size,
        FRACTAL_NZ_ROW * c0Size, FRACTAL_NZ_ROW, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5, Layout::NZ>;

    using SrcTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using DstTileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using MatTileData = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>; 

    SrcTileData srcTile(Rows, Cols);
    TmpTileData tmpTile(Rows, Cols);
    DstTileData dstTile(Rows, Cols);
    MatTileData matTile(Rows, Cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(matTile, 0x0);

    SrcGlobalData srcGlobal(src);
    OutGlobalData dstGlobal(out);
    uint8_t syncId = 0;
    uint8_t eventIdNum = 16;
    uint16_t blockLen = Rows * Cols * sizeof(T) / BLOCK_BYTE_SIZE;
    __cbuf__ T *srcMatAddr = matTile.data();
    __ubuf__ T *dstUbAddr = dstTile.data();

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal); //gm->ub
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMOV(tmpTile, srcTile);  //ub2Ub
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TMOV(matTile, tmpTile);  //ub2l1
    set_intra_block(PIPE_MTE3, syncId);
#endif

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_MTE1, syncId); // MTE1 等待V侧MTE3流水
    wait_intra_block(PIPE_MTE1, syncId + eventIdNum);

    set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 0, 1, blockLen, 0, 0);  // move to vector0
    copy_cbuf_to_ubuf(
        (__ubuf__ void *)dstUbAddr, (__cbuf__ void *)srcMatAddr, 1, 1, blockLen, 0, 0);  // move to vector1
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);

    set_intra_block(PIPE_MTE1, syncId);  //ub2l1 告诉V侧已经搬完,C侧L12UB MTE1流水
    set_intra_block(PIPE_MTE1, syncId + eventIdNum);
#endif

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    TSTORE(dstGlobal, dstTile);  // UB -> GM : AIV
#endif
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTmovUb2l1_1(__gm__ uint64_t *out, __gm__ uint64_t *src)
{
    constexpr uint32_t rows = 16;
    constexpr uint32_t cols = 32;
    runTmovUb2l1<half, rows, cols>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

template <int32_t testKey>
void launchTmovUb2l1(uint64_t *out, uint64_t *src, void* stream){
    cout << "launchTmovUb2l1 start!" << endl;
    if constexpr (testKey == 1) {
        launchTmovUb2l1_1<<<1, nullptr, stream>>>(out, src);
    }
    cout << "launchTmovUb2l1 end!" << endl;
}

template void launchTmovUb2l1<1>(uint64_t *out, uint64_t *src, void* stream);
