/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

namespace TQuantTest {

// Quantize fp32 tile to fp8 (e4m3) and exponent-only (e8m0).
// Pad columns to multiples of 32 using min fill to avoid reading garbage.
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuant(__gm__ uint8_t __out__ *out_e8m0, __gm__ uint8_t __out__ *out_fp8,
                                 __gm__ float __in__ *src)
{
    // pad each row to multiple of 32 elements
    constexpr int paddedCols = PTO_CEIL(validCols, 32);
    constexpr int groupedCols_flattened = validRows * (paddedCols / 32);
    constexpr int groupedCols_valid = paddedCols / 32;
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, groupedCols_flattened>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstFP8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;
    using DstE8Tile = Tile<TileType::Vec, uint8_t, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1,
                           SLayout::NoneBox, 512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, uint8_t, validRows, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                            512, PadValue::Zero>;
    using MaxTile = Tile<TileType::Vec, float, 1, groupedCols_flattened, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    SrcTile scalingTile(validRows, validCols);
    DstFP8Tile fp8Tile(validRows, paddedCols);
    DstE8Tile e8Tile(1, groupedCols_flattened);
    MaxTile maxPerGpTile(1, groupedCols_flattened);

    SrcGlobal srcGlobal(src);
    DstE8Global e8Global(out_e8m0);
    DstFP8Global fp8Global(out_fp8);

    TASSIGN(srcTile, 0x0);          // 64  KB = 0x10000
    TASSIGN(maxPerGpTile, 0x10100); // 1.5 KB = 0x1800 (Max and Scaling can overlap)
    TASSIGN(scalingTile, 0x21820);  // 3 KB   = 0xC00
    TASSIGN(e8Tile, 0x30100);       // 0.5 KB = 0x600
    TASSIGN(fp8Tile, 0x38160);      // 16  KB = 0x4000
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<SrcTile, DstE8Tile, DstFP8Tile, MaxTile, mode>(srcTile, e8Tile, fp8Tile, maxPerGpTile, scalingTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(e8Global, e8Tile);
    TSTORE(fp8Global, fp8Tile);
}

template <int validRows, int validCols, int mode>
void LaunchTQuant(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream)
{
    runTQuant<validRows, validCols, mode><<<1, nullptr, stream>>>(out_e8m0, out_fp8, src);
}

} // namespace TQuantTest

template void TQuantTest::LaunchTQuant<32, 32, 0>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);
template void TQuantTest::LaunchTQuant<32, 64, 0>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);
template void TQuantTest::LaunchTQuant<64, 128, 0>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);
template void TQuantTest::LaunchTQuant<128, 128, 0>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);
template void TQuantTest::LaunchTQuant<32, 64, 1>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);
template void TQuantTest::LaunchTQuant<64, 128, 1>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);
template void TQuantTest::LaunchTQuant<128, 128, 1>(uint8_t *out_e8m0, uint8_t *out_fp8, float *src, void *stream);