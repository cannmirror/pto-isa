/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "engram_common.h"
#include <pto/npu/kernels/Pto_prefetch.hpp>

using namespace std;
using namespace pto;

#ifndef PTO_INLINE
#define PTO_INLINE __attribute__((always_inline)) inline
#endif

constexpr float kGateBiasF = 0.125f;

// Empty kernel to warm up cores
__global__ AICORE __attribute__((aiv)) void warmup_kernel()
{}

template <typename T, typename TIdx, int kTableRows, int kTableCols, int kNumHeads, int kEmbDim>
inline AICORE void runEngramBaseline(__gm__ T __out__ *output, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices,
                                     __gm__ T __in__ *hidden, __gm__ T __in__ *gate_weight)
{
    static_assert(kEmbDim >= 32 && (kEmbDim & (kEmbDim - 1)) == 0, "kEmbDim must be a power of 2 >= 32");
    static_assert(kNumHeads > 0 && kNumHeads <= 16, "kNumHeads must be in [1, 16]");
    static_assert(kTableCols >= kEmbDim, "kTableCols must be >= kEmbDim");

    constexpr int kIdxPad = ((kNumHeads * (int)sizeof(TIdx) + 31) / 32) * 32 / (int)sizeof(TIdx);
    constexpr int rowBytesF = kEmbDim * (int)sizeof(float);
    constexpr int rowAlF = ((rowBytesF + 31) / 32) * 32;

    constexpr int idxBytes = kIdxPad * (int)sizeof(TIdx);
    constexpr int idxAl = ((idxBytes + 31) / 32) * 32;
    constexpr int hidOff = idxAl;
    constexpr int gwOff = hidOff + rowAlF;
    constexpr int lookBase = gwOff + rowAlF;
    constexpr int afterLook = lookBase + kNumHeads * rowAlF;
    constexpr int aggFOff = afterLook;
    constexpr int tmpFOff = aggFOff + rowAlF;
    constexpr int gsFOff = tmpFOff + rowAlF;

    using TileIdxLoad = Tile<TileType::Vec, TIdx, 1, kIdxPad, BLayout::RowMajor, -1, -1>;
    using TileRowF = Tile<TileType::Vec, float, 1, kEmbDim, BLayout::RowMajor, -1, -1>;
    using TileLookF2D = Tile<TileType::Vec, float, kNumHeads, kEmbDim, BLayout::RowMajor, kNumHeads, kEmbDim>;
    using TileGSF = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1>;
    using TileGSF_CM = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1>;

    using EmbRowGMShape = pto::Shape<1, 1, 1, 1, kEmbDim>;
    using EmbRowGMStride = pto::Stride<1, 1, 1, kEmbDim, 1>;
    using GlobalEmbRow = GlobalTensor<float, EmbRowGMShape, EmbRowGMStride>;
    using IdxGMShape = pto::Shape<1, 1, 1, 1, kIdxPad>;
    using IdxGMStride = pto::Stride<1, 1, 1, kIdxPad, 1>;
    using GlobalIdx = GlobalTensor<TIdx, IdxGMShape, IdxGMStride>;
    using OutGMShape = pto::Shape<1, 1, 1, 1, kEmbDim>;
    using OutGMStride = pto::Stride<1, 1, 1, kEmbDim, 1>;
    using GlobalOut = GlobalTensor<float, OutGMShape, OutGMStride>;

    TileIdxLoad idxTile(1, kIdxPad);
    TileRowF hiddenF(1, kEmbDim);
    TileRowF gateWF(1, kEmbDim);
    TileRowF aggF(1, kEmbDim);
    TileRowF tmpF(1, kEmbDim);
    TileGSF gsF(1, 8);

    TASSIGN(idxTile, 0);
    TASSIGN(hiddenF, hidOff);
    TASSIGN(gateWF, gwOff);
    TASSIGN(aggF, aggFOff);
    TASSIGN(tmpF, tmpFOff);
    TASSIGN(gsF, gsFOff);

    GlobalIdx idxGM(indices);
    GlobalEmbRow hiddenGM(hidden);
    GlobalEmbRow gateWGM(gate_weight);

    TLOAD(idxTile, idxGM);
    TLOAD(hiddenF, hiddenGM);
    TLOAD(gateWF, gateWGM);

    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)((__ubuf__ uint8_t *)idxTile.data());

    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    for (int h = 0; h < kNumHeads; ++h) {
        TIdx rowIdx = idxPtr[h];
        GlobalEmbRow embGM(table + (int)rowIdx * kEmbDim);
        TileRowF headTile(1, kEmbDim);
        TASSIGN(headTile, lookBase + h * rowAlF);
        TLOAD(headTile, embGM);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    {
        TileLookF2D lookupF2D;
        TASSIGN(lookupF2D, lookBase);
        TCOLSUM(aggF, lookupF2D, lookupF2D, true);
        constexpr float invHeads = 1.0f / (float)kNumHeads;
        TMULS(aggF, aggF, invHeads);
    }

    TMUL(tmpF, hiddenF, gateWF);
    TROWSUM(gsF, tmpF, tmpF);
    TADDS(gsF, gsF, kGateBiasF);
    TMULS(gsF, gsF, -1.0f);
    TEXP(gsF, gsF);
    TADDS(gsF, gsF, 1.0f);
    TDIVS(gsF, 1.0f, gsF);

    {
        TileGSF_CM gsCM(8, 1);
        TASSIGN(gsCM, gsFOff);
        TROWEXPANDMUL(tmpF, aggF, gsCM);
    }
    TADD(tmpF, hiddenF, tmpF);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

    GlobalOut outGM(output);
    TSTORE(outGM, tmpF);
}

template <uint32_t kNumHeads, uint32_t kEmbDim>
__simt_vf__ AICORE LAUNCH_BOUND(1024) PTO_INLINE
    void simt_fused_engram_kernel(__gm__ float *__restrict__ gmOutput, __gm__ const float *__restrict__ gmTable,
                                  __gm__ const int32_t *__restrict__ gmIndices,
                                  __gm__ const float *__restrict__ gmHidden, __gm__ const float *__restrict__ gmGateW)
{
    static_assert(kEmbDim >= 128 && kEmbDim <= 1024, "kEmbDim must be 128, 256, 512, or 1024");
    static_assert((kEmbDim & (kEmbDim - 1)) == 0, "kEmbDim must be a power of 2");
    static_assert(kNumHeads > 0 && kNumHeads <= 16, "kNumHeads must be in [1, 16]");
    static_assert(kEmbDim / 32u <= 32u, "kEmbDim / 32 must not exceed max warp count");

    constexpr uint32_t kLanes = 32u;
    constexpr uint32_t kWarps = kEmbDim / kLanes;
    constexpr float kInvHeads = 1.0f / (float)kNumHeads;

    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();

    if (ty >= kWarps)
        return;

    const uint32_t col = ty * kLanes + tx;

    float h_val = gmHidden[col];
    float g_val = gmGateW[col];

    int32_t idx[kNumHeads];
#pragma unroll
    for (uint32_t h = 0; h < kNumHeads; ++h)
        idx[h] = gmIndices[h];

    float warp_dot = __builtin_cce_redux_add_f32(h_val * g_val);

    __ubuf__ float *scrBuf = (__ubuf__ float *)((__ubuf__ uint8_t *)0);
    scrBuf[ty] = warp_dot;
    __sync_workitems();

    float dot;
    if constexpr (kWarps <= 16) {
        dot = kGateBiasF;
#pragma unroll
        for (uint32_t w = 0; w < kWarps; ++w)
            dot += scrBuf[w];
    } else {
        static_assert(kWarps == kLanes, "hierarchical reduction requires kWarps == kLanes");
        if (ty == 0) {
            float partial = scrBuf[tx];
            float total = __builtin_cce_redux_add_f32(partial);
            scrBuf[0] = total + kGateBiasF;
        }
        __sync_workitems();
        dot = scrBuf[0];
    }

    float gate = 1.0f / (1.0f + __builtin_cce_expf(-dot));

    float agg = gmTable[(uint32_t)idx[0] * kEmbDim + col];
#pragma unroll
    for (uint32_t h = 1; h < kNumHeads; ++h)
        agg += gmTable[(uint32_t)idx[h] * kEmbDim + col];

    gmOutput[col] = h_val + (gate * kInvHeads) * agg;
}

template <uint32_t kNumHeads, uint32_t kEmbDim>
__tf__ AICORE void FusedEngramImpl(__gm__ float *__restrict__ gmOutput, __gm__ const float *__restrict__ gmTable,
                                   __gm__ const int32_t *__restrict__ gmIndices,
                                   __gm__ const float *__restrict__ gmHidden, __gm__ const float *__restrict__ gmGateW)
{
    static_assert(kEmbDim >= 128 && kEmbDim <= 1024, "kEmbDim must be 128, 256, 512, or 1024");
    constexpr uint32_t kWarps = kEmbDim / 32u;
    cce::async_invoke<simt_fused_engram_kernel<kNumHeads, kEmbDim>>(cce::dim3{32, kWarps}, gmOutput, gmTable, gmIndices,
                                                                    gmHidden, gmGateW);
}

template <typename TIdx, int kTableRows, int kTableCols, int kNumHeads, int kEmbDim>
inline AICORE void runEngramFused(__gm__ float __out__ *output, __gm__ float __in__ *table, __gm__ TIdx __in__ *indices,
                                  __gm__ float __in__ *hidden, __gm__ float __in__ *gate_weight)
{
    __gm__ const int32_t *idxPtr = reinterpret_cast<__gm__ const int32_t *>(indices);
    FusedEngramImpl<(uint32_t)kNumHeads, (uint32_t)kEmbDim>(output, table, idxPtr, hidden, gate_weight);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ AICORE void runEngram_baseline_float_128x128_8x128(__gm__ float *out, __gm__ float *table,
                                                                         __gm__ int32_t *idx, __gm__ float *hid,
                                                                         __gm__ float *gw)
{
    runEngramBaseline<float, int32_t, 128, 128, 8, 128>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_baseline_float_256x256_8x256(__gm__ float *out, __gm__ float *table,
                                                                         __gm__ int32_t *idx, __gm__ float *hid,
                                                                         __gm__ float *gw)
{
    runEngramBaseline<float, int32_t, 256, 256, 8, 256>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_baseline_float_512x512_8x512(__gm__ float *out, __gm__ float *table,
                                                                         __gm__ int32_t *idx, __gm__ float *hid,
                                                                         __gm__ float *gw)
{
    runEngramBaseline<float, int32_t, 512, 512, 8, 512>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_baseline_float_1024x1024_8x1024(__gm__ float *out, __gm__ float *table,
                                                                            __gm__ int32_t *idx, __gm__ float *hid,
                                                                            __gm__ float *gw)
{
    runEngramBaseline<float, int32_t, 1024, 1024, 8, 1024>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_fused_float_128x128_8x128(__gm__ float *out, __gm__ float *table,
                                                                      __gm__ int32_t *idx, __gm__ float *hid,
                                                                      __gm__ float *gw)
{
    runEngramFused<int32_t, 128, 128, 8, 128>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_fused_float_256x256_8x256(__gm__ float *out, __gm__ float *table,
                                                                      __gm__ int32_t *idx, __gm__ float *hid,
                                                                      __gm__ float *gw)
{
    runEngramFused<int32_t, 256, 256, 8, 256>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_fused_float_512x512_8x512(__gm__ float *out, __gm__ float *table,
                                                                      __gm__ int32_t *idx, __gm__ float *hid,
                                                                      __gm__ float *gw)
{
    runEngramFused<int32_t, 512, 512, 8, 512>(out, table, idx, hid, gw);
}

extern "C" __global__ AICORE void runEngram_fused_float_1024x1024_8x1024(__gm__ float *out, __gm__ float *table,
                                                                         __gm__ int32_t *idx, __gm__ float *hid,
                                                                         __gm__ float *gw)
{
    runEngramFused<int32_t, 1024, 1024, 8, 1024>(out, table, idx, hid, gw);
}

template <typename TIdx, int kTableRows, int kTableCols, int kNumHeads, int kEmbDim>
void LaunchEngramBaseline(float *out, float *table, TIdx *indices, float *hid, float *gw, void *stream);

template <typename TIdx, int kTableRows, int kTableCols, int kNumHeads, int kEmbDim>
void LaunchEngramFused(float *out, float *table, TIdx *indices, float *hid, float *gw, void *stream);

template <>
void LaunchEngramBaseline<int32_t, 128, 128, 8, 128>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                     void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_baseline_float_128x128_8x128<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramBaseline<int32_t, 256, 256, 8, 256>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                     void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_baseline_float_256x256_8x256<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramBaseline<int32_t, 512, 512, 8, 512>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                     void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_baseline_float_512x512_8x512<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramBaseline<int32_t, 1024, 1024, 8, 1024>(float *out, float *table, int32_t *indices, float *hid,
                                                        float *gw, void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_baseline_float_1024x1024_8x1024<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramFused<int32_t, 128, 128, 8, 128>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                  void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_fused_float_128x128_8x128<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramFused<int32_t, 256, 256, 8, 256>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                  void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_fused_float_256x256_8x256<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramFused<int32_t, 512, 512, 8, 512>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                  void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_fused_float_512x512_8x512<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}

template <>
void LaunchEngramFused<int32_t, 1024, 1024, 8, 1024>(float *out, float *table, int32_t *indices, float *hid, float *gw,
                                                     void *stream)
{
    warmup_kernel<<<64, nullptr, stream>>>();
    runEngram_fused_float_1024x1024_8x1024<<<1, nullptr, stream>>>(out, table, indices, hid, gw);
}
