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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;

// Wrapper types for FP8 testing - use int8_t storage but distinguish types
struct fp8_e4m3_wrapper {
    int8_t value;
    operator int8_t() const
    {
        return value;
    }
    operator float() const
    {
        return static_cast<float>(value);
    }
};
struct fp8_e5m2_wrapper {
    int8_t value;
    operator int8_t() const
    {
        return value;
    }
    operator float() const
    {
        return static_cast<float>(value);
    }
};
struct hifloat8_wrapper {
    int8_t value;
    operator int8_t() const
    {
        return value;
    }
    operator float() const
    {
        return static_cast<float>(value);
    }
};

template <typename T, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
          int kValidCols_ = kTCols_>
__global__ AICORE void runTCVT(__gm__ T *out, __gm__ S *src)
{
    using DynShapeDim4 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim4 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src = GlobalTensor<S, DynShapeDim4, DynStridDim4>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    // Use dynamic tiles when valid dimensions differ from tile dimensions
    constexpr bool useDynamicTile = (kValidRows_ != kTRows_) || (kValidCols_ != kTCols_);

    using TileDataSrc =
        std::conditional_t<useDynamicTile, Tile<TileType::Vec, S, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>,
                           Tile<TileType::Vec, S, kTRows_, kTCols_, BLayout::RowMajor>>;
    using TileDataDst =
        std::conditional_t<useDynamicTile, Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>,
                           Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>>;

    TileDataSrc srcTile;
    TileDataDst dstTile;

    if constexpr (useDynamicTile) {
        srcTile = TileDataSrc(kValidRows_, kValidCols_);
        dstTile = TileDataDst(kValidRows_, kValidCols_);
    }

    TASSIGN(srcTile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x20000 + 0x400 * block_idx);

    GlobalData_src srcGlobal(src);

    GlobalData_dst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // FP16->H8 conversion only supports ROUND_A or ROUND_H, use CAST_ROUND instead of CAST_RINT
    if constexpr (std::is_same_v<T, hifloat8_t> && std::is_same_v<S, half>) {
        TCVT(dstTile, srcTile, RoundMode::CAST_ROUND);
    } else {
        TCVT(dstTile, srcTile, RoundMode::CAST_RINT);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
          int kValidCols_ = kTCols_>
void launchTCVT(D *dst, S *src, void *stream)
{
    // Map aclFloat16 to half for kernel execution
    using DstType = std::conditional_t<
        std::is_same_v<D, aclFloat16>, half,
        std::conditional_t<std::is_same_v<D, fp8_e4m3_wrapper>, float8_e4m3_t,
                           std::conditional_t<std::is_same_v<D, fp8_e5m2_wrapper>, float8_e5m2_t,
                                              std::conditional_t<std::is_same_v<D, hifloat8_wrapper>, hifloat8_t, D>>>>;
    using SrcType = std::conditional_t<
        std::is_same_v<S, aclFloat16>, half,
        std::conditional_t<std::is_same_v<S, fp8_e4m3_wrapper>, float8_e4m3_t,
                           std::conditional_t<std::is_same_v<S, fp8_e5m2_wrapper>, float8_e5m2_t,
                                              std::conditional_t<std::is_same_v<S, hifloat8_wrapper>, hifloat8_t, S>>>>;

    runTCVT<DstType, SrcType, kGRows_, kGCols_, kTRows_, kTCols_, kValidRows_, kValidCols_>
        <<<1, nullptr, stream>>>(reinterpret_cast<DstType *>(dst), reinterpret_cast<SrcType *>(src));
}

// Macro to generate template instantiations for all shapes for a given type pair
#define INSTANTIATE_TCVT(dst_type, src_type)                                                                           \
    template void launchTCVT<dst_type, src_type, 1, 128, 1, 128>(dst_type * dst, src_type * src, void *stream);        \
    template void launchTCVT<dst_type, src_type, 2, 64, 2, 64>(dst_type * dst, src_type * src, void *stream);          \
    template void launchTCVT<dst_type, src_type, 4, 32, 4, 32>(dst_type * dst, src_type * src, void *stream);          \
    template void launchTCVT<dst_type, src_type, 2, 128, 2, 128>(dst_type * dst, src_type * src, void *stream);        \
    template void launchTCVT<dst_type, src_type, 4, 128, 4, 128, 4, 65>(dst_type * dst, src_type * src, void *stream); \
    template void launchTCVT<dst_type, src_type, 4, 256, 4, 256, 4, 200>(dst_type * dst, src_type * src,               \
                                                                         void *stream);                                \
    template void launchTCVT<dst_type, src_type, 1, 256, 1, 256, 1, 129>(dst_type * dst, src_type * src, void *stream);

// FP32 Source → fp16, bf16, int16, int32, int64, fp8 variants
INSTANTIATE_TCVT(aclFloat16, float)
INSTANTIATE_TCVT(bfloat16_t, float)
INSTANTIATE_TCVT(int16_t, float)
INSTANTIATE_TCVT(int32_t, float)
INSTANTIATE_TCVT(int64_t, float)
INSTANTIATE_TCVT(fp8_e4m3_wrapper, float)
INSTANTIATE_TCVT(fp8_e5m2_wrapper, float)
INSTANTIATE_TCVT(hifloat8_wrapper, float)
INSTANTIATE_TCVT(float, float)

// FP16 Source → fp32, int32, int16, int8, uint8, h8
INSTANTIATE_TCVT(float, aclFloat16)
INSTANTIATE_TCVT(int32_t, aclFloat16)
INSTANTIATE_TCVT(int16_t, aclFloat16)
INSTANTIATE_TCVT(int8_t, aclFloat16)
INSTANTIATE_TCVT(uint8_t, aclFloat16)
INSTANTIATE_TCVT(hifloat8_wrapper, aclFloat16)

// BF16 Source → fp32, int32, half
INSTANTIATE_TCVT(float, bfloat16_t)
INSTANTIATE_TCVT(int32_t, bfloat16_t)
// INSTANTIATE_TCVT(aclFloat16, bfloat16_t)

// U8 Source → half, uint16
INSTANTIATE_TCVT(aclFloat16, uint8_t)
// INSTANTIATE_TCVT(uint16_t, uint8_t)

// I8 Source → half, int16, int32
INSTANTIATE_TCVT(aclFloat16, int8_t)
INSTANTIATE_TCVT(int16_t, int8_t)
INSTANTIATE_TCVT(int32_t, int8_t)

// I16 Source → uint8, half, float, uint32, int32
INSTANTIATE_TCVT(uint8_t, int16_t)
INSTANTIATE_TCVT(aclFloat16, int16_t)
INSTANTIATE_TCVT(float, int16_t)
INSTANTIATE_TCVT(uint32_t, int16_t)
INSTANTIATE_TCVT(int32_t, int16_t)

// I32 Source → float, int16, uint16, int64, uint8
INSTANTIATE_TCVT(float, int32_t)
INSTANTIATE_TCVT(int16_t, int32_t)
// INSTANTIATE_TCVT(uint16_t, int32_t)
INSTANTIATE_TCVT(int64_t, int32_t)
INSTANTIATE_TCVT(uint8_t, int32_t)

// U32 Source → uint8, uint16, int16
INSTANTIATE_TCVT(uint8_t, uint32_t)
// INSTANTIATE_TCVT(uint16_t, uint32_t)
INSTANTIATE_TCVT(int16_t, uint32_t)

// I64 Source → float, int32
INSTANTIATE_TCVT(float, int64_t)
INSTANTIATE_TCVT(int32_t, int64_t)

// FP8 Source → float
INSTANTIATE_TCVT(float, fp8_e4m3_wrapper)
INSTANTIATE_TCVT(float, fp8_e5m2_wrapper)
INSTANTIATE_TCVT(float, hifloat8_wrapper)