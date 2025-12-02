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

using namespace pto;

constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;
template <typename T>
__aicore__ inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

__aicore__ inline unsigned CalcLinearOffset(unsigned GmShape1, unsigned Offset0, unsigned Offset1)
{
    return Offset1 + Offset0 * GmShape1;
}

template <typename GMT, typename L1T>
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1,
    unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved)
{  // ND2NZ
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = GmShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(TShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 0;

    if constexpr (std::is_same<GMT, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T *)dst,
            (__gm__ GMT *)src,
            0 /*sid*/,
            ndNum,
            nValue,
            dValue,
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, half>::value || std::is_same<GMT, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ L1T *)dst,
            (__gm__ GMT *)src,
            0 /*sid*/,
            ndNum,
            nValue,
            dValue,
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride);
    }

    if constexpr (std::is_same<GMT, float>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ L1T *)dst,
            (__gm__ GMT *)src,
            0 /*sid*/,
            ndNum,
            nValue,
            dValue,
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride);
    }
}

template <AtomicType atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0,
    int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2,
    int gWholeShape3, int gWholeShape4, int validM, int validN, int validK>
__aicore__ inline void TStoreAcc2gmNZ2ND(__gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        gWholeShape4,
        1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;

    constexpr int Rows = M;
    constexpr int Cols = N;
    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;

    int offset = 0;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = Tile<Location::Left, srcDataType, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, K, N>;
    using AccTile = Tile<Location::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    DynL1CopyIn<srcDataType, srcDataType>(srcAAddr, src0, validM, validK, validM, validK, 0, 0, 0);
    DynL1CopyIn<srcDataType, srcDataType>(srcBAddr, src1, validK, validN, validK, validN, 0, 0, 0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE<AccTile, GlobalDataOut, atomicType>(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <AtomicType atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0,
    int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2,
    int gWholeShape3, int gWholeShape4, int validM, int validN, int validK>
__aicore__ inline void TStoreAcc2gmNZ2NZ(__gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape2 * gWholeShape3 * gWholeShape4,
        gWholeShape3 * gWholeShape4,
        gWholeShape4,
        1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;

    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;
    int validRow = gShape2 * gShape3;
    int validCol = gShape0 * gShape1 * gShape4;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;

    int offset = 0;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = Tile<Location::Left, srcDataType, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, K, N>;
    using AccTile = Tile<Location::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validRow, validCol);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    DynL1CopyIn<srcDataType, srcDataType>(srcAAddr, src0, validM, validK, validM, validK, 0, 0, 0);
    DynL1CopyIn<srcDataType, srcDataType>(srcBAddr, src1, validK, validN, validK, validN, 0, 0, 0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE<AccTile, GlobalDataOut, atomicType>(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <int floatType, AtomicType atomicType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
    int gWholeShape4, int validM, int validN, int validK>
__global__ __aicore__ void LaunchTStoreAcc2gmNZ2ND(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (std::is_same_v<srcDataType, uint16_t> && std::is_same_v<dstDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2ND<atomicType,
                float,
                half,
                half,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ half *>(out),
                reinterpret_cast<__gm__ half *>(src0),
                reinterpret_cast<__gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2ND<atomicType,
                float,
                bfloat16_t,
                bfloat16_t,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ bfloat16_t *>(out),
                reinterpret_cast<__gm__ bfloat16_t *>(src0),
                reinterpret_cast<__gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (std::is_same_v<srcDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2ND<atomicType,
                float,
                dstDataType,
                half,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ half *>(src0),
                reinterpret_cast<__gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2ND<atomicType,
                float,
                dstDataType,
                bfloat16_t,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ bfloat16_t *>(src0),
                reinterpret_cast<__gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (!(std::is_same_v<srcDataType, uint16_t> || std::is_same_v<dstDataType, uint16_t>)) {
        if constexpr (std::is_same_v<srcDataType, int8_t>) {
            TStoreAcc2gmNZ2ND<atomicType,
                int32_t,
                dstDataType,
                srcDataType,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ srcDataType *>(src0),
                reinterpret_cast<__gm__ srcDataType *>(src1));
        } else if constexpr (std::is_same_v<srcDataType, float>) {
            TStoreAcc2gmNZ2ND<atomicType,
                float,
                dstDataType,
                srcDataType,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ srcDataType *>(src0),
                reinterpret_cast<__gm__ srcDataType *>(src1));
        }
    }
}

template <int floatType, AtomicType atomicType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
    int gWholeShape4, int validM, int validN, int validK>
__global__ __aicore__ void LaunchTStoreAcc2gmNZ2NZ(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (std::is_same_v<srcDataType, uint16_t> && std::is_same_v<dstDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2NZ<atomicType,
                float,
                half,
                half,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ half *>(out),
                reinterpret_cast<__gm__ half *>(src0),
                reinterpret_cast<__gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2NZ<atomicType,
                float,
                bfloat16_t,
                bfloat16_t,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ bfloat16_t *>(out),
                reinterpret_cast<__gm__ bfloat16_t *>(src0),
                reinterpret_cast<__gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (std::is_same_v<srcDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2NZ<atomicType,
                float,
                dstDataType,
                half,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ half *>(src0),
                reinterpret_cast<__gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2NZ<atomicType,
                float,
                dstDataType,
                bfloat16_t,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ bfloat16_t *>(src0),
                reinterpret_cast<__gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (!(std::is_same_v<srcDataType, uint16_t> || std::is_same_v<dstDataType, uint16_t>)) {
        if constexpr (std::is_same_v<srcDataType, int8_t>) {
            TStoreAcc2gmNZ2NZ<atomicType,
                int32_t,
                dstDataType,
                srcDataType,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ int32_t *>(out),
                reinterpret_cast<__gm__ srcDataType *>(src0),
                reinterpret_cast<__gm__ srcDataType *>(src1));
        } else if constexpr (std::is_same_v<srcDataType, float>) {
            TStoreAcc2gmNZ2NZ<atomicType,
                float,
                dstDataType,
                srcDataType,
                gShape0,
                gShape1,
                gShape2,
                gShape3,
                gShape4,
                gWholeShape0,
                gWholeShape1,
                gWholeShape2,
                gWholeShape3,
                gWholeShape4,
                validM,
                validN,
                validK>(reinterpret_cast<__gm__ dstDataType *>(out),
                reinterpret_cast<__gm__ srcDataType *>(src0),
                reinterpret_cast<__gm__ srcDataType *>(src1));
        }
    }
}

template <int format, int floatType, int atomicType, typename dstDataType, typename srcDataType, int gShape0,
    int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2,
    int gWholeShape3, int gWholeShape4, int validM, int validN, int validK>
void LaunchTStoreAcc2gm(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (format == 1) {
        LaunchTStoreAcc2gmNZ2ND<floatType,
            atomicTypeEnum,
            dstDataType,
            srcDataType,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4,
            validM,
            validN,
            validK><<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (format == 2) {
        LaunchTStoreAcc2gmNZ2NZ<floatType,
            atomicTypeEnum,
            dstDataType,
            srcDataType,
            gShape0,
            gShape1,
            gShape2,
            gShape3,
            gShape4,
            gWholeShape0,
            gWholeShape1,
            gWholeShape2,
            gWholeShape3,
            gWholeShape4,
            validM,
            validN,
            validK><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTStoreAcc2gm<1, 0, 1, float, float, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, 61>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 0, 0, float, float, 1, 1, 1, 31, 32, 1, 2, 3, 31, 32, 31, 32, 126>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 0, 0, float, uint16_t, 1, 1, 1, 65, 128, 1, 2, 3, 65, 128, 65, 128, 96>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 0, 0, uint16_t, uint16_t, 1, 1, 1, 73, 64, 2, 2, 3, 73, 64, 73, 64, 32>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 1, 1, float, uint16_t, 1, 1, 1, 13, 32, 2, 3, 7, 13, 32, 13, 32, 25>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 1, 1, uint16_t, uint16_t, 1, 1, 1, 100, 222, 5, 7, 7, 100, 222, 100, 222, 60>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<2, 0, 0, float, float, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 25>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, 0, float, float, 1, 2, 3, 16, 16, 1, 2, 3, 16, 16, 48, 32, 45>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, 0, float, uint16_t, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 24>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, 1, uint16_t, uint16_t, 2, 3, 6, 16, 16, 2, 3, 6, 16, 16, 96, 96, 23>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 1, 0, float, uint16_t, 2, 3, 3, 16, 16, 2, 3, 3, 16, 16, 48, 96, 22>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 1, 1, uint16_t, uint16_t, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<1, 0, 1, int32_t, int8_t, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, 1, int32_t, int8_t, 2, 3, 4, 16, 16, 2, 3, 4, 16, 16, 64, 96, 30>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<2, 0, 0, float, float, 3, 8, 4, 16, 8, 3, 8, 4, 16, 8, 64, 192, 43>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);