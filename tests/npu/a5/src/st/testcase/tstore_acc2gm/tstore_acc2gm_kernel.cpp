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

template <typename T>
__aicore__ inline void DynGM2L1NZ2NZ(__cbuf__ T *dst, __gm__ T *src, unsigned TShape0, unsigned TShape1)
{
    uint16_t nBurst = 1;
    uint16_t lenBurst = TShape0 * TShape1 * sizeof(T);
    uint16_t srcGap = 0;
    uint16_t dstGap = 0;
    if (std::is_same<T, uint64_t>::value) {
        __cbuf__ uint32_t *dstTmp = reinterpret_cast<__cbuf__ uint32_t *>(dst);
        __gm__ uint32_t *srcTmp = reinterpret_cast<__gm__ uint32_t *>(src);
        copy_gm_to_cbuf_align_v2(dstTmp, srcTmp, 0, nBurst, lenBurst, 0, 0, false, 0, 0, 0);
    } else {
        copy_gm_to_cbuf_align_v2(dst, src, 0, nBurst, lenBurst, 0, 0, false, 0, 0, 0);
    }
}

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T>
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved)
{
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = TShape1;
    uint16_t blockCubeSize = std::is_same<GMT, int8_t>::value ? 32 : 16;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(GmShape0, blockCubeSize);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;   // 源操作数相邻ND矩阵起始地址间的偏移
    constexpr uint16_t dstNzNStride = 1;        // 目的NZ矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移
    constexpr uint16_t dstNzMatrixStride = 1;   // 目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移

    auto c0Size = 32 / sizeof(GMT);
    uint64_t loop1SrcStride = srcDValue * sizeof(GMT);
    uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(GMT);

    uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
    uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
    // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / c0_size
    uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(GMT) / c0Size);
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // mte2_nz_para[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // mte2_nz_para[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // mte2_nz_para[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);             // mte2_nz_para[15:0]
    set_mte2_nz_para(mte2NzPara);   // cce: store parameters for nd2nz DMA instructions

    copy_gm_to_cbuf_multi_nd2nz((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue,
                                loop4SrcStride, false);
}

template <typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
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

    using LeftTile = TileLeft<srcDataType, M, K, M, K>;
    using RightTile = TileRight<srcDataType, K, N, K, N>;
    using AccTile = Tile<Location::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

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

    if (quantMode == 0) {
        TSTORE(dstGlobal, cTile);
    } else if (quantMode == 1) {
        float tmp = 1;
        uint64_t preQuantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
        TSTORE(dstGlobal, cTile, preQuantScalar);
    }

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
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
    using AccTile = Tile<Location::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    using LeftTile = TileLeft<srcDataType, M, K, M, K>;
    using RightTile = TileRight<srcDataType, K, N, K, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

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

    TSTORE(dstGlobal, cTile);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
__aicore__ inline void TStoreAcc2gmNZ2NDFp(__gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1,
    __gm__ uint64_t *quantTensor)
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
    constexpr int alignScalingN = ((validN * sizeof(uint64_t) + 127) / 128) * 128 / sizeof(uint64_t);

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;

    int offset = 0;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileMatScalingData =
        Tile<Location::Mat, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = TileLeft<srcDataType, M, K, M, K>;
    using RightTile = TileRight<srcDataType, K, N, K, N>;
    using AccTile = Tile<Location::Acc, accDataType, Rows, Cols, BLayout::ColMajor, M, K, SLayout::RowMajor, 1024>;
    using ScalingTile =
        Tile<Location::Scaling, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatScalingData scalingMatTile(alignScalingN);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(scalingMatTile, 0x20000);

    uint32_t fbMatSize = alignScalingN * sizeof(uint64_t);
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    ScalingTile scalingTile(alignScalingN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();
    __cbuf__ uint64_t *srcScalingAddr = scalingMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    DynL1CopyIn<srcDataType, srcDataType>(srcAAddr, src0, M, K, M, K, 0, 0, 0);
    DynL1CopyIn<srcDataType, srcDataType>(srcBAddr, src1, K, N, K, N, 0, 0, 0);
    DynGM2L1NZ2NZ<uint64_t>(srcScalingAddr, quantTensor, 1, alignScalingN);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(scalingTile, scalingMatTile);

    TSTORE<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, cTile, scalingTile);

    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int floatType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2, int gShape3,
    int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int validM,
    int validN, int validK, int quantMode>
__global__ __aicore__ void LaunchTStoreAcc2gmNZ2ND(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (std::is_same_v<srcDataType, uint16_t> && std::is_same_v<dstDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2ND<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ half *>(out),
                reinterpret_cast < __gm__ half *>(src0),
                reinterpret_cast < __gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2ND<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ bfloat16_t *>(out),
                reinterpret_cast < __gm__ bfloat16_t *>(src0),
                reinterpret_cast < __gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (std::is_same_v<srcDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2ND<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ dstDataType *>(out),
                reinterpret_cast < __gm__ half *>(src0),
                reinterpret_cast < __gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2ND<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ dstDataType *>(out),
                reinterpret_cast < __gm__ bfloat16_t *>(src0),
                reinterpret_cast < __gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (!(std::is_same_v<srcDataType, uint16_t> || std::is_same_v<dstDataType, uint16_t>)) {
        if constexpr (std::is_same_v<srcDataType, int8_t>) {
            TStoreAcc2gmNZ2ND<int32_t,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ int32_t *>(out),
                reinterpret_cast < __gm__ srcDataType *>(src0),
                reinterpret_cast < __gm__ srcDataType *>(src1));
        } else if constexpr (std::is_same_v<srcDataType, float>) {
            TStoreAcc2gmNZ2ND<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ dstDataType *>(out),
                reinterpret_cast < __gm__ srcDataType *>(src0),
                reinterpret_cast < __gm__ srcDataType *>(src1));
        }
    } else if constexpr (quantMode == 1 && std::is_same_v<srcDataType, int8_t> && std::is_same_v<dstDataType, uint16_t>) {
        TStoreAcc2gmNZ2ND<int32_t,
            half,
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
            validK,
            quantMode>(reinterpret_cast < __gm__ half *>(out),
            reinterpret_cast < __gm__ srcDataType *>(src0),
            reinterpret_cast < __gm__ srcDataType *>(src1));
    }
}

template <int floatType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2, int gShape3,
    int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int validM,
    int validN, int validK, int quantMode>
__global__ __aicore__ void LaunchTStoreAcc2gmNZ2NZ(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    if constexpr (std::is_same_v<srcDataType, uint16_t> && std::is_same_v<dstDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2NZ<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ half *>(out),
                reinterpret_cast < __gm__ half *>(src0),
                reinterpret_cast < __gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2NZ<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ bfloat16_t *>(out),
                reinterpret_cast < __gm__ bfloat16_t *>(src0),
                reinterpret_cast < __gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (std::is_same_v<srcDataType, uint16_t>) {
        if constexpr (floatType == 0) {
            TStoreAcc2gmNZ2NZ<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ dstDataType *>(out),
                reinterpret_cast < __gm__ half *>(src0),
                reinterpret_cast < __gm__ half *>(src1));
        } else if constexpr (floatType == 1) {
            TStoreAcc2gmNZ2NZ<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ dstDataType *>(out),
                reinterpret_cast < __gm__ bfloat16_t *>(src0),
                reinterpret_cast < __gm__ bfloat16_t *>(src1));
        }
    } else if constexpr (!(std::is_same_v<srcDataType, uint16_t> || std::is_same_v<dstDataType, uint16_t>)) {
        if constexpr (std::is_same_v<srcDataType, int8_t>) {
            TStoreAcc2gmNZ2NZ<int32_t,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ int32_t *>(out),
                reinterpret_cast < __gm__ srcDataType *>(src0),
                reinterpret_cast < __gm__ srcDataType *>(src1));
        } else if constexpr (std::is_same_v<srcDataType, float>) {
            TStoreAcc2gmNZ2NZ<float,
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
                validK,
                quantMode>(reinterpret_cast < __gm__ dstDataType *>(out),
                reinterpret_cast < __gm__ srcDataType *>(src0),
                reinterpret_cast < __gm__ srcDataType *>(src1));
        }
    }
}

template <int format, int floatType, typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
void LaunchTStoreAcc2gm(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (format == 1) {
        LaunchTStoreAcc2gmNZ2ND<floatType,
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
            validK,
            quantMode><<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (format == 2) {
        LaunchTStoreAcc2gmNZ2NZ<floatType,
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
            validK,
            quantMode><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2, int gShape3,
    int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int validM,
    int validN, int validK, int quantMode>
__global__ __aicore__ void LaunchTStoreAcc2gmNZ2NDFp(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1,
    __gm__ uint8_t *quantTensor)
{
    TStoreAcc2gmNZ2NDFp<int32_t,
        half,
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
        validK,
        quantMode>(reinterpret_cast < __gm__ half *>(out),
        reinterpret_cast < __gm__ srcDataType *>(src0),
        reinterpret_cast < __gm__ srcDataType *>(src1),
        reinterpret_cast < __gm__ uint64_t *>(quantTensor));
}

template <typename dstDataType, typename srcDataType, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
    int validM, int validN, int validK, int quantMode>
void LaunchTStoreAcc2gmFp(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream)
{
    LaunchTStoreAcc2gmNZ2NDFp<
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
        validK,
        quantMode><<<1, nullptr, stream>>>(out, src0, src1, quantTensor);
}

template void LaunchTStoreAcc2gm<1, 0, float, float, 1, 1, 1, 128, 128, 1, 2, 3, 256, 128, 128, 128, 16, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 0, float, float, 1, 1, 1, 31, 32, 1, 2, 3, 31, 32, 31, 32, 15, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 0, float, uint16_t, 1, 1, 1, 65, 128, 1, 2, 3, 65, 128, 65, 128, 96, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 0, uint16_t, uint16_t, 1, 1, 1, 73, 64, 2, 2, 3, 73, 64, 73, 64, 32, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 1, float, uint16_t, 1, 1, 1, 13, 32, 2, 3, 7, 13, 32, 13, 32, 25, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<1, 1, uint16_t, uint16_t, 1, 1, 1, 100, 222, 5, 7, 7, 100, 222, 100, 222, 60, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<2, 0, float, float, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 25, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, float, float, 1, 2, 3, 16, 16, 1, 2, 3, 16, 16, 48, 32, 45, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, float, uint16_t, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 24, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, uint16_t, uint16_t, 2, 3, 6, 16, 16, 2, 3, 6, 16, 16, 96, 96, 23, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 1, float, uint16_t, 2, 3, 3, 16, 16, 2, 3, 3, 16, 16, 48, 96, 22, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 1, uint16_t, uint16_t, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<1, 0, int32_t, int8_t, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gm<2, 0, int32_t, int8_t, 2, 3, 4, 16, 16, 2, 3, 4, 16, 16, 64, 96, 30, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<2, 0, float, float, 3, 8, 4, 16, 8, 3, 8, 4, 16, 8, 64, 192, 43, 0>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gm<1, 0, uint16_t, int8_t, 1, 1, 1, 32, 32, 1, 2, 3, 32, 32, 32, 32, 32, 1>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmFp<uint16_t, int8_t, 1, 1, 1, 32, 32, 1, 2, 3, 32, 32, 32, 32, 32, 2>(
    uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream);
