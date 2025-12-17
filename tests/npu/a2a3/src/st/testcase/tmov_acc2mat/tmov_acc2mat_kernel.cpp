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

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
using CType = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

template <typename GlobalData, typename TileData>
AICORE inline void MatCopyOut(GlobalData &dst, TileData &src, int gShape0, int gShape1, int gShape2, int gShape3,
    int gShape4, int gStride0, int gStride1)
{
    int ValidRow = src.GetValidRow();
    int ValidCol = src.GetValidCol();

    typename GlobalData::DType *dstAddr = dst.data();
    __cbuf__ typename TileData::DType *srcAddr = src.data();
    typename GlobalData::DType *dstAddrP = dstAddr;
    __cbuf__ typename TileData::DType *srcAddrP = srcAddr;

    uint16_t nBurst = gShape1;
    uint16_t lenBurst = ValidRow;
    uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType) / BLOCK_BYTE_SIZE;
    uint32_t l1Gap = TileData::Rows - ValidRow;

    uint64_t tileStride = TileData::Rows * gShape1 * gShape4;
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * tileStride;
        dstAddrP = dstAddr + i * gStride0;
        copy_cbuf_to_gm(dstAddrP, srcAddrP, 0, nBurst, lenBurst, l1Gap, gmGap);
    }
}

template <typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN>
AICORE inline void runMATMUL(__gm__ aType *src0, __gm__ bType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<aType,
        pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType,
        pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    using LeftTile = TileLeft<aType, M, K, validM, validK>;
    using RightTile = TileRight<bType, K, N, validK, validN>;
    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename aType, typename bType, typename fbType, int M, int K, int N, int validM, int validK, int validN>
AICORE inline void runMATMULFB(__gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType,
        pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType,
        pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<fbType,
        pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<TileType::Mat, aType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, bType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<TileType::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(fbMatTile, 0x20000);
    __cbuf__ fbType *srcFbAddr = fbMatTile.data();

    using LeftTile = TileLeft<aType, M, K, validM, validK>;
    using RightTile = TileRight<bType, K, N, validK, validN>;
    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(fbMatTile, src2Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN>
__global__ AICORE void runTMOV_nz2nz(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(512, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(validM, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(validN, sGCols_);

    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    constexpr uint16_t gStride0 = kGCols_ * kGRows_ * sGCols_ * sGRows_;
    constexpr uint16_t gStride1 = kGRows_ * sGCols_ * sGRows_;
    using DynStridDim5 = pto::Stride<gStride0, gStride1, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    using DstTileData = Tile<TileType::Mat, outType, M, N, BLayout::ColMajor, validM, validN, SLayout::RowMajor, 512>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

    TMOV(dstTileData, cTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    MatCopyOut<GlobalDataOut, DstTileData>(
        dstGlobal, dstTileData, 1, kGCols_, kGRows_, sGRows_, sGCols_, gStride0, gStride1);
    out = dstGlobal.data();
}

template <typename outType, typename aType, typename bType, typename fbType, int M, int K, int N, int validM,
    int validK, int validN>
__global__ AICORE void runVectorQuantTMOV_nz2nz(
    __gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(512, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(validM, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(validN, sGCols_);

    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    constexpr uint16_t gStride0 = kGCols_ * kGRows_ * sGCols_ * sGRows_;
    constexpr uint16_t gStride1 = kGRows_ * sGCols_ * sGRows_;
    using DynStridDim5 = pto::Stride<gStride0, gStride1, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    GlobalDataOut dstGlobal(out);
    using TileMatFbData = Tile<TileType::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;

    TileMatFbData fbMatTile;
    TASSIGN(fbMatTile, 0x20000);

    runMATMULFB<aType, bType, fbType, M, K, N, validM, validK, validN>(src0, src1, src2);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    using FbTile = Tile<TileType::Scaling, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    FbTile fbTile;
    TASSIGN(fbTile, 0x0);

    using DstTileData = Tile<TileType::Mat, outType, M, N, BLayout::ColMajor, validM, validN, SLayout::RowMajor, 512>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

    TMOV(fbTile, fbMatTile);  // L1-> FB1
    TMOV_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    MatCopyOut<GlobalDataOut, DstTileData>(
        dstGlobal, dstTileData, 1, kGCols_, kGRows_, sGRows_, sGCols_, gStride0, gStride1);
    out = dstGlobal.data();
}

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN>
__global__ AICORE void runScalarQuantTMOV_nz2nz(
    __gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, float scalar)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(512, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(validM, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(validN, sGCols_);

    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    constexpr uint16_t gStride0 = kGCols_ * kGRows_ * sGCols_ * sGRows_;
    constexpr uint16_t gStride1 = kGRows_ * sGCols_ * sGRows_;
    using DynStridDim5 = pto::Stride<gStride0, gStride1, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    using DstTileData = Tile<TileType::Mat, outType, M, N, BLayout::ColMajor, validM, validN, SLayout::RowMajor, 512>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);
    uint64_t preQuantScalar = 0;
    if (std::is_same<outType, int16_t>::value) {
        int value = static_cast<int>(scalar);
        uint8_t bits = static_cast<uint8_t>(value - 1);
        uint8_t low4 = bits & 0x0F;
        preQuantScalar |= (static_cast<uint64_t>(low4) << 32);
    } else {
        preQuantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
        if (sizeof(outType) == 1) {
            constexpr bool sign = (std::is_same_v<typename DstTileData::DType, int8_t>) ? true : false;
            preQuantScalar = (preQuantScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
        }
    }
    TMOV<DstTileData, AccTile>(dstTileData, cTile, preQuantScalar);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    MatCopyOut<GlobalDataOut, DstTileData>(
        dstGlobal, dstTileData, 1, kGCols_, kGRows_, sGRows_, sGCols_, gStride0, gStride1);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void launchTMOVAcc2MatNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMOV_nz2nz<half, half, half, 64, 128, 128, 64, 128, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 2) {
        runTMOV_nz2nz<bfloat16_t, half, half, 48, 128, 64, 48, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}
template void launchTMOVAcc2MatNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVAcc2MatNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVAcc2MatSCQuantNz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runScalarQuantTMOV_nz2nz<half, int8_t, int8_t, 48, 64, 128, 48, 64, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        runScalarQuantTMOV_nz2nz<int8_t, half, half, 48, 64, 128, 48, 64, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1), 2);
    } else if constexpr (tilingKey == 3) {
        runScalarQuantTMOV_nz2nz<int8_t, int8_t, int8_t, 48, 64, 128, 48, 64, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 2);
    } else if constexpr (tilingKey == 4) {
        runScalarQuantTMOV_nz2nz<uint8_t, int8_t, int8_t, 48, 64, 128, 48, 64, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 1);
    } else if constexpr (tilingKey == 5) {
        runScalarQuantTMOV_nz2nz<int16_t, int8_t, int8_t, 48, 64, 128, 48, 64, 128><<<1, nullptr, stream>>>(
            reinterpret_cast<int16_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 2);
    }
}

template void launchTMOVAcc2MatSCQuantNz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVAcc2MatSCQuantNz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVAcc2MatSCQuantNz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVAcc2MatSCQuantNz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVAcc2MatSCQuantNz<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVAcc2MatFBQuantNz(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        runVectorQuantTMOV_nz2nz<half, int8_t, int8_t, uint64_t, 80, 128, 64, 80, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        runVectorQuantTMOV_nz2nz<int8_t, half, half, uint64_t, 80, 128, 64, 80, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out),
                reinterpret_cast<half *>(src0),
                reinterpret_cast<half *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3) {
        runVectorQuantTMOV_nz2nz<int8_t, int8_t, int8_t, uint64_t, 80, 128, 64, 80, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        runVectorQuantTMOV_nz2nz<uint8_t, int8_t, int8_t, uint64_t, 80, 128, 64, 80, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        runVectorQuantTMOV_nz2nz<int16_t, int8_t, int8_t, uint64_t, 80, 128, 64, 80, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int16_t *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    }
}
template void launchTMOVAcc2MatFBQuantNz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVAcc2MatFBQuantNz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVAcc2MatFBQuantNz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVAcc2MatFBQuantNz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVAcc2MatFBQuantNz<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
