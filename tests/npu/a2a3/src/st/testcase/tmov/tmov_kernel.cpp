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

template <typename T>
__aicore__ inline void DynGM2L1NZ2NZ(__cbuf__ T* dst, __gm__ T* src, unsigned TShape0, unsigned TShape1)
{
    uint16_t nBurst = 1;
    uint16_t lenBurst = TShape0 * TShape1 * sizeof(T) / 32;
    uint16_t srcGap = 0;
    uint16_t dstGap = 0;
    copy_gm_to_cbuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap, (pad_t)0);
}

template <typename T>
__aicore__ inline void DynGM2L1ND2NZ(__cbuf__ T* dst, __gm__ T* src, unsigned TShape0, unsigned TShape1, int isBias = 0,
                                     int isScaling = 0)
{  // ND2NZ
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = TShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(TShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 0;

    if constexpr (std::is_same<T, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ T*)dst, (__gm__ T*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                       srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ T*)dst, (__gm__ T*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                        srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ T*)dst, (__gm__ T*)src, 0 /*sid*/, ndNum, nValue, dValue,
                                         srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
__aicore__ inline void L0CCopyOut(__gm__ GMT* dst, __cc__ L0CT* src, unsigned GmShape0, unsigned GmShape1,
                                  unsigned GmOffset0, unsigned GmOffset1, int uf = 0, uint8_t reluMode = 0)
{  // NZ2ND
    uint16_t MSize = oriTShape0 < GmShape0 ? oriTShape0 : GmShape0;
    uint16_t NSize = TShape1 < GmShape1 ? TShape1 : GmShape1;
    uint32_t dstStride_dst_D = GmShape1;
    uint16_t srcStride = TShape0;
    uint64_t ndNum = 1;
    uint64_t src_nd_stride = 0;
    uint64_t dst_nd_stride = 0;

    uint8_t UnitFlagMode = uf;
    uint64_t QuantPRE = NoQuant;
    uint8_t ReLUPRE = reluMode;
    bool channelSplit = false;
    bool NZ2ND_EN = true;

    uint64_t config = 0, nd_para = 0;
    nd_para = nd_para | (ndNum & 0xffff);
    nd_para = nd_para | ((src_nd_stride & 0xffff) << 16);
    nd_para = nd_para | ((dst_nd_stride & 0xffff) << 32);
    set_nd_para(nd_para);

    // 量化模式选择
    if constexpr (std::is_same<L0CT, float>::value) {
        if constexpr (std::is_same<GMT, int8_t>::value) {
            QuantPRE = QuantMode_t::VQF322B8_PRE;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    } else if constexpr (std::is_same<L0CT, int32_t>::value) {
        if constexpr (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::VDEQF16;
        } else if constexpr (std::is_same<GMT, int8_t>::value) {
            QuantPRE = QuantMode_t::VREQ8;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }

    copy_matrix_cc_to_gm((__gm__ GMT*)dst, (__cc__ L0CT*)src, 0, NSize, MSize, dstStride_dst_D, srcStride, UnitFlagMode,
                         QuantPRE, ReLUPRE, channelSplit, NZ2ND_EN);
}

template <typename cType, typename aType, typename bType, typename biasInputType, typename l0CType, int M, int N, int K,
          int isAtranspose, int isBtranspose>
__global__ __aicore__ void TMOV2BTKernel(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1,
                                         __gm__ biasInputType* src2)
{
    // bias按照64B对齐申请tile大小和搬运
    constexpr int alignBiasN =
        ((N * sizeof(biasInputType) + 63) / 64) * 64 / sizeof(biasInputType);
    using GlobalDataSrc0 = GlobalTensor<aType, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<biasInputType, Shape<1, 1, 1, 1, alignBiasN>,
                                        Stride<1 * alignBiasN, 1 * alignBiasN, alignBiasN, alignBiasN, 1>>;
    using GlobalDataOut = GlobalTensor<cType, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<Location::Mat, aType, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<Location::Mat, aType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using TileMatBiasData =
        Tile<Location::Mat, biasInputType, 1, alignBiasN, BLayout::RowMajor, 1, alignBiasN, SLayout::NoneBox>;

    using LeftTile = Tile<Location::Left, aType, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, K, N, K, N>;
    using AccTile = TileAcc<l0CType, M, N, M, N>;

    using BiasTile = Tile<Location::Bias, l0CType, 1, alignBiasN, BLayout::RowMajor, 1, N, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatBiasData biasMatTile;
    uint32_t aMatSize = M * K * sizeof(aType);
    uint32_t bMatSize = K * N * sizeof(bType);
    uint32_t biasMatSize = alignBiasN * sizeof(biasInputType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(biasMatTile, 0x0 + aMatSize + bMatSize);
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    BiasTile biasTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType* srcAAddr = aMatTile.data();
    __cbuf__ BType* srcBAddr = bMatTile.data();
    __cbuf__ biasInputType* srcBiasAddr = biasMatTile.data();

    __ca__ AType* a = (__ca__ AType*)(aTile.data());
    __cb__ BType* b = (__cb__ BType*)(bTile.data());
    __cc__ CType* c = (__cc__ CType*)(cTile.data());

    /*************************************GM->L1(NZ2NZ)****************************************/
    DynGM2L1NZ2NZ<AType>(srcAAddr, src0, M, K);
    DynGM2L1NZ2NZ<BType>(srcBAddr, src1, K, N);
    DynGM2L1NZ2NZ<biasInputType>(srcBiasAddr, src2, 1, N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV(L1->L0A/L0B/Bias) && TMATMUL**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, M, K, N, false, false, true, false);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<cType, l0CType, M, N, M, N>(out, c, M, N, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename biasInputType, typename l0CType, int M, int N, int K,
          int isAtranspose, int isBtranspose>
__global__ __aicore__ void TMOV2BTDyncmicKernel(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1,
                                                __gm__ biasInputType* src2, int m, int k, int n)
{
    // bias按照64B对齐申请tile大小和搬运
    constexpr int alignN =
        ((N * sizeof(biasInputType) + 63) / 64) * 64 / sizeof(biasInputType);
    using DynShapeADim5 = pto::Shape<1, 1, 1, M, K>;
    using DynSTridADim5 = pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>;

    using DynShapeBDim5 = pto::Shape<1, 1, 1, K, N>;
    using DynSTridBDim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using DynShapeBiasDim5 = pto::Shape<1, 1, 1, 1, alignN>;
    using DynSTridBiasDim5 = pto::Stride<1 * alignN, 1 * alignN, alignN, alignN, 1>;

    using DynShapeCDim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTridCDim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using GlobalDataSrc0 = GlobalTensor<aType, DynShapeADim5, DynSTridADim5>;
    using GlobalDataSrc1 = GlobalTensor<bType, DynShapeBDim5, DynSTridBDim5>;
    using GlobalDataSrc2 = GlobalTensor<biasInputType, DynShapeBiasDim5, DynSTridBiasDim5>;
    using GlobalDataOut = GlobalTensor<cType, DynShapeCDim5, DynSTridCDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<Location::Mat, aType, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<Location::Mat, aType, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    using TileMatBiasData = Tile<Location::Mat, biasInputType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = Tile<Location::Left, aType, M, K, BLayout::RowMajor, -1, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, K, N, K, -1>;
    using AccTile = TileAcc<l0CType, M, N, M, -1>;

    using BiasTile = Tile<Location::Bias, l0CType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TileMatBiasData biasMatTile(n);

    uint32_t aMatSize = M * K * sizeof(aType);
    uint32_t bMatSize = K * N * sizeof(bType);
    uint32_t biasMatSize = alignN * sizeof(biasInputType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(biasMatTile, 0x0 + aMatSize + bMatSize);

    LeftTile aTile(m);
    RightTile bTile(n);
    AccTile cTile(n);
    BiasTile biasTile(n);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType* srcAAddr = aMatTile.data();
    __cbuf__ BType* srcBAddr = bMatTile.data();
    __cbuf__ biasInputType* srcBiasAddr = biasMatTile.data();

    __ca__ AType* a = (__ca__ AType*)(aTile.data());
    __cb__ BType* b = (__cb__ BType*)(bTile.data());
    __cc__ CType* c = (__cc__ CType*)(cTile.data());

    /*************************************GM->L1(NZ2NZ)****************************************/
    DynGM2L1NZ2NZ<AType>(srcAAddr, src0, M, K);
    DynGM2L1NZ2NZ<BType>(srcBAddr, src1, K, N);
    DynGM2L1NZ2NZ<biasInputType>(srcBiasAddr, src2, 1, alignN);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV(L1->L0A/L0B/Bias) && TMATMUL**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, M, K, N, false, false, true, false);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<cType, l0CType, M, N, M, N>(out, c, M, N, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename scalingType, typename l0cType, int M, int N, int K,
          int isAtranspose, int isBtranspose, uint8_t reluMode>
__global__ __aicore__ void TMOV2ScalingKernel(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1,
                                              __gm__ scalingType* src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<scalingType, Shape<1, 1, 1, 1, N>, Stride<1 * N, 1 * N, N, N, 1>>;
    using GlobalDataOut = GlobalTensor<cType, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<Location::Mat, aType, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<Location::Mat, aType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using TileMatFbData = Tile<Location::Mat, scalingType, 1, N, BLayout::RowMajor, 1, N, SLayout::NoneBox>;

    using LeftTile = Tile<Location::Left, aType, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, K, N, K, N>;
    using AccTile = TileAcc<l0cType, M, N, M, N>;

    using FbTile = Tile<Location::Scaling, scalingType, 1, N, BLayout::RowMajor, 1, N, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;

    uint32_t aMatSize = M * K * sizeof(aType);
    uint32_t bMatSize = K * N * sizeof(bType);
    uint32_t fbMatSize = N * sizeof(scalingType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(fbMatTile, 0x0 + aMatSize + bMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(fbTile, 0x0);
    TASSIGN(cTile, 0x0);

    __cbuf__ aType* srcAAddr = aMatTile.data();
    __cbuf__ bType* srcBAddr = bMatTile.data();
    __cbuf__ uint64_t* srcFbAddr = fbMatTile.data();

    __ca__ aType* a = (__ca__ aType*)(aTile.data());
    __cb__ bType* b = (__cb__ bType*)(bTile.data());
    __cc__ l0cType* c = (__cc__ l0cType*)(cTile.data());

    /*************************************GM->L1(NZ2NZ)****************************************/
    DynGM2L1NZ2NZ<aType>(srcAAddr, src0, M, K);
    DynGM2L1NZ2NZ<bType>(srcBAddr, src1, K, N);
    DynGM2L1NZ2NZ<uint64_t>(srcFbAddr, src2, 1, N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV(L1->L0A/L0B/Bias/Fb) && TMATMUL**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, M, K, N, false, false, false, true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(fbTile, fbMatTile);
    __fbuf__ uint64_t* fb = (__fbuf__ uint64_t*)(0);
    // 将fb的地址设置到寄存器[15:0]表示地址，[21:16]表示model
    uint64_t deqTensorAddr = ((uint64_t)fb >> static_cast<uint64_t>(7)) << 8;
    set_fpc((uint64_t)deqTensorAddr);
    pipe_barrier(PIPE_FIX);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<cType, l0cType, M, N, M, N>(out, c, M, N, 0, 0, 0, reluMode);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename biasInputType, typename scalingType,
          typename l0CType, int M, int N, int K, int isAtranspose, int isBtranspose>
__global__ __aicore__ void TMOV2NdDyncmicKernel(__gm__ cType* out, __gm__ aType* src0, __gm__ bType* src1,
                                                __gm__ biasInputType* src2, __gm__ scalingType* src3, int m, int k,
                                                int n)
{
    // copy_l1_bias + copy_l1_fb + ndInput + dynamic + unalign
    // Bias按照64B对齐申请tile大小和搬运
    constexpr int alignBiasN =
        ((N * sizeof(biasInputType) + 63) / 64) * 64 / sizeof(biasInputType);
    // Scaling按照128B对齐申请tile大小和搬运
    constexpr int alignScalingN =
        ((N * sizeof(scalingType) + 127) / 128) * 128 / sizeof(scalingType);
    using DynShapeADim5 = pto::Shape<1, 1, 1, M, K>;
    using DynSTridADim5 = pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>;

    using DynShapeBDim5 = pto::Shape<1, 1, 1, K, N>;
    using DynSTridBDim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using DynShapeBiasDim5 = pto::Shape<1, 1, 1, 1, alignBiasN>;
    using DynSTridBiasDim5 = pto::Stride<1 * alignBiasN, 1 * alignBiasN, alignBiasN, alignBiasN, 1>;

    using DynShapeScalingDim5 = pto::Shape<1, 1, 1, 1, alignScalingN>;
    using DynSTridScalingDim5 = pto::Stride<1 * alignScalingN, 1 * alignScalingN, alignScalingN, alignScalingN, 1>;

    using DynShapeCDim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTridCDim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using GlobalDataSrc0 = GlobalTensor<aType, DynShapeADim5, DynSTridADim5>;
    using GlobalDataSrc1 = GlobalTensor<bType, DynShapeBDim5, DynSTridBDim5>;
    using GlobalDataSrc2 = GlobalTensor<biasInputType, DynShapeBiasDim5, DynSTridBiasDim5>;
    using GlobalDataSrc3 = GlobalTensor<scalingType, DynShapeScalingDim5, DynSTridScalingDim5>;
    using GlobalDataOut = GlobalTensor<cType, DynShapeCDim5, DynSTridCDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataSrc3 src3Global(src3);
    GlobalDataOut dstGlobal(out);

    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(aType);
    // A不转置[M,K]输入，M向16对齐，K向C0Size对齐；B转置[N,K]输入，N向16对齐，K向C0Size对齐；
    constexpr uint16_t alignM = (M + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr uint16_t alignN = (N + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr uint16_t alignK = (K + blockCubeK - 1) / blockCubeK * blockCubeK;

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<Location::Mat, aType, alignM, alignK, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, aType, alignM, alignK, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData = Tile<Location::Mat, bType, alignK, alignN, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    using TileMatBiasData =
        Tile<Location::Mat, biasInputType, 1, alignBiasN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using TileMatScalingData =
        Tile<Location::Mat, scalingType, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = Tile<Location::Left, aType, alignM, alignK, BLayout::RowMajor, -1, alignK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<bType, alignK, alignN, alignK, -1>;
    using AccTile = TileAcc<l0CType, alignM, alignN, alignM, -1>;

    using BiasTile = Tile<Location::Bias, l0CType, 1, alignBiasN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using ScalingTile =
        Tile<Location::Scaling, scalingType, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile(alignM, alignK);
    TileMatBData bMatTile(alignK, alignN);
    TileMatBiasData biasMatTile(alignBiasN);
    TileMatScalingData scalingMatTile(alignScalingN);

    uint32_t aMatSize = alignM * alignK * sizeof(aType);
    uint32_t bMatSize = alignK * alignN * sizeof(bType);
    uint32_t biasMatSize = alignBiasN * sizeof(biasInputType);
    uint32_t fbMatSize = alignScalingN * sizeof(scalingType);

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(biasMatTile, 0x0 + aMatSize + bMatSize);
    TASSIGN(scalingMatTile, 0x0 + aMatSize + bMatSize + biasMatSize);

    LeftTile aTile(alignM);
    RightTile bTile(alignN);
    AccTile cTile(alignN);
    BiasTile biasTile(alignBiasN);
    ScalingTile scalingTile(alignScalingN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType* srcAAddr = aMatTile.data();
    __cbuf__ BType* srcBAddr = bMatTile.data();
    __cbuf__ biasInputType* srcBiasAddr = biasMatTile.data();
    __cbuf__ scalingType* srcScalingAddr = scalingMatTile.data();

    __ca__ AType* a = (__ca__ AType*)(aTile.data());
    __cb__ BType* b = (__cb__ BType*)(bTile.data());
    __cc__ CType* c = (__cc__ CType*)(cTile.data());

    /*************************************GM->L1(ND2NZ)****************************************/
    DynGM2L1ND2NZ<AType>(srcAAddr, src0, M, K);
    DynGM2L1ND2NZ<BType>(srcBAddr, src1, N, K);
    DynGM2L1NZ2NZ<biasInputType>(srcBiasAddr, src2, 1, alignBiasN);
    DynGM2L1NZ2NZ<scalingType>(srcScalingAddr, src3, 1, alignScalingN);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV(L1->L0A/L0B/Bias/Fb) && MATMUL**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    TMOV(biasTile, biasMatTile);
    TMOV(scalingTile, scalingMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, M, K, N, false, false, true, false);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // l1->fb
    TMOV(scalingTile, scalingMatTile);
    __fbuf__ uint64_t* fb = (__fbuf__ uint64_t*)(0);
    // 将fb的地址设置到寄存器[15:0]表示地址，[21:16]表示model
    uint64_t deqTensorAddr = ((uint64_t)fb >> static_cast<uint64_t>(7)) << 8;
    set_fpc((uint64_t)deqTensorAddr);
    pipe_barrier(PIPE_FIX);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<cType, l0CType, alignM, alignN, alignM, alignN>(out, c, M, N, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename AT, typename BT, typename L0CT, typename BiasT, typename GMT, typename ScalingT, int M, int N, int K,
          int isAtranspose, int isBtranspose, int IsBias, int IsQuant, int ReluMode, int Isdynamic, int IsNd = 0>
void LaunchTMOV(GMT* out, AT* src0, BT* src1, BiasT* src2, ScalingT* src3, void* stream)
{
    if constexpr (!Isdynamic) {
        if constexpr (IsBias) {
            if constexpr (std::is_same_v<AT, uint16_t> && std::is_same_v<BiasT, uint16_t>) {
                TMOV2BTKernel<GMT, half, half, half, L0CT, M, N, K, isAtranspose, isBtranspose><<<1, nullptr, stream>>>(
                    out, reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), reinterpret_cast<half*>(src2));
            } else if constexpr (std::is_same_v<BiasT, uint16_t>) {
                TMOV2BTKernel<GMT, AT, BT, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, reinterpret_cast<half*>(src2));
            } else if constexpr (std::is_same_v<AT, uint16_t>) {
                TMOV2BTKernel<GMT, half, half, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), src2);
            } else {
                TMOV2BTKernel<GMT, AT, BT, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, src2);
            }
        } else if constexpr (IsQuant) {
            if constexpr (std::is_same_v<AT, uint16_t>) {
                TMOV2ScalingKernel<GMT, half, half, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose, ReluMode>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), src3);
            } else if constexpr (std::is_same_v<GMT, uint16_t>) {
                TMOV2ScalingKernel<half, AT, BT, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose, ReluMode>
                    <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), src0, src1, src3);
            } else {
                TMOV2ScalingKernel<GMT, AT, BT, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose, ReluMode>
                    <<<1, nullptr, stream>>>(out, src0, src1, src3);
            }
        }
    } else {
        if constexpr (IsBias && !IsQuant) {
            // 输入aType为half, biasType为float, A不转置，B不转置
            if constexpr (std::is_same_v<AT, uint16_t> && std::is_same_v<BiasT, uint16_t>) {
                TMOV2BTDyncmicKernel<GMT, half, half, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
                                             reinterpret_cast<half*>(src2), M, N, K);
            } else if constexpr (std::is_same_v<BiasT, uint16_t>) {
                TMOV2BTDyncmicKernel<GMT, AT, BT, half, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, reinterpret_cast<half*>(src2), M, N, K);
            } else if constexpr (std::is_same_v<AT, uint16_t>) {
                TMOV2BTDyncmicKernel<GMT, half, half, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), src2, M,
                                             N, K);
            } else {
                TMOV2BTDyncmicKernel<GMT, AT, BT, BiasT, L0CT, M, N, K, isAtranspose, isBtranspose>
                    <<<1, nullptr, stream>>>(out, src0, src1, src2, M, N, K);
            }
        } else if constexpr (IsBias && IsQuant && IsNd) {
            TMOV2NdDyncmicKernel<GMT, AT, BT, BiasT, ScalingT, L0CT, M, N, K, isAtranspose, isBtranspose>
                <<<1, nullptr, stream>>>(out, src0, src1, src2, src3, M, N, K);
        }
    }
}

// atype, btype, l0ctype, biastype, gmtype, scalingtype, M, N, K, is_atrans, is_btrans, is_bias, is_quant, relu_mode, isdynamic
template void LaunchTMOV<uint16_t, uint16_t, float, float, float, uint64_t, 64, 32, 80, 0, 1, 1, 0, 0, 0>(
    float* out, uint16_t* src0, uint16_t* src1, float* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int32_t, uint64_t, 128, 64, 128, 0, 1, 1, 0, 0, 0>(
    int32_t* out, int8_t* src0, int8_t* src1, int32_t* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<float, float, float, float, float, uint64_t, 128, 48, 64, 0, 1, 1, 0, 0, 0>(
    float* out, float* src0, float* src1, float* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<uint16_t, uint16_t, float, uint16_t, float, uint64_t, 64, 32, 80, 0, 1, 1, 0, 0, 1>(
    float* out, uint16_t* src0, uint16_t* src1, uint16_t* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<float, float, float, uint16_t, float, uint64_t, 112, 48, 96, 0, 1, 1, 0, 0, 1>(
    float* out, float* src0, float* src1, uint16_t* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<float, float, float, uint16_t, float, uint64_t, 64, 128, 96, 0, 1, 1, 0, 0, 0>(
    float* out, float* src0, float* src1, uint16_t* src2, uint64_t* src3, void* stream);

template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 128, 112, 32, 0, 1, 0, 1, 0, 0>(
    int8_t* out, int8_t* src0, int8_t* src1, int32_t* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, uint16_t, uint64_t, 144, 80, 160, 0, 1, 0, 1, 0, 0>(
    uint16_t* out, int8_t* src0, int8_t* src1, int32_t* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<uint16_t, uint16_t, float, float, int8_t, uint64_t, 64, 32, 80, 0, 1, 0, 1, 0, 0>(
    int8_t* out, uint16_t* src0, uint16_t* src1, float* src2, uint64_t* src3, void* stream);

template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 60, 17, 80, 0, 1, 1, 1, 0, 1, 1>(
    int8_t* out, int8_t* src0, int8_t* src1, int32_t* src2, uint64_t* src3, void* stream);
template void LaunchTMOV<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 15, 10, 30, 0, 1, 1, 1, 0, 1, 1>(
    int8_t* out, int8_t* src0, int8_t* src1, int32_t* src2, uint64_t* src3, void* stream);