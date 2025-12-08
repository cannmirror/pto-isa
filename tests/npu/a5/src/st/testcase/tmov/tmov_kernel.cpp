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

using namespace pto;

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
__aicore__ inline T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
__aicore__ inline void DynGM2L1(__cbuf__ T *dst, __gm__ T *src, unsigned TShape0, unsigned TShape1)
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
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1,
                                   unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1,
                                   int reserved)
{ // ND2NZ
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
    uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(GMT); //

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

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline void DynL1ToL0A(__ca__ T *dst, __cbuf__ T *src, unsigned dstM, unsigned dstK, unsigned srcM,
                                  unsigned srcK)
{
    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(T);
    uint16_t blockCubeSize = std::is_same<T, int8_t>::value ? 32 : 16;
    dstM = CeilAlign<uint16_t>(dstM, blockCubeSize);
    dstK = CeilAlign<uint16_t>(dstK, blockCubeK);
    srcM = CeilAlign<uint16_t>(srcM, blockCubeSize);
    srcK = CeilAlign<uint16_t>(srcK, blockCubeK);

    uint16_t srcStride = srcM / 16;
    uint16_t dstStride = dstM / 16;
    uint16_t mStep = dstM / 16;
    uint16_t kStep = dstK * sizeof(T) / 32;

    load_cbuf_to_ca(dst, src, 0, 0, mStep, kStep, srcStride, dstStride, 0);
}

// Nz2Zn
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline void DynL1ToL0B(__cb__ T *dst, __cbuf__ T *src, unsigned dstK, unsigned dstN, unsigned srcK,
                                  unsigned srcN)
{
    auto nBlockSize = 32;
    int64_t frac_num = 32 / sizeof(T);
    dstK = (dstK + frac_num - 1) / frac_num * frac_num;
    dstN = (dstN + frac_num - 1) / frac_num * frac_num;
    srcN = (srcN + frac_num - 1) / frac_num * frac_num;
    srcK = (srcK + frac_num - 1) / frac_num * frac_num;

    uint16_t srcStride = srcK / 16;
    uint16_t dstStride = dstN / 16;
    uint16_t mStep = dstK / 16;
    uint16_t kStep = dstN * sizeof(T) / 32;

    load_cbuf_to_cb(dst, src, 0, 0, mStep, kStep, srcStride, dstStride, 1);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
__aicore__ inline void L0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned GmShape0, unsigned GmShape1,
                                  unsigned GmOffset0, unsigned GmOffset1, int uf)
{  // NZ2ND
    uint16_t MSize = oriTShape0 < (GmShape0 - GmOffset0) ? oriTShape0 : (GmShape0 - GmOffset0);
    uint16_t NSize = TShape1 < (GmShape1 - GmOffset1) ? TShape1 : (GmShape1 - GmOffset1);
    uint32_t dstStride_dst_D = GmShape1;
    uint16_t srcStride = TShape0;
    uint64_t ndNum = 1;
    uint64_t src_nd_stride = 0;
    uint64_t dst_nd_stride = 0;

    uint8_t UnitFlagMode = uf;
    uint64_t QuantPRE = NoQuant;
    uint8_t ReLUPRE = 0;
    bool channelSplit = false;
    bool NZ2ND_EN = true;

    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<GMT, int8_t>::value) {
            QuantPRE = QuantMode_t::VQF322B8_PRE;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    } else if (std::is_same<L0CT, int32_t>::value) {
        if (std::is_same<GMT, int8_t>::value) {
            QuantPRE = QuantMode_t::VREQ8;
        } else if (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::VDEQF16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            QuantPRE = QuantMode_t::VQS322BF16_PRE;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }
    uint64_t config = (static_cast<uint64_t>(dst_nd_stride) << 32) | (static_cast<uint64_t>(src_nd_stride) << 16) |
                      (static_cast<uint64_t>(ndNum));
    set_loop3_para(config);
    copy_matrix_cc_to_gm((__gm__ GMT *)(dst + (GmOffset0 * GmShape1) + GmOffset1), (__cc__ L0CT *)src, 0, NSize, MSize,
                         dstStride_dst_D, srcStride, 0, 0, UnitFlagMode, QuantPRE, ReLUPRE, 0, NZ2ND_EN, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0);
}

template <typename cType, typename aType, typename bType, typename biasType, int M, int K, int N, int ValidM,
          int ValidK, int ValidN>
__global__ __aicore__ void runTMovL12Bias(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1,
                                          __gm__ biasType *src2)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    using GlobalDataOut =
        GlobalTensor<cType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    constexpr int alignN =
        ((N * sizeof(biasType) + 63) / 64) * 64 / sizeof(biasType); // bias按照64位对齐

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::RowMajor, ValidM, ValidK, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, ValidK, ValidN, SLayout::ColMajor, 512>;
    using TileMatBiasData = Tile<Location::Mat, biasType, 1, alignN, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox, 512>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<cType, M, N, ValidM, ValidN>;

    using BiasTile = Tile<Location::Bias, cType, 1, alignN, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox, 512>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatBiasData biasMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasMatTile, 0x20000);

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
    using BiasType = typename BiasTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();
    __cbuf__ biasType *srcBiasAddr = biasMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /******************************GM->L1(NZ2NZ)*****************************/
    DynL1CopyIn<AType, AType>(srcAAddr, src0, ValidM, ValidK, ValidM, ValidK, 0, 0, 0);
    DynL1CopyIn<BType, BType>(srcBAddr, src1, ValidK, ValidN, ValidK, ValidN, 0, 0, 0);
    DynGM2L1<biasType>(srcBiasAddr, src2, 1, alignN);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV**************************/
    DynL1ToL0A<aType, 0, 0>(a, srcAAddr, M, K, M, K); // Nz2Zz
    DynL1ToL0B<bType, 0, 0>(b, srcBAddr, K, N, K, N); // Nz2Zn [K, N]
    TMOV(biasTile, biasMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /****************************TMATMUL********************************/
    mad(c, a, b, M, K, N, false, false, true, false);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    L0CCopyOut<cType, cType, M, N, M, N>(out, c, ValidM, ValidN, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename biasType, int M, int K, int N, int ValidM,
          int ValidK, int ValidN>
__global__ __aicore__ void runTMovL12BiasDynamic(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1,
                                                 __gm__ biasType *src2)
{
    using GlobalDataSrc0 =
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<biasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    using GlobalDataOut =
        GlobalTensor<cType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    constexpr int alignN =
        ((N * sizeof(biasType) + 63) / 64) * 64 / sizeof(biasType); // bias按照64位对齐

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    using TileMatBiasData = Tile<Location::Mat, biasType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512>;

    using LeftTile = TileLeft<aType, M, K, -1, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, -1>;
    using AccTile = TileAcc<cType, M, N, ValidM, -1>;

    using BiasTile = Tile<Location::Bias, cType, 1, alignN, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512>;

    TileMatAData aMatTile(ValidM, ValidK);
    TileMatBData bMatTile(ValidK, ValidN);
    TileMatBiasData biasMatTile(ValidN);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasMatTile, 0x20000);

    LeftTile aTile(ValidM);
    RightTile bTile(ValidN);
    AccTile cTile(ValidN);
    BiasTile biasTile(ValidN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(biasTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;
    using BiasType = typename BiasTile::DType;
    using BiasMatType = typename TileMatBiasData::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();
    __cbuf__ BiasMatType *srcBiasAddr = biasMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /******************************GM->L1(NZ2NZ)*****************************/
    DynL1CopyIn<AType, AType>(srcAAddr, src0, ValidM, ValidK, ValidM, ValidK, 0, 0, 0);
    DynL1CopyIn<BType, BType>(srcBAddr, src1, ValidK, ValidN, ValidK, ValidN, 0, 0, 0);
    DynGM2L1<BiasMatType>(srcBiasAddr, src2, 1, alignN);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV**************************/
    DynL1ToL0A<aType, 0, 0>(a, srcAAddr, M, K, M, K); // Nz2Zz
    DynL1ToL0B<bType, 0, 0>(b, srcBAddr, K, N, K, N); // Nz2Zn [K, N]
    TMOV(biasTile, biasMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /****************************TMATMUL********************************/
    mad(c, a, b, M, K, N, false, false, true, false);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    L0CCopyOut<cType, cType, M, N, M, N>(out, c, ValidM, ValidN, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename cType, typename aType, typename bType, typename fbType, typename l0cType, int M, int K, int N,
          int ValidM, int ValidK, int ValidN>
__global__ __aicore__ void runTMovL12Fb(__gm__ cType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    // static shape
    using GlobalDataSrc0 =
        GlobalTensor<aType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<bType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataSrc2 = GlobalTensor<fbType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    using GlobalDataOut =
        GlobalTensor<cType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::RowMajor, ValidM, ValidK, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, ValidK, ValidN, SLayout::ColMajor, 512>;
    using TileMatFbData = Tile<Location::Mat, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    using LeftTile = TileLeft<aType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<bType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<l0cType, M, N, ValidM, ValidN>;

    using FbTile = Tile<Location::Scaling, fbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(fbMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    FbTile fbTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(fbTile, 0x0);

    __cbuf__ aType *srcAAddr = aMatTile.data();
    __cbuf__ bType *srcBAddr = bMatTile.data();
    __cbuf__ fbType *srcFbAddr = fbMatTile.data();

    __ca__ aType *a = (__ca__ aType *)(aTile.data());
    __cb__ bType *b = (__cb__ bType *)(bTile.data());
    __cc__ l0cType *c = (__cc__ l0cType *)(cTile.data());

    /******************************GM->L1(NZ2NZ)*****************************/
    DynL1CopyIn<aType, aType>(srcAAddr, src0, ValidM, ValidK, ValidM, ValidK, 0, 0, 0);
    DynL1CopyIn<bType, bType>(srcBAddr, src1, ValidK, ValidN, ValidK, ValidN, 0, 0, 0);
    DynGM2L1<fbType>(srcFbAddr, src2, 1, N);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV**************************/
    DynL1ToL0A<aType, 0, 0>(a, srcAAddr, M, K, M, K); // Nz2Zz
    DynL1ToL0B<bType, 0, 0>(b, srcBAddr, K, N, K, N); // Nz2Zn [K, N]
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    mad(c, a, b, M, K, N, false, false, false, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(fbTile, fbMatTile);

    // [7:0]ReLU_pre address, [15:8]Quant_pre address
    __fbuf__ fbType *quantPreAddr = (__fbuf__ fbType *)(fbTile.data());
    uint64_t config = 0;
    uint64_t reluAddr = 0;
    config = config | (((uint64_t)quantPreAddr >> 7) << 8); // align with 128bit
    set_fpc((uint64_t)config);
    /********************************TSTORE****************************/
    L0CCopyOut<cType, l0cType, M, N, M, N>(out, c, ValidM, ValidN, 0, 0, 0);
    out = dstGlobal.data();
}

template <int32_t tilingKey>
void launchTMovL12Bias(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *bias, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMovL12Bias<float, half, half, float, 64, 96, 32, 64, 96, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(bias));
    } else if constexpr (tilingKey == 2) {
        runTMovL12Bias<float, float, float, half, 128, 128, 64, 128, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<half *>(bias));
    } else if constexpr (tilingKey == 3) {
        runTMovL12Bias<float, float, float, bfloat16_t, 64, 80, 32, 64, 80, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<bfloat16_t *>(bias));
    } else if constexpr (tilingKey == 4) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 128, 96, 64, 128, 96, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(bias));
    } else if constexpr (tilingKey == 5) {
        runTMovL12Bias<int32_t, int8_t, int8_t, int32_t, 32, 32, 64, 31, 32, 63>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int32_t *>(bias));
    } else if constexpr (tilingKey == 6) {
        runTMovL12BiasDynamic<float, half, half, half, 64, 80, 32, 64, 80, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(bias));
    } else if constexpr (tilingKey == 7) {
        runTMovL12BiasDynamic<float, float, float, bfloat16_t, 112, 96, 48, 112, 96, 48>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<bfloat16_t *>(bias));
    } else if constexpr (tilingKey == 8) {
        runTMovL12BiasDynamic<float, float, float, bfloat16_t, 16, 96, 64, 15, 96, 63>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<bfloat16_t *>(bias));
    }
}

template <int32_t tilingKey>
void launchTMovL12Fb(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *scaling, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMovL12Fb<int8_t, int8_t, int8_t, uint64_t, int32_t, 32, 32, 128, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(scaling));
    } else if constexpr (tilingKey == 2) {
        runTMovL12Fb<half, int8_t, int8_t, uint64_t, int32_t, 96, 32, 64, 96, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(scaling));
    } else if constexpr (tilingKey == 3) {
        runTMovL12Fb<bfloat16_t, int8_t, int8_t, uint64_t, int32_t, 128, 96, 64, 128, 96, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(scaling));
    } else if constexpr (tilingKey == 4) {
        runTMovL12Fb<int8_t, float, float, uint64_t, float, 112, 96, 48, 112, 96, 48>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(scaling));
    } else if constexpr (tilingKey == 5) {
        runTMovL12Fb<int8_t, float, float, uint64_t, float, 32, 96, 32, 31, 96, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(scaling));
    }
}

template void launchTMovL12Bias<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Bias<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template void launchTMovL12Fb<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Fb<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Fb<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Fb<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMovL12Fb<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);