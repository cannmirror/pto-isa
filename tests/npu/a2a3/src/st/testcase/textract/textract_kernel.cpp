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

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = std::conditional_t<isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

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
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose, int targetM,
    int targetN, int targetK, bool isKAlign = false>
__aicore__ inline void runTMOV_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    // static shape
    using GlobalDataSrc0 = std::conditional_t<isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>,
            pto::Stride<targetM * targetK, targetM * targetK, targetM * targetK, 1, targetM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>,
            pto::Stride<targetM * targetK, targetM * targetK, targetM * targetK, targetK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>,
            pto::Stride<targetN * targetK, targetN * targetK, targetN * targetK, 1, targetK>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>,
            pto::Stride<targetN * targetK, targetN * targetK, targetN * targetK, targetN, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<isAtranspose,
        Tile<Location::Mat, U, targetM, targetK, BLayout::RowMajor, targetM, targetK, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, targetM, targetK, BLayout::ColMajor, targetM, targetK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<isBtranspose,
        Tile<Location::Mat, S, targetK, targetN, BLayout::RowMajor, targetK, targetN, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, targetK, targetN, BLayout::ColMajor, targetK, targetN, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, targetM, targetK, targetM, targetK>;
    using RightTile = TileRight<S, targetK, targetN, targetK, targetN>;
    using AccTile = TileAcc<T, targetM, targetN, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV*******************************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, targetM, K, targetN, false, isKAlign, false, true);  // 是否使能K16对齐，避免b32场景读取脏数据
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
    bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;
    // static shape
    using GlobalDataSrc0 = std::conditional_t<isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
        pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, mValid, kValid>;
    using RightTile = TileRight<S, kValid, nValid, kValid, nValid>;
    using AccTile = TileAcc<T, mValid, nValid, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

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
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
    bool isAtranspose, bool isBtranspose, int targetM, int targetN, int targetK, bool isKAlign = false>
__aicore__ inline void runTEXTRACT_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;

    using GlobalDataSrc0 = std::conditional_t<isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>,
            pto::Stride<targetM * targetK, targetM * targetK, targetM * targetK, 1, targetM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, targetM, targetK>,
            pto::Stride<targetM * targetK, targetM * targetK, targetM * targetK, targetK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>,
            pto::Stride<targetN * targetK, targetN * targetK, targetN * targetK, 1, targetK>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, targetK, targetN>,
            pto::Stride<targetN * targetK, targetN * targetK, targetN * targetK, targetN, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
        pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<isAtranspose,
        Tile<Location::Mat, U, targetM, targetK, BLayout::RowMajor, targetM, targetK, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, targetM, targetK, BLayout::ColMajor, targetM, targetK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<isBtranspose,
        Tile<Location::Mat, S, targetK, targetN, BLayout::RowMajor, targetK, targetN, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, targetK, targetN, BLayout::ColMajor, targetK, targetN, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, targetM - indexM, targetK - indexK, targetM - indexM, targetK - indexK>;
    using RightTile = TileRight<S, targetK - indexK, targetN - indexN, targetK - indexK, targetN - indexN>;
    using AccTile = TileAcc<T, targetM - indexM, targetN - indexN, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TEXTRACT*******************************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, targetM - indexM, kValid, targetN - indexN, false, isKAlign, false, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
    bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT_DYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int n, int k)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;
    // static shape
    using GlobalDataSrc0 = std::conditional_t<isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
        pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, -1, -1>;
    using RightTile = TileRight<S, kValid, nValid, kValid, -1>;
    using AccTile = TileAcc<T, mValid, nValid, -1, nValid>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile(mValid, kValid);
    RightTile bTile(nValid);
    AccTile cTile(mValid);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTMOV_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<float, half, half, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<float, float, float, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 96;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<float, half, half, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<float, float, float, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 80;
    constexpr uint32_t K = 96;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 44;

    constexpr uint16_t targetM = 32;
    constexpr uint16_t targetN = 32;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t targetM = 32;
    constexpr uint16_t targetN = 32;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    constexpr bool isKAlign = true;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK, isKAlign>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 80;
    constexpr uint16_t targetN = 96;
    constexpr uint16_t targetK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_24(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 82;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 80;
    constexpr uint16_t targetN = 96;
    constexpr uint16_t targetK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_25(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t targetM = 48;
    constexpr uint16_t targetN = 48;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 44;

    constexpr uint16_t targetM = 32;
    constexpr uint16_t targetN = 32;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t targetM = 32;
    constexpr uint16_t targetN = 32;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    constexpr bool isKAlign = true;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK, isKAlign>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 96;
    constexpr uint16_t targetN = 80;
    constexpr uint16_t targetK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_34(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 82;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 96;
    constexpr uint16_t targetN = 96;
    constexpr uint16_t targetK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_35(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t targetM = 48;
    constexpr uint16_t targetN = 48;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

template <int32_t tilingKey>
void launchTMOV(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMOV_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMOV_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMOV_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMOV_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTMOV_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTMOV_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTMOV_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 14) {
        launchTMOV_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        launchTMOV_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        launchTMOV_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        launchTMOV_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 24) {
        launchTMOV_24<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 25) {
        launchTMOV_25<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        launchTMOV_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        launchTMOV_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        launchTMOV_33<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 34) {
        launchTMOV_34<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 35) {
        launchTMOV_35<<<1, nullptr, stream>>>(out, src0, src1);
    }
}
template void launchTMOV<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<24>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<25>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<34>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<35>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

extern "C" __global__ __aicore__ void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 48;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 16;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 96;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 80;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t targetM = 32;
    constexpr uint16_t targetN = 32;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_UNALIGN<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t targetM = 80;
    constexpr uint16_t targetN = 96;
    constexpr uint16_t targetK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t targetM = 48;
    constexpr uint16_t targetN = 48;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t targetM = 32;
    constexpr uint16_t targetN = 32;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_UNALIGN<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t targetM = 96;
    constexpr uint16_t targetN = 80;
    constexpr uint16_t targetK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t targetM = 48;
    constexpr uint16_t targetN = 48;
    constexpr uint16_t targetK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, targetM, targetN, targetK>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_DYNAMIC_41(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT_DYNAMIC<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, N, K);
}

extern "C" __global__ __aicore__ void launchTEXTRACT_DYNAMIC_42(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_DYNAMIC<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, N, K);
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTEXTRACT_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTEXTRACT_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTEXTRACT_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTEXTRACT_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 14) {
        launchTEXTRACT_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        launchTEXTRACT_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        launchTEXTRACT_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        launchTEXTRACT_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        launchTEXTRACT_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        launchTEXTRACT_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        launchTEXTRACT_33<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 41) {
        launchTEXTRACT_DYNAMIC_41<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 42) {
        launchTEXTRACT_DYNAMIC_42<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTEXTRACT<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);