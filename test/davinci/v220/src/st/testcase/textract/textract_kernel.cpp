#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

using namespace pto;
template <typename T>
__aicore__ inline void DynGM2L1(__cbuf__ T *dst, __gm__ T *src, unsigned TShape0, unsigned TShape1)
{
    uint16_t nBurst = 1;
    uint16_t lenBurst = TShape0 * TShape1 * sizeof(T) / 32;
    uint16_t srcGap = 0;
    uint16_t dstGap = 0;
    copy_gm_to_cbuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap, (pad_t)0);
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

    uint64_t config = 0, nd_para = 0;
    nd_para = nd_para | (ndNum & 0xffff);
    nd_para = nd_para | ((src_nd_stride & 0xffff) << 16);
    nd_para = nd_para | ((dst_nd_stride & 0xffff) << 32);

    if (std::is_same<L0CT, float>::value) {
        if (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::F322F16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            QuantPRE = QuantMode_t::F322BF16;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }
    set_nd_para(nd_para);
    copy_matrix_cc_to_gm((__gm__ GMT *)(dst + (GmOffset0 * GmShape1) + GmOffset1),
        (__cc__ L0CT *)src,
        0,
        NSize,
        MSize,
        dstStride_dst_D,
        srcStride,
        UnitFlagMode,
        QuantPRE,
        ReLUPRE,
        channelSplit,
        NZ2ND_EN);
}

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1) {

    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1*M*K, 1*M*K, M*K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1*K*N, 1*K*N, K*N, N, 1>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1*M*N, 1*M*N, M*N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>
    >;
    using TileMatBData = std::conditional_t<
        isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>
    >;
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

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /*************************************GM->L1(NZ2NZ)****************************************/
    DynGM2L1<U>(srcAAddr, src0, M,K);
    DynGM2L1<U>(srcBAddr, src1, K,N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, M, K, N, false, false, false, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<T, T, M,N,M,N>(out, c, M,N,0,0,0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1) {

    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>
    >;
    using TileMatBData = std::conditional_t<
        isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>
    >;
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

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /*************************************TLOAD****************************************/
    DynGM2L1<U>(srcAAddr, src0, M,K);
    DynGM2L1<U>(srcBAddr, src1, K,N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, mValid, kValid, nValid, false, false, false, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    L0CCopyOut<T, T, mValid, nValid, mValid, nValid>(out, c, mValid, nValid, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACTDYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int n, int k) {

    // static shape
    using DynShape1Dim5 = pto::Shape<1, 1, 1, M, K>;
    using DynSTrid1Dim5 = pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>;

    using DynShape2Dim5 = pto::Shape<1, 1, 1, K, N>;
    using DynSTrid2Dim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using DynShape3Dim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTrid3Dim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using GlobalDataSrc0 = GlobalTensor<U, DynShape1Dim5, DynSTrid1Dim5>;
    using GlobalDataSrc1 = GlobalTensor<S, DynShape2Dim5, DynSTrid2Dim5>;
    using GlobalDataOut = GlobalTensor<T, DynShape3Dim5, DynSTrid3Dim5>;

    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>
    >;
    using TileMatBData = std::conditional_t<
        isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>
    >;
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

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /*************************************TLOAD****************************************/
    DynGM2L1<U>(srcAAddr, src0, M,K);
    DynGM2L1<U>(srcBAddr, src1, K,N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, mValid, kValid, nValid, false, false, false, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    L0CCopyOut<T, T, mValid, nValid, mValid, nValid>(out, c, mValid, nValid, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTMOVDYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int n, int k) {

    // static shape
    using DynShape1Dim5 = pto::Shape<1, 1, 1, M, K>;
    using DynSTrid1Dim5 = pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>;

    using DynShape2Dim5 = pto::Shape<1, 1, 1, K, N>;
    using DynSTrid2Dim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using DynShape3Dim5 = pto::Shape<1, 1, 1, M, N>;
    using DynSTrid3Dim5 = pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>;

    using GlobalDataSrc0 = GlobalTensor<U, DynShape1Dim5, DynSTrid1Dim5>;
    using GlobalDataSrc1 = GlobalTensor<S, DynShape2Dim5, DynSTrid2Dim5>;
    using GlobalDataOut = GlobalTensor<T, DynShape3Dim5, DynSTrid3Dim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>
    >;
    using TileMatBData = std::conditional_t<
        isBtranspose,
        Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>
    >;
    using LeftTile = TileLeft<U, M, K, -1, -1>;
    using RightTile = TileRight<S, K, N, K, -1>;
    using AccTile = TileAcc<T, M, N, -1, N>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile(m, k);
    RightTile bTile(n);
    AccTile cTile(m);
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

    /*************************************TLOAD****************************************/
    DynGM2L1<U>(srcAAddr, src0, M,K);
    DynGM2L1<U>(srcBAddr, src1, K,N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, M, K, N, false, false, false, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    L0CCopyOut<T, T, M, N, M, N>(out, c, M, N, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose, int targetM, int targetN, int targetK>
__aicore__ inline void runTMOV_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1) {

    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, targetM, targetK>, Stride<1*targetM*targetK, 1*targetM*targetK, targetM*targetK, targetK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, targetK, targetN>, Stride<1*targetK*targetN, 1*targetK*targetN, targetK*targetN, targetN, 1>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, targetM, targetN>, Stride<1*targetM*targetN, 1*targetM*targetN, targetM*targetN, targetN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, targetM, targetK, BLayout::RowMajor, targetM, targetK, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, targetM, targetK, BLayout::ColMajor, targetM, targetK, SLayout::RowMajor, 512>
    >;
    using TileMatBData = std::conditional_t<
        isBtranspose,
        Tile<Location::Mat, S, targetK, targetN, BLayout::RowMajor, targetK, targetN, SLayout::ColMajor, 512>,
        Tile<Location::Mat, S, targetK, targetN, BLayout::ColMajor, targetK, targetN, SLayout::RowMajor, 512>
    >;
    using LeftTile = TileLeft<U, targetM, targetK, targetM, targetK>;
    using RightTile = TileRight<S, targetK, targetN, targetK, targetN>;
    using AccTile = TileAcc<T, targetM, targetN, targetM, targetN>;

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

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());

    /*************************************GM->L1(NZ2NZ)****************************************/
    DynGM2L1<U>(srcAAddr, src0, targetM, targetK);
    DynGM2L1<U>(srcBAddr, src1, targetK, targetN);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(c, a, b, targetM, K, targetN, false, true, false, true);//使能K16对齐，避免b32读取脏数据
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<T, T, targetM,targetN,targetM,targetN>(out, c, M, N, 0, 0, 0);
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
extern "C" __global__ __aicore__ void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 48;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 64;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 48;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 16;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
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

extern "C" __global__ __aicore__ void launchTMOV_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
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

extern "C" __global__ __aicore__ void launchTMOV_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
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
extern "C" __global__ __aicore__ void launchTMOV_24(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
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
extern "C" __global__ __aicore__ void launchTEXTRACT_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 96;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 64;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 32;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_34(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 80;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 48;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_UNALIGN_41(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 80;
    constexpr uint16_t targetN = 80;
    constexpr uint16_t targetK = 48;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_UNALIGN_42(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 96;
    constexpr uint16_t targetN = 96;
    constexpr uint16_t targetK = 64;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOV_UNALIGN_43(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 80;
    constexpr uint16_t targetN = 80;
    constexpr uint16_t targetK = 48;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, half, half, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOV_UNALIGN_44(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t targetM = 80;
    constexpr uint16_t targetN = 80;
    constexpr uint16_t targetK = 48;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose, targetM, targetN, targetK>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_DYNAMIC_51(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 32;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACTDYNAMIC<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, N, K);
}

extern "C" __global__ __aicore__ void launchTEXTRACT_DYNAMIC_52(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 32;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACTDYNAMIC<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, K, N);
}

extern "C" __global__ __aicore__ void launchTMOV_DYNAMIC_53(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOVDYNAMIC<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, N, K);
}

extern "C" __global__ __aicore__ void launchTMOV_DYNAMIC_54(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOVDYNAMIC<float, half, half, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, N, K);
}
template <int32_t tilingKey>
void launchTEXTRACT_demo(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        // TMOV: 输入为half， A不转置，B不转置
        launchTMOV_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        // TMOV: 输入为int8， A不转置，B不转置
        launchTMOV_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        // TMOV: 输入为float， A不转置，B不转置
        launchTMOV_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        // TMOV: 输入为Bf16， A不转置，B不转置
        launchTMOV_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        // TExtract: 输入为half， A不转置，B不转置 , mIdx = 16, nIdx = 16, kIdx = 32
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        // TExtract: 输入为int8， A不转置，B不转置 , mIdx = 48, nIdx = 32, kIdx = 64
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        // TExtract: 输入为float， A不转置，B不转置 , mIdx = 32, nIdx = 16, kIdx = 48
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 14) {
        // TExtract: 输入为Bf16， A不转置，B不转置 , mIdx = 32, nIdx = 32, kIdx = 16
        launchTEXTRACT_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        // TMOV: 输入为half， A转置，B转置
        launchTMOV_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        // TMOV: 输入为int8， A转置，B转置
        launchTMOV_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        // TMOV: 输入为float， A转置，B转置
        launchTMOV_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 24) {
        // TMOV: 输入为Bf16， A转置，B转置
        launchTMOV_24<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        // TExtract: 输入为half， A转置，B转置 , mIdx = 96, nIdx = 32, kIdx = 64
        launchTEXTRACT_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        // TExtract: 输入为int8， A转置，B转置 , mIdx = 32, nIdx = 32, kIdx = 32
        launchTEXTRACT_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        // TExtract: 输入为float， A转置，B转置 , mIdx = 32, nIdx = 16, kIdx = 16
        launchTEXTRACT_33<<<1, nullptr, stream>>>(out, src0, src1);
    }  else if constexpr (tilingKey == 34) {
        // TExtract: 输入为Bf16， A转置，B转置 , mIdx = 32, nIdx = 64, kIdx = 48
        launchTEXTRACT_34<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 41) {
        // TMOV Unalign: 输入为half， A转置，B转置
        launchTMOV_UNALIGN_41<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 42) {
        // TMOV Unalign: 输入为int8， A转置，B转置
        launchTMOV_UNALIGN_42<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 43) {
        // TMOV Unalign: 输入为float， A转置，B转置
        launchTMOV_UNALIGN_43<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 44) {
        // TMOV Unalign: 输入为Bf16， A转置，B转置
        launchTMOV_UNALIGN_44<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 51) {
        // TExtract Dynamic：输入为half， A不转置，B不转置
        launchTEXTRACT_DYNAMIC_51<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 52) {
        // TExtract Dynamic：输入为int8， A转置，B不转置
        launchTEXTRACT_DYNAMIC_52<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 53) {
       // TMov Dynamic：输入为int8， A转置，B不转置
        launchTMOV_DYNAMIC_53<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 54) {
        // TMov Dynamic：输入为half， A转置，B不转置
        launchTMOV_DYNAMIC_54<<<1, nullptr, stream>>>(out, src0, src1);
    } 
}

template void launchTEXTRACT_demo<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=1 的版本
template void launchTEXTRACT_demo<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<24>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<34>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT_demo<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<43>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<44>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<51>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<52>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<53>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
template void launchTEXTRACT_demo<54>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  
