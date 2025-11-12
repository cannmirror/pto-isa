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

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexRow, uint16_t indexCol, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1) {

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
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using LeftTile = Tile<Location::Left, U, M, K, BLayout::RowMajor, M, K, SLayout::RowMajor, 512>;
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
    TEXTRACT(aTile, aMatTile, indexRow, indexCol);
    TEXTRACT(bTile, bMatTile, indexRow, indexCol);
    // TMOV(aTile, aMatTile);
    // TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, M, K, N, false, false, false, true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    /****************************************TSTORE*****************************************/
    L0CCopyOut<T, T, M,N,M,N>(out, c, M,N,0,0,0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexRow, uint16_t indexCol, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACTUNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1) {

    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    constexpr int mValid = M - indexRow;
    constexpr int kValid = K - indexCol;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>
    >;
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using LeftTile = Tile<Location::Left, U, mValid, kValid, BLayout::RowMajor, mValid, kValid, SLayout::RowMajor, 512>;
    using RightTile = TileRight<S, kValid, N, kValid, N>;
    using AccTile = TileAcc<T, mValid, N, mValid, N>;

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
    TEXTRACT(aTile, aMatTile, indexRow, indexCol);
    TEXTRACT(bTile, bMatTile, indexCol, 0);
    // TMOV(aTile, aMatTile);
    // TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, mValid, kValid, N, false, false, false, true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    L0CCopyOut<T, T, mValid, N, mValid, N>(out, c, mValid, N, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexRow, uint16_t indexCol, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACTUNALIGNDYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int k, int n) {

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

    constexpr int mValid = M - indexRow;
    constexpr int kValid = K - indexCol;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>
    >;
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    using LeftTile = Tile<Location::Left, U, mValid, kValid, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    using RightTile = TileRight<S, kValid, N, kValid, -1>;
    using AccTile = TileAcc<T, mValid, N, -1, N>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    int rowValid = m - indexRow;
    int colValid = k - indexCol;

    LeftTile aTile(rowValid, colValid);
    RightTile bTile(n);
    AccTile cTile(rowValid);
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
    TEXTRACT(aTile, aMatTile, indexRow, indexCol);
    TEXTRACT(bTile, bMatTile, indexCol, 0);
    // TMOV(aTile, aMatTile);
    // TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, mValid, kValid, N, false, false, false, true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    L0CCopyOut<T, T, mValid, N, mValid, N>(out, c, rowValid, N, 0, 0, 0);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexRow, uint16_t indexCol, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTMOVDYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int k, int n) {

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

    constexpr int mValid = M - indexRow;
    constexpr int kValid = K - indexCol;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose,
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>
    >;
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
    using LeftTile = Tile<Location::Left, U, mValid, kValid, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512>;
    using RightTile = TileRight<S, kValid, N, kValid, -1>;
    using AccTile = TileAcc<T, mValid, N, -1, N>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    int rowValid = m - indexRow;
    int colValid = k - indexCol;

    LeftTile aTile(rowValid, colValid);
    RightTile bTile(n);
    AccTile cTile(rowValid);
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
    // mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
    // Indicates the C matrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    mad(c, a, b, mValid, kValid, N, false, false, false, true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    L0CCopyOut<T, T, mValid, N, mValid, N>(out, c, rowValid, N, 0, 0, 0);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexRow = 16;
    constexpr uint16_t indexCol = 32;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGN<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 48;
    constexpr uint16_t indexCol = 64;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGN<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 48;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGN<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 96;
    constexpr uint16_t indexCol = 64;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGN<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 32;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGN<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 16;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGN<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_41(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexRow = 16;
    constexpr uint16_t indexCol = 32;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGNDYNAMIC<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, K, N);
}

extern "C" __global__ __aicore__ void launchTEXTRACT_42(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 32;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACTUNALIGNDYNAMIC<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, K, N);
}

extern "C" __global__ __aicore__ void launchTEXTRACT_43(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOVDYNAMIC<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, K, N);
}

extern "C" __global__ __aicore__ void launchTEXTRACT_44(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOVDYNAMIC<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, K, N);
}

template <int32_t tilingKey>
void launchTEXTRACT_demo(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        // 输入为B16， A不转置，B不转置
        launchTEXTRACT_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        // 输入为B8， A不转置，B转置
        launchTEXTRACT_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        // 输入为B32， A不转置，B转置
        launchTEXTRACT_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        // 输入为B16， A不转置，B转置 , mIdx = 16, kIdx = 32
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        // 输入为B8， A不转置，B转置 , mIdx = 48, kIdx = 64
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        // 输入为B32， A不转置，B转置 , mIdx = 32, kIdx = 48
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        // 输入为B16， A转置，B转置 
        launchTEXTRACT_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        // 输入为B8， A转置，B转置 
        launchTEXTRACT_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        // 输入为B32， A转置，B转置 
        launchTEXTRACT_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        // 输入为B16， A转置，B转置 ,mIdx = 96, kIdx = 64
        launchTEXTRACT_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        // 输入为B8， A转置，B转置 , mIdx = 32, kIdx = 32
        launchTEXTRACT_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        // 输入为B32， A转置，B转置 
        launchTEXTRACT_33<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 41) {
        // 动态tiling 输入为B16， A不转置，B转置 , mIdx = 16, kIdx = 32
        launchTEXTRACT_41<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 42) {
        // 动态tiling 输入为B8， A转置，B转置 , mIdx = 32, kIdx = 32
        launchTEXTRACT_42<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 43) {
        // 动态tiling 输入为B8， A不转置，B转置
        launchTEXTRACT_43<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 44) {
        // 动态tiling 输入为B16， A转置，B转置 
        launchTEXTRACT_44<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTEXTRACT_demo<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=1 的版本
template void launchTEXTRACT_demo<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=2 的版本
template void launchTEXTRACT_demo<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=3 的版本
template void launchTEXTRACT_demo<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=11 的版本
template void launchTEXTRACT_demo<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=12 的版本
template void launchTEXTRACT_demo<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=13 的版本
template void launchTEXTRACT_demo<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=21 的版本
template void launchTEXTRACT_demo<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=22 的版本
template void launchTEXTRACT_demo<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=23 的版本
template void launchTEXTRACT_demo<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=31 的版本
template void launchTEXTRACT_demo<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=32 的版本
template void launchTEXTRACT_demo<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=33 的版本
template void launchTEXTRACT_demo<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=41 的版本
template void launchTEXTRACT_demo<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=42 的版本
template void launchTEXTRACT_demo<43>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=43 的版本
template void launchTEXTRACT_demo<44>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=44 的版本
