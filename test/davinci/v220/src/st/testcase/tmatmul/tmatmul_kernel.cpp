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
__aicore__ inline T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

__aicore__ inline unsigned CalcLinearOffset(unsigned GmShape1, unsigned Offset0, unsigned Offset1)
{
    return Offset1 + Offset0 * GmShape1;
}

template <typename GMT, typename L1T>
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, 
    unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved)
{   // ND2NZ
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = GmShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(TShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;

    if constexpr (std::is_same<GMT, int8_t>::value) {
        copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ L1T * )dst,
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

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
__aicore__ inline void L0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src,  unsigned GmShape0, unsigned GmShape1, 
    unsigned GmOffset0, unsigned GmOffset1, int uf)
{ // NZ2ND
    uint16_t MSize = oriTShape0 < (GmShape0 - GmOffset0) ? oriTShape0 :(GmShape0 - GmOffset0);
    uint16_t NSize = TShape1 < (GmShape1 - GmOffset1) ? TShape1 :(GmShape1 - GmOffset1);
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

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline void DynL1ToL0A(
    __ca__ T *dst, __cbuf__ T *src, unsigned dstM, unsigned dstK, unsigned srcM, unsigned srcK)
{
    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(T);
    dstM = CeilAlign<uint16_t>(dstM, BLOCK_CUBE_M_N);
    dstK = CeilAlign<uint16_t>(dstK, blockCubeK);
    srcM = CeilAlign<uint16_t>(srcM, BLOCK_CUBE_M_N);
    srcK = CeilAlign<uint16_t>(srcK, blockCubeK);

    uint8_t repeat = dstK / blockCubeK;
    uint16_t srcStride = srcM / BLOCK_CUBE_M_N;
    uint16_t dstStride = 0;
    int32_t dstOffset = 0;
    int32_t dstOffsetStep = BLOCK_CUBE_M_N * dstK;
    int32_t srcOffset = Offset0 * blockCubeK + Offset1 * srcM;
    int32_t srcOffsetStep = BLOCK_CUBE_M_N * blockCubeK;

    for (int32_t mIdx = 0; mIdx < static_cast<int32_t>(dstM / BLOCK_CUBE_M_N); ++mIdx) {
        load_cbuf_to_ca(dst + dstOffset, src + srcOffset, 0, repeat, srcStride, dstStride, 0, 0, inc);
        dstOffset += dstOffsetStep;
        srcOffset += srcOffsetStep;
    }

}

// Nz2Zn
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline void DynL1ToL0B(
    __cb__ T *dst, __cbuf__ T *src, unsigned dstK, unsigned dstN, unsigned srcK, unsigned srcN)
{
    auto nBlockSize = 32;
    int64_t frac_num = 32 / sizeof(T);

    dstK = (dstK + frac_num - 1) / frac_num * frac_num;
    dstN = (dstN + frac_num - 1) / frac_num * frac_num;
    srcN = (srcN + frac_num - 1) / frac_num * frac_num;
    srcK = (srcK + frac_num - 1) / frac_num * frac_num;

    if constexpr (std::is_same<T, int8_t>::value) {
        for (auto index = 0; index < dstN / nBlockSize; ++index) {
            auto repeatTimes = dstK / nBlockSize;
            auto srcStride = 1;
            auto dstGap = (nBlockSize * dstN - nBlockSize * 16) / (16 * nBlockSize);
            auto dstFracGap = 0;
            load_cbuf_to_cb_transpose(dst + index * nBlockSize * nBlockSize,
                src + Offset0 * nBlockSize + Offset1 *srcK + index * nBlockSize * srcK,
                0,
                repeatTimes,
                srcStride,
                dstGap,
                inc,
                dstFracGap);
        }
        return;
    } 

    if constexpr (std::is_same<T, float>::value) {
        uint8_t repeat = dstN / BLOCK_CUBE_M_N;
        uint16_t srcStride = 1;
        uint16_t dstStride = 0;
        uint16_t dstFracStride = repeat - 1;
        // load_cbuf_to_cb_transpose(dst, src, 0, repeat, srcStride, dstStride, inc, dstFracStride); 
        for (auto index = 0; index < dstK / BLOCK_CUBE_M_N; ++index) { // ToDo: 此处还有点问题
            load_cbuf_to_cb_transpose(
                dst + index * 16 * srcN, src + index * 16 * 8, 0, repeat, srcStride, dstStride, inc, dstFracStride);
        }
        return;
    }
    // L1 n1k1k0no -> k1n1n0k0
    
    int64_t k_frac = dstK / frac_num; // B32
    uint8_t repeat = dstN / 16;
    uint16_t srcStride = srcK / frac_num;
    uint16_t dstStride = 0; //gap
    for (int64_t k_idx = 0; k_idx < k_frac; ++k_idx) {
        load_cbuf_to_cb(dst + k_idx * frac_num * dstN,
            src + k_idx * 16 * frac_num + (Offset0 * 16 + Offset1 * srcK),
            0,
            repeat,
            srcStride,
            dstStride,
            0,
            1,
            inc);
    }
}

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline void DynL1ToL0Bt(
    __cb__ T *dst, __cbuf__ T *src, unsigned dstK, unsigned dstN, unsigned srcN, unsigned srcK)
{
    int64_t frac_num = 32 / sizeof(T);
    dstK = (dstK + frac_num - 1) / frac_num * frac_num;
    dstN = (dstN + frac_num - 1) / frac_num * frac_num;
    srcN = (srcN + frac_num - 1) / frac_num * frac_num;
    srcK = (srcK + frac_num - 1) / frac_num * frac_num;

    if (dstN == srcN) {
        uint8_t repeat = dstN / (32 / sizeof(T)) * dstK / 16;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;

        load_cbuf_to_cb(dst,
            src + (Offset0 * frac_num + Offset1 * srcN),
            (uint16_t)0,
            repeat,
            srcStride,
            dstStride,
            (uint8_t)0,
            (bool)0,
            (addr_cal_mode_t)0);
    } else {
        int64_t k_frac = dstK / frac_num;
        uint8_t repeat = dstN / 16;
        constexpr uint16_t srcStride = 1;
        constexpr uint16_t dstStride = 0;

        for (int64_t k_idx = 0; k_idx < k_frac; ++k_idx) {
        load_cbuf_to_cb(dst + k_idx * frac_num * dstN,
            src + k_idx * srcN * frac_num + (Offset0 * frac_num + Offset1 * srcN),
            (uint16_t)0,
            repeat,
            srcStride,
            dstStride,
            (uint8_t)0,
            (bool)0,
            (addr_cal_mode_t)0);
        }
    }
}    

template <typename T, typename U, typename S, int M, int K, int N>
__aicore__ inline void runTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut  = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    int offset = 0;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = 
        Tile<Location::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // L1上都是大n小z
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

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

    /******************************TLOAD*****************************/

    DynL1CopyIn<U, U>(srcAAddr, src0, M, K, M, K, 0, 0, 0);
    DynL1CopyIn<S, S>(srcBAddr, src1, K, N, K, N, 0, 0, 0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**************************TMOV && TEXTRACT**************************/
    DynL1ToL0A<U, 0, 0>(a, srcAAddr, M, K, M, K);
    DynL1ToL0B<S, 0, 0>(b, srcBAddr, K, N, K, N); // Nz2Zn [K, N]

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);
    
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /********************************TSTORE****************************/
    L0CCopyOut<T, T, M, N, M, N>(out, c, M, N, 0, 0, 0);
    out = dstGlobal.data();
}   


template <typename T, typename U, typename S, int M, int K, int N>
__aicore__ inline void runTMATMUL_SPLIT_K(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, Shape<1, 1, 1, M, K>, Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, Shape<1, 1, 1, K, N>, Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut  = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;
    constexpr int BASEK = 64;
    constexpr int BASEM = 128;
    constexpr int BASEN = 64;
    int offset = 0;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = 
        Tile<Location::Mat, U, BASEM, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>; // L1上都是大n小z
    using TileMatBData = Tile<Location::Mat, S, K, BASEN, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<U, BASEM, BASEK, M, BASEK>;
    using RightTile = TileRight<S, BASEK, BASEN, BASEK, N>; 
    using AccTile = TileAcc<T, 128, BASEN, M, N>; 

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

    constexpr int iter = K / BASEK;

    for (int i = 0; i < iter; i++) { // baseK = 64
        /******************************TLOAD*****************************/
        DynL1CopyIn<U, U>(srcAAddr, src0, M, BASEK, M, K, 0, i * BASEK, 0);
        DynL1CopyIn<S, S>(srcBAddr, src1, BASEK, N, K, N, i * BASEK, 0, 0);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**************************TMOV && TEXTRACT**************************/
        DynL1ToL0A<U, 0, 0>(a, srcAAddr, M, BASEK, M, BASEK);
        DynL1ToL0B<S, 0, 0>(b, srcBAddr, BASEK, N, BASEK, N); // Nz2Zn [K, N]

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            TMATMUL(cTile, aTile, bTile); // L0C清空
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile); // L0C不清空
        }
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    L0CCopyOut<T, T, BASEM, BASEN, M, N>(out, c, M, N, 0, 0, 0);
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTMATMUL_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<int32_t, int8_t, int8_t, M, K, N>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTMATMUL_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, half, half, M, K, N>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMATMUL_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 127;
    constexpr uint32_t N = 63;
    constexpr uint32_t K = 128;

    runTMATMUL_SPLIT_K<float, half, half, M, K, N>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

template <int32_t tilingKey>
void launchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr(tilingKey == 1) {
        launchTMATMUL_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr(tilingKey == 2) {
        launchTMATMUL_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr(tilingKey == 3) {
        launchTMATMUL_3<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); //实例化 tilingKey=1
template void launchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); //实例化 tilingKey=2
template void launchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); //实例化 tilingKey=3
