#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

using namespace pto;

constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;
template <typename T>
__aicore__ inline  T CeilAlign(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
__aicore__ inline  T CeilDiv(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

__aicore__ inline unsigned CalcLinearOffset(unsigned GmShape1, unsigned Offset0, unsigned Offset1)
{
    return Offset1 + Offset0 * GmShape1;
}

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T>
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved) { // ND2NZ
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = GmShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(GmShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;   // 源操作数相邻ND矩阵起始地址间的偏移
    constexpr uint16_t dstNzNStride = 1;        // 目的NZ矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移
    constexpr uint16_t dstNzMatrixStride = 1;   // 目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移

    auto c0Size = 32 / sizeof(GMT);
    uint64_t loop1SrcStride = srcDValue * sizeof(GMT);
    uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(GMT); //
    
    uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
    uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
    // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_size
    uint16_t loop4DstStride =
        static_cast<uint16_t>(dstNzMatrixStride * sizeof(GMT) / c0Size);
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);            // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);   // CCE: store parameters for ND2NZ DMA instructions

    copy_gm_to_cbuf_multi_nd2nz((__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, loop1SrcStride, 0,
        nValue, dValue, loop4SrcStride, false);
}

template <typename T>
__aicore__ inline void DynL1CopyInND(__cbuf__ T *dst, __gm__ T *src, unsigned TShape)
{
    uint16_t burstLen = TShape * sizeof(T);
    copy_gm_to_cbuf_align_v2(dst, src, 0, 1, burstLen, 0, 0, 0, 0, 0, 0);
}

// Nz2Zz
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline  void DynL1ToL0A(__ca__ T *dst, __cbuf__ T *src, unsigned dstM, unsigned dstK, unsigned srcM, unsigned srcK) {
    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(T);
    dstM = CeilAlign<uint16_t>(dstM, BLOCK_CUBE_M_N);
    dstK = CeilAlign<uint16_t>(dstK, blockCubeK);
    srcM = CeilAlign<uint16_t>(srcM, BLOCK_CUBE_M_N);
    srcK = CeilAlign<uint16_t>(srcK, blockCubeK);

    uint16_t srcStride = srcM / 16;
    uint16_t dstStride = dstM / 16;
    uint16_t mStep = dstM / 16;
    uint16_t kStep = dstK * sizeof(T) / 32;

    load_cbuf_to_ca(dst, src, 0, 0, mStep, kStep, srcStride, dstStride, 0);
}

// Nz2Zn
template <typename T, unsigned Offset0, unsigned Offset1>
__aicore__ inline  void DynL1ToL0B(__cb__ T *dst, __cbuf__ T *src, unsigned dstK, unsigned dstN, unsigned srcK, unsigned srcN) {
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

template <typename T, typename B>
__aicore__ inline void DynL1ToBt(uint64_t dst, __cbuf__ B *src, unsigned len) {
    constexpr uint16_t blockCubeK = BLOCK_ALIGN_BYTE / sizeof(B);
    bool convControl = false;
    if constexpr (std::is_same_v<B, half> && std::is_same_v<T, float>) {
        convControl = true;
    }
    uint16_t nBurst = 1;
    uint16_t blockLen = CeilAlign<uint16_t>(len, blockCubeK) / blockCubeK;
    uint16_t sourceGap = 0;
    uint16_t dstGap = 0;
    
    copy_cbuf_to_bt(dst, src, convControl, nBurst, blockLen, sourceGap, dstGap);
}

template <typename GMT, typename L0CT, unsigned TShape0, unsigned TShape1, unsigned oriTShape0, unsigned oriTShape1>
__aicore__ inline void L0CCopyOut(__gm__ GMT *dst, __cc__ L0CT *src, unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int uf) { // NZ2ND
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
        if (std::is_same<GMT, half>::value) {
            QuantPRE = QuantMode_t::F322F16;
        } else if (std::is_same<GMT, bfloat16_t>::value) {
            QuantPRE = QuantMode_t::F322BF16;
        } else {
            QuantPRE = QuantMode_t::NoQuant;
        }
    }
    uint64_t config = (static_cast<uint64_t>(dst_nd_stride) << 32) | (static_cast<uint64_t>(src_nd_stride) << 16) |
                        (static_cast<uint64_t>(ndNum));
    set_loop3_para(config);
    copy_matrix_cc_to_gm((__gm__ GMT *)(dst + (GmOffset0 * GmShape1) + GmOffset1), (__cc__ L0CT *)src, 0, NSize, MSize,
        dstStride_dst_D, srcStride, 0, 0, UnitFlagMode,
        QuantPRE, ReLUPRE, 0, NZ2ND_EN, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0);
}

template <typename T, typename U, typename S, typename B, int M, int K, int N, int ValidM, int ValidK, int ValidN, bool is_bias>
__aicore__ inline void runTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, U, M, K, BLayout::RowMajor, ValidM, ValidK, SLayout::ColMajor, 512>;//大N小Z
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;//大Z小N
    using TileBiasData = Tile<Location::Mat, B, 1, N, BLayout::RowMajor, 1, N>;

    using LeftTile = TileLeft<U, M, K, ValidM, ValidK>;
    using RightTile = TileRight<S, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<T, M, N, ValidM, ValidN>;
    using BiasTile = Tile<Location::Bias, T, 1, N, BLayout::RowMajor, 1, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

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
    using BiasType = typename TileBiasData::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();
    __cbuf__ BiasType *srcBiasAddr = biasDataTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());
    uint64_t bias = biasTile.data();

    /*************************************TLOAD****************************************/
    DynL1CopyIn<U, U>(srcAAddr, src0, ValidM, ValidK, ValidM, ValidK, 0,0,0);
    DynL1CopyIn<S, S>(srcBAddr, src1, ValidK, ValidN, ValidK, ValidN, 0,0,0);
    if constexpr(is_bias) {
        DynL1CopyInND<B>(srcBiasAddr, src2, N);
    }
    
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    DynL1ToL0A<U, 0, 0>(a, srcAAddr, M, K, M, K );
    DynL1ToL0B<S, 0, 0>(b, srcBAddr, K, N, K, N ); // Nz2Zn [K,N]
    if constexpr(is_bias) {
        DynL1ToBt<T, B>(bias, srcBiasAddr, ValidN);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);


    /**********************************TMATMUL**********************************/
    if constexpr(is_bias) {
        TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
    } else {
        TMATMUL(cTile, aTile, bTile);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/
    L0CCopyOut<T, T, M, N, ValidM, ValidN>(out, c, ValidM, ValidN, 0, 0, 0);

    out = dstGlobal.data();
}

template <typename T, typename U, typename S, typename B, int M, int K, int N, bool is_bias>
__aicore__ inline void runTMATMUL_SPLIT_K(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;
    constexpr int BASEK = 64;
    constexpr int BASEM = 128;
    constexpr int BASEN = 64;

    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, U, BASEM, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, S, K, BASEN, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<Location::Mat, B, 1, BASEN, BLayout::RowMajor, 1, BASEN>;

    using LeftTile = TileLeft<U, BASEM, BASEK, M, BASEK>;
    using RightTile = TileRight<S, BASEK, BASEN, BASEK, N>;
    using AccTile = TileAcc<T, BASEM, BASEN, M, N>;
    using BiasTile = Tile<Location::Bias, T, 1, BASEN, BLayout::RowMajor, 1, BASEN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(biasDataTile, 0x20000);

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
    using BiasType = typename TileBiasData::DType;

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();
    __cbuf__ BiasType *srcBiasAddr = biasDataTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ CType *c = (__cc__ CType *)(cTile.data());
    uint64_t bias = biasTile.data();

    constexpr int iter = K / BASEK;
    for (int i = 0; i < iter; i++) {  // baseK = 64
        /*************************************TLOAD****************************************/
        DynL1CopyIn<U, U>(srcAAddr, src0, M, BASEK, M, K, 0, i * BASEK, 0);
        DynL1CopyIn<S, S>(srcBAddr, src1, BASEK, N, BASEK, N, i * BASEK, 0, 0);
        if constexpr(is_bias) {
            DynL1CopyInND<B>(srcBiasAddr, src2, BASEN);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        /**********************************TMOV && TEXTRACT**********************************/
        DynL1ToL0A<U, 0, 0>(a, srcAAddr, M, BASEK, M, BASEK);
        DynL1ToL0B<S, 0, 0>(b, srcBAddr, BASEK, N, BASEK, N);  // Nz2Zn [K,N]
        if constexpr(is_bias) {
            DynL1ToBt<B>(bias, srcBiasAddr, BASEN);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            if constexpr(is_bias) {
                TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
            } else {
                TMATMUL(cTile, aTile, bTile);
            }
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);  // L0C不清空
        }
        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    L0CCopyOut<T, T, BASEM, BASEN, M, N>(out, c, M, N, 0, 0, 0);

    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTMATMUL_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t ValidM = 127;

    runTMATMUL<float, half, half, float, M, K, N, ValidM, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t ValidK = 127;

    runTMATMUL<int32_t, int8_t, int8_t, int8_t, M, K, N, M, ValidK, N, false>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 127;
    constexpr uint32_t N = 61;
    constexpr uint32_t K = 128;

    runTMATMUL_SPLIT_K<float, half, half, float, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t ValidM = 127;
    constexpr uint32_t ValidN = 63;
    constexpr uint32_t ValidK = 127;

    runTMATMUL<float, float, float, float, M, K, N, ValidM, ValidK, ValidN, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, bfloat16_t, bfloat16_t, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e4m3_t, float8_e4m3_t, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e4m3_t, float8_e5m2_t, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e5m2_t, float8_e4m3_t, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e5m2_t, float8_e5m2_t, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1),
        nullptr);
}

extern "C" __global__ __aicore__ void launchTMATMUL_10(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, hifloat8_t, hifloat8_t, float, M, K, N, M, K, N, false>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1),
        nullptr);
}

template <int32_t tilingKey>
void launchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMATMUL_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMATMUL_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMATMUL_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMATMUL_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTMATMUL_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTMATMUL_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTMATMUL_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTMATMUL_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTMATMUL_9<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 10) {
        launchTMATMUL_10<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTMATMUL<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMATMUL<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

extern "C" __global__ __aicore__ void launchTMATMULBIAS_1(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<int32_t, int8_t, int8_t, int32_t, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1),
        reinterpret_cast<__gm__ int32_t *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_2(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, half, half, half, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        reinterpret_cast<__gm__ half *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_3(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t ValidK = 127;

    runTMATMUL<float, half, half, bfloat16_t, M, K, N, M, ValidK, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        reinterpret_cast<__gm__ bfloat16_t *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_4(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t ValidN = 63;

    runTMATMUL<float, bfloat16_t, bfloat16_t, bfloat16_t, M, K, N, M, K, ValidN, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1),
        reinterpret_cast<__gm__ bfloat16_t *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_5(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 127;
    constexpr uint32_t N = 63;
    constexpr uint32_t K = 128;

    runTMATMUL_SPLIT_K<float, half, half, float, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_6(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t ValidM = 127;
    constexpr uint32_t ValidN = 63;
    constexpr uint32_t ValidK = 128;

    runTMATMUL<float, float, float, float, M, K, N, ValidM, ValidK, ValidN, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_7(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e4m3_t, float8_e4m3_t, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_8(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e4m3_t, float8_e5m2_t, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_9(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e5m2_t, float8_e4m3_t, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e4m3_t *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_10(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, float8_e5m2_t, float8_e5m2_t, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src0),
        reinterpret_cast<__gm__ float8_e5m2_t *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

extern "C" __global__ __aicore__ void launchTMATMULBIAS_11(__gm__ uint8_t *out, __gm__ uint8_t *src0,
        __gm__ uint8_t *src1, __gm__ uint8_t *src2)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    runTMATMUL<float, hifloat8_t, hifloat8_t, float, M, K, N, M, K, N, true>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ hifloat8_t *>(src0),
        reinterpret_cast<__gm__ hifloat8_t *>(src1),
        reinterpret_cast<__gm__ float *>(src2));
}

template <int32_t tilingKey>
void launchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMATMULBIAS_1<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 2) {
        launchTMATMULBIAS_2<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 3) {
        launchTMATMULBIAS_3<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 4) {
        launchTMATMULBIAS_4<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 5) {
        launchTMATMULBIAS_5<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 6) {
        launchTMATMULBIAS_6<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 7) {
        launchTMATMULBIAS_7<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 8) {
        launchTMATMULBIAS_8<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 9) {
        launchTMATMULBIAS_9<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 10) {
        launchTMATMULBIAS_10<<<1, nullptr, stream>>>(out, src0, src1, src2);
    } else if constexpr (tilingKey == 11) {
        launchTMATMULBIAS_11<<<1, nullptr, stream>>>(out, src0, src1, src2);
    }
}

template void launchTMATMULBIAS<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMATMULBIAS<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
