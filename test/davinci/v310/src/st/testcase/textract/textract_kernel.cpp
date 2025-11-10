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

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T>
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1, unsigned GmShape0,
    unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved) { // ND2NZ
    // src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = TShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(GmShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;   // 源操作数相邻ND矩阵起始地址间的偏移
    constexpr uint16_t dstNzNStride = 1;        // 目的NZ矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移
    constexpr uint16_t dstNzMatrixStride = 0;   // 目的NZ矩阵中，相邻NZ矩阵起始地址间的偏移

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

template <typename T, typename U, typename S, int M, int K, int N, uint16_t indexRow, uint16_t indexCol, bool isAtranspose, bool isBtranspose>
__aicore__ inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    // static shape
    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

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

    using LeftTile = TileLeft<U, mValid, kValid, mValid, kValid>;
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

    // DynL1CopyIn<U, U>(srcAAddr, src0, mValid, kValid, mValid,kValid,0,0,0);
    // DynL1CopyIn<S, S>(srcBAddr, src1, kValid, N, kValid,N,0,0,0);

    DynL1CopyIn<U, U>(srcAAddr, src0, M, K, M,K,0,0,0);
    DynL1CopyIn<S, S>(srcBAddr, src1, K, N, K,N,0,0,0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexRow, indexCol);
    DynL1ToL0B<S, 0, 0>(b, srcBAddr, K, N, K, N ); // Nz2Zn [K,N]
    // TMOV(aTile, aMatTile);       // L1 -> L0A // Todo
    // TMOV(bTile, bMatTile);       // L1 -> L0B // Todo

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    mad(c, a, b, mValid, kValid, N, false, false, false, true);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /****************************************TSTORE*****************************************/

    L0CCopyOut<T, T, mValid, N, mValid, N>(out, c, mValid, N, 0, 0, 0);

    out = dstGlobal.data();

}


extern "C" __global__ __aicore__ void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 48;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ __aicore__ void launchTEXTRACT_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 96;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_5(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_6(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 32;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_7(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, half, half, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_8(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT<float, float, float, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ __aicore__ void launchTEXTRACT_9(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    constexpr uint16_t indexRow = 0;
    constexpr uint16_t indexCol = 0;
    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;
    runTEXTRACT<int32_t, int8_t, int8_t, M, K, N, indexRow, indexCol, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
        reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
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
    } else if constexpr (tilingKey == 5) {
        launchTEXTRACT_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTEXTRACT_6<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 7) {
        launchTEXTRACT_7<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 8) {
        launchTEXTRACT_8<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 9) {
        launchTEXTRACT_9<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTEXTRACT<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);  // 实例化 Key=1 的版本
template void launchTEXTRACT<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
template void launchTEXTRACT<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream); 
