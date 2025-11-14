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

/*
 * brief: dynamic l1 copy in nd2nz functions
 */
template <typename GMT, typename L1T>
__aicore__ inline void DynL1CopyIn(__cbuf__ L1T *dst, __gm__ GMT *src, unsigned TShape0, unsigned TShape1,
    unsigned GmShape0, unsigned GmShape1, unsigned GmOffset0, unsigned GmOffset1, int reserved)
{
    src += CalcLinearOffset(GmShape1, GmOffset0, GmOffset1);
    uint16_t nValue = TShape0;
    uint16_t dValue = TShape1;
    uint16_t srcDValue = GmShape1;
    uint16_t dstNzC0Stride = CeilAlign<uint16_t>(GmShape0, BLOCK_CUBE_M_N);

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t srcNdMatrixStride = 0;
    constexpr uint16_t dstNzNStride = 1;
    constexpr uint16_t dstNzMatrixStride = 1;

    auto c0Size = 32 / sizeof(GMT);
    uint64_t loop1SrcStride = srcDValue * sizeof(GMT);
    uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(GMT);
    
    uint16_t loop2DstStride = dstNzNStride;
    uint16_t loop3DstStride = dstNzC0Stride;
    uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(GMT) / c0Size);
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48;
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;
    mte2NzPara |= static_cast<uint64_t>(ndNum);
    set_mte2_nz_para(mte2NzPara);

    copy_gm_to_cbuf_multi_nd2nz(
        (__cbuf__ L1T *)dst, (__gm__ GMT *)src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue, loop4SrcStride, false);
}

template <typename T>
__aicore__ inline void DynL1CopyInND(__cbuf__ T *dst, __gm__ T *src, unsigned TShape)
{
    uint16_t burstLen = TShape * sizeof(T);
    copy_gm_to_cbuf_align_v2(dst, src, 0, 1, burstLen, 0, 0, 0, 0, 0, 0);
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

    uint16_t srcStride = srcM / 16;
    uint16_t dstStride = dstM / 16;
    uint16_t mStep = dstM / 16;
    uint16_t kStep = dstK * sizeof(T) / 32;

    load_cbuf_to_ca(dst, src, 0, 0, mStep, kStep, srcStride, dstStride, 0);
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

    uint16_t srcStride = srcK / 16;
    uint16_t dstStride = dstN / 16;
    uint16_t mStep = dstK / 16;
    uint16_t kStep = dstN * sizeof(T) / 32;

    load_cbuf_to_cb(dst, src, 0, 0, mStep, kStep, srcStride, dstStride, 1);
}

template <typename T>
using CType = typename std::conditional<std::is_same<T, uint8_t>::value, int32_t, float>::type;

template <int SubBlockId, int DualDstCtl>
__aicore__ inline constexpr uint8_t getMode()
{
    if constexpr (DualDstCtl == 0) {
        return SubBlockId;
    }
    return 1 + DualDstCtl;
}

template <typename T, typename U, typename S, int M, int K, int N, int ValidM, int ValidK, int ValidN, int Row, int Col,
    int SubBlockId>
__aicore__ inline void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, ValidM, ValidN>,
        pto::Stride<1 * ValidM * ValidN, 1 * ValidM * ValidN, ValidM * ValidN, ValidN, 1>,
        Layout::ND>;
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<Location::Mat, U, M, K, BLayout::RowMajor, ValidM, ValidK, SLayout::ColMajor, 512>;
    using TileMatBData =
        Tile<Location::Mat, S, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;

    using C = CType<U>;
    using LeftTile = TileLeft<U, M, K, ValidM, ValidK>;
    using RightTile = TileRight<S, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<C, M, N, ValidM, ValidN>;

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
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, T, Row, Col, BLayout::RowMajor, ValidM, ValidN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    DynL1CopyIn<U, U>(srcAAddr, src0, ValidM, ValidK, ValidM, ValidK, 0, 0, 0);
    DynL1CopyIn<S, S>(srcBAddr, src1, ValidK, ValidN, ValidK, ValidN, 0, 0, 0);
    
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    DynL1ToL0A<U, 0, 0>(a, srcAAddr, M, K, M, K );
    DynL1ToL0B<S, 0, 0>(b, srcBAddr, K, N, K, N ); // Nz2Zn [K,N]

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/
    constexpr uint8_t mode = getMode<SubBlockId, 0>();
    if (SubBlockId == 0) {
        TMOV(dstTileData, cTile);
    } else {
        TMOV<DstTileData, AccTile, static_cast<L0cToUBMode>(mode)>(dstTileData, cTile);
    }

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    if (idx == SubBlockId) {
        TSTORE(dstGlobal, dstTileData);
    }

#endif
    out = dstGlobal.data();
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UB_1(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 128;

    runTMOV<float, half, half, M, K, N, M, K, N, M, N, 0>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UB_2(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;

    runTMOV<half, half, half, M, K, N, M, K, N, M, N, 0>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UB_3(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t Row = 64;
    constexpr uint32_t Col = 128;

    runTMOV<float, half, half, M, K, N, M, K, N, Row, Col, 0>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UB_4(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 32;
    constexpr uint32_t N = 32;
    constexpr uint32_t ValidM = 31;
    constexpr uint32_t ValidK = 24;
    constexpr uint32_t ValidN = 24;

    runTMOV<float, half, half, M, K, N, ValidM, ValidK, ValidN, ValidM, ValidN, 0>(
        reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UB_5(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 32;
    constexpr uint32_t N = 64;

    runTMOV<float, half, half, M, K, N, M, K, N, M, N, 1>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UB_6(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 128;

    runTMOV<bfloat16_t, half, half, M, K, N, M, K, N, M, N, 0>(reinterpret_cast<__gm__ bfloat16_t *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

template <int32_t tilingKey>
void launchTMOVL0c2UB(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMOVL0c2UB_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMOVL0c2UB_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMOVL0c2UB_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMOVL0c2UB_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTMOVL0c2UB_5<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 6) {
        launchTMOVL0c2UB_6<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTMOVL0c2UB<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UB<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UB<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UB<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UB<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UB<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);