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
__aicore__ constexpr inline T CeilDiv(T num_1, T num_2)
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
        copy_gm_to_cbuf_align_v2(dstTmp, srcTmp, 0, nBurst, lenBurst, 0, 0, 0, 0, 0, 0);
    } else {
        copy_gm_to_cbuf_align_v2(dst, src, 0, nBurst, lenBurst, 0, 0, 0, 0, 0, 0);
    }
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
using CType = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

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
__global__ __aicore__ void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
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

template <Layout LayoutType>
__aicore__ inline constexpr BLayout GetTileBLayout()
{
    if constexpr (LayoutType == Layout::NZ) {
        return BLayout::ColMajor;
    } else {
        return BLayout::RowMajor;
    }
}

template <Layout LayoutType>
__aicore__ inline constexpr SLayout GetTileSLayout()
{
    if constexpr (LayoutType == Layout::NZ) {
        return SLayout::RowMajor;
    } else {
        return SLayout::NoneBox;
    }
}

template <typename T, typename GlobalData, typename TileData>
__aicore__ inline void UBCopyOut(GlobalData &dst, TileData &src, int validRow, int rows, int cols)
{
    constexpr uint32_t c0Size = 64;
    int gShape0 = dst.GetShape(0);
    int gShape1 = dst.GetShape(1);
    int gShape4 = dst.GetShape(4);
    int gStride0 = dst.GetStride(0);
    int gStride1 = dst.GetStride(1);

    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * c0Size;
    uint64_t burstDstStride = gStride1 * sizeof(typename TileData::DType);
    uint32_t burstSrcStride = TileData::Rows * c0Size;
    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    typename GlobalData::DType *dstAddr = dst.data();
    __ubuf__ typename TileData::DType *srcAddr = src.data();
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * tileStride;
        copy_ubuf_to_gm_align_v2(dstGlobalAddr, srcTileAddr, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
    }
}

template <typename T, typename U, typename S, int M, int K, int N, int ValidM, int ValidK, int ValidN, Layout LayoutType,
           int SFractalSize, int SubBlockId>
__aicore__ inline void runTMOV_nz2nz(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(SFractalSize, sGRows_ * sizeof(T));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(M, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(N, sGCols_);
    
    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    using DynStrideDim5 = pto::Stride< kGCols_ * kGRows_ * sGCols_ * sGRows_, kGRows_* sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<T, DynShapeDim5, DynStrideDim5, LayoutType>;
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

    using DstTileData = Tile<Location::Vec, T, M, N,
                            GetTileBLayout<LayoutType>(),
                            ValidM, ValidN,
                            GetTileSLayout<LayoutType>(), SFractalSize>;
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
        if (SFractalSize == 512){
            TSTORE(dstGlobal, dstTileData);
        } else {
            UBCopyOut<T, GlobalDataOut, DstTileData>(dstGlobal, dstTileData, ValidM, M, N);
        }
    }

#endif
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int K, int N, bool SplitM>
__global__ __aicore__ void runSplitTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mSize = SplitM ? M / 2 : M;
    constexpr int nSize = SplitM ? N : N / 2;
    using GlobalDataOut = GlobalTensor<T,
        pto::Shape<1, 1, 1, mSize, nSize>,
        pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>,
        Layout::ND>;
    GlobalDataOut dstGlobal1(out);
    constexpr int stride = SplitM ? mSize * nSize : nSize;
    GlobalDataOut dstGlobal2(out + stride);

    using TileMatAData = Tile<Location::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using C = CType<U>;
    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using AccTile = TileAcc<C, M, N, M, N>;

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

    using DstTileData = Tile<Location::Vec, T, M, N, BLayout::RowMajor, mSize, nSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    DynL1CopyIn<U, U>(srcAAddr, src0, M, K, M, K, 0, 0, 0);
    DynL1CopyIn<S, S>(srcBAddr, src1, K, N, K, N, 0, 0, 0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    DynL1ToL0A<U, 0, 0>(a, srcAAddr, M, K, M, K);
    DynL1ToL0B<S, 0, 0>(b, srcBAddr, K, N, K, N);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/

    constexpr int dualDstCtl = SplitM ? 1 : 2;
    constexpr uint8_t mode = getMode<0, dualDstCtl>();
    TMOV<DstTileData, AccTile, static_cast<L0cToUBMode>(mode)>(dstTileData, cTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    if (idx == 0) {
        TSTORE(dstGlobal1, dstTileData);
    } else {
        TSTORE(dstGlobal2, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, typename FbType, int M, int K, int N,
    int ValidM, int ValidK, int ValidN>
__global__ __aicore__ void runVectorQuantTMOV(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ FbType *src2)
{
    using GlobalDataOut =
        GlobalTensor<OutType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, AType, M, K, BLayout::RowMajor, ValidM, ValidK, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, BType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<Location::Mat, FbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>; 

    using C = CType<AType>;
    using LeftTile = TileLeft<AType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<BType, K, N, ValidK, ValidN>;
    using AccTile = TileAcc<C, M, N, ValidM, ValidN>;

    using FbTile = Tile<Location::Scaling, FbType, 1, N, BLayout::RowMajor, 1, ValidN, SLayout::NoneBox>;

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

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();
    __cbuf__ FbType *srcFbAddr = fbMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ C *c = (__cc__ C *)(cTile.data());
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, OutType, M, N, BLayout::RowMajor, ValidM, ValidN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    DynL1CopyIn<AType, AType>(srcAAddr, src0, M, K, M, K, 0, 0, 0);
    DynL1CopyIn<BType, BType>(srcBAddr, src1, K, N, K, N, 0, 0, 0);
    DynGM2L1<FbType>(srcFbAddr, src2, 1, N);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    DynL1ToL0A<AType, 0, 0>(a, srcAAddr, M, K, M, K);
    DynL1ToL0B<BType, 0, 0>(b, srcBAddr, K, N, K, N);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/

    TMOV(fbTile, fbMatTile);

    TMOV<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    if (idx == 0) {
        TSTORE(dstGlobal, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, int M, int K, int N,
    int ValidM, int ValidK, int ValidN>
__global__ __aicore__ void runScalarQuantTMOV(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, float scalar)
{
    using GlobalDataOut = 
        GlobalTensor<OutType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<Location::Mat, AType, M, K, BLayout::RowMajor, ValidM, ValidK, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, BType, K, N, BLayout::ColMajor, ValidK, ValidN, SLayout::RowMajor, 512>;

    using C = CType<AType>;
    using LeftTile = TileLeft<AType, M, K, ValidM, ValidK>;
    using RightTile = TileRight<BType, K, N, ValidK, ValidN>;
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

    __cbuf__ AType *srcAAddr = aMatTile.data();
    __cbuf__ BType *srcBAddr = bMatTile.data();

    __ca__ AType *a = (__ca__ AType *)(aTile.data());
    __cb__ BType *b = (__cb__ BType *)(bTile.data());
    __cc__ C *c = (__cc__ C *)(cTile.data());
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, OutType, M, N, BLayout::RowMajor, ValidM, ValidN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    DynL1CopyIn<AType, AType>(srcAAddr, src0, M, K, M, K, 0, 0, 0);
    DynL1CopyIn<BType, BType>(srcBAddr, src1, K, N, K, N, 0, 0, 0);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    DynL1ToL0A<AType, 0, 0>(a, srcAAddr, M, K, M, K);
    DynL1ToL0B<BType, 0, 0>(b, srcBAddr, K, N, K, N);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    /**********************************TSTORE**********************************/

    uint64_t preScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
    if (sizeof(OutType) == 1) {
        constexpr bool sign = (std::is_same_v<typename DstTileData::DType, int8_t>) ? true : false;
        preScalar = (preScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    TMOV<DstTileData, AccTile>(dstTileData, cTile, preScalar);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    if (idx == 0) {
        TSTORE(dstGlobal, dstTileData);
    }
#endif
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UBNZ2NZ_1(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 16;
    constexpr uint32_t N = 16;
    constexpr uint32_t SFractalSize = 512;

    runTMOV_nz2nz<float, half, half, M, K, N, M, K, N, Layout::NZ, SFractalSize, 0>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}
extern "C" __global__ __aicore__ void launchTMOVL0c2UBNZ2NZ_2(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t SFractalSize = 512;

    runTMOV_nz2nz<float, half, half, M, K, N, M, K, N, Layout::NZ, SFractalSize, 0>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UBNZ2NZ_3(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t SFractalSize = 512;

    runTMOV_nz2nz<half, half, half, M, K, N, M, K, N, Layout::NZ, SFractalSize, 0>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UBNZ2NZ_4(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t SFractalSize = 1024;

    runTMOV_nz2nz<float, half, half, M, K, N, M, K, N, Layout::NZ, SFractalSize, 0>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ __aicore__ void launchTMOVL0c2UBNZ2NZ_5(
    __gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t SFractalSize = 512;

    runTMOV_nz2nz<float, float, float, M, K, N, M, K, N, Layout::NZ, SFractalSize, 0>(reinterpret_cast<__gm__ float *>(out),
        reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

template <int32_t tilingKey>
void launchTMOVL0c2UBNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMOV<float, half, half, 64, 128, 128, 64, 128, 128, 64, 128, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 2) {
        runTMOV<half, half, half, 128, 128, 64, 128, 128, 64, 128, 64, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 3) {
        runTMOV<float, half, half, 64, 64, 64, 64, 64, 64, 64, 128, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        runTMOV<float, half, half, 32, 32, 32, 31, 24, 24, 31, 24, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 5) {
        runTMOV<float, half, half, 32, 32, 64, 32, 32, 64, 32, 64, 1><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 6) {
        runTMOV<bfloat16_t, half, half, 128, 64, 128, 128, 64, 128, 128, 128, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 7) {
        runSplitTMOV<float, half, half, 64, 32, 32, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 8) {
        runSplitTMOV<float, half, half, 32, 32, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}

template void launchTMOVL0c2UBNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2ND<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMOVL0c2UBNZ2NZ_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMOVL0c2UBNZ2NZ_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMOVL0c2UBNZ2NZ_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMOVL0c2UBNZ2NZ_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 5) {
        launchTMOVL0c2UBNZ2NZ_5<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTMOVL0c2UBNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBFBQuant(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        runVectorQuantTMOV<int8_t, int8_t, int8_t, uint64_t, 32, 32, 128, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        runVectorQuantTMOV<half, int8_t, int8_t, uint64_t, 32, 128, 32, 32, 128, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3) {
        runVectorQuantTMOV<bfloat16_t, int8_t, int8_t, uint64_t, 128, 64, 96, 128, 64, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out),
                reinterpret_cast<int8_t *>(src0),
                reinterpret_cast<int8_t *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        runVectorQuantTMOV<int8_t, float, float, uint64_t, 112, 48, 96, 112, 48, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out),
                reinterpret_cast<float *>(src0),
                reinterpret_cast<float *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        runVectorQuantTMOV<half, float, float, uint64_t, 32, 128, 128, 31, 128, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out),
                reinterpret_cast<float *>(src0),
                reinterpret_cast<float *>(src1),
                reinterpret_cast<uint64_t *>(src2));
    }
}

template void launchTMOVL0c2UBFBQuant<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBFBQuant<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBFBQuant<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBFBQuant<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBFBQuant<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBSCQuant(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runScalarQuantTMOV<half, float, float, 112, 48, 96, 112, 48, 96><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        runScalarQuantTMOV<bfloat16_t, float, float, 112, 96, 48, 112, 96, 48><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 5);
    } else if constexpr (tilingKey == 3) {
        runScalarQuantTMOV<half, int8_t, int8_t, 32, 128, 64, 32, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 3);
    } else if constexpr (tilingKey == 4) {
        runScalarQuantTMOV<int8_t, int8_t, int8_t, 32, 32, 32, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 1);
    }
}

template void launchTMOVL0c2UBSCQuant<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuant<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuant<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuant<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);