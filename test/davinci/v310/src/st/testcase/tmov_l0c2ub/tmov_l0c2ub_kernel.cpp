#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>

using namespace pto;

constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;

template <typename T>
__aicore__ constexpr inline T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <typename T>
using CType = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

template <int subBlockId, int DualDstCtl>
__aicore__ inline constexpr uint8_t getMode()
{
    if constexpr (DualDstCtl == 0) {
        return subBlockId;
    }
    return 1 + DualDstCtl;
}

template <typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN>
__aicore__ inline void runMATMUL(__gm__ aType *src0, __gm__ bType *src1)
{
    using GlobalDataSrc0 = GlobalTensor<aType,
        pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType,
        pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    using LeftTile = TileLeft<aType, M, K, validM, validK>;
    using RightTile = TileRight<bType, K, N, validK, validN>;
    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
#if defined(__DAV_CUBE__)
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

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
}

template <typename aType, typename bType, typename fbType, int M, int K, int N, int validM, int validK, int validN>
__aicore__ inline void runMATMULFB(__gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 = GlobalTensor<aType,
        pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<bType,
        pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<fbType,
        pto::Shape<1, 1, 1, 1, validN>,
        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileMatFbData = Tile<Location::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(fbMatTile, 0x20000);
    __cbuf__ fbType *srcFbAddr = fbMatTile.data();

    using LeftTile = TileLeft<aType, M, K, validM, validK>;
    using RightTile = TileRight<bType, K, N, validK, validN>;
    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(fbMatTile, src2Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
}

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN,
    int row, int col, int subBlockId>
__global__ __aicore__ void runTMOV(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    using GlobalDataOut = GlobalTensor<outType,
        pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    uint8_t syncId = 0;
    using DstTileData = Tile<Location::Vec, outType, row, col, BLayout::RowMajor, validM, validN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if (subBlockId == 0) {
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

    if (idx == subBlockId) {
        TSTORE(dstGlobal, dstTileData);
    }

#endif
    out = dstGlobal.data();
}

template <Layout layoutType>
__aicore__ inline constexpr BLayout GetTileBLayout()
{
    if constexpr (layoutType == Layout::NZ) {
        return BLayout::ColMajor;
    } else {
        return BLayout::RowMajor;
    }
}

template <Layout layoutType>
__aicore__ inline constexpr SLayout GetTileSLayout()
{
    if constexpr (layoutType == Layout::NZ) {
        return SLayout::RowMajor;
    } else {
        return SLayout::NoneBox;
    }
}

template <typename T, typename GlobalData, typename TileData>
__aicore__ inline void UBCopyOut(GlobalData &dst, TileData &src, int rows, int cols, int startDstAddr)
{
    constexpr uint32_t c0Size = 64;
    int gShape0 = dst.GetShape(0);
    int gShape1 = dst.GetShape(1);
    int gShape4 = dst.GetShape(4);
    int gStride0 = dst.GetStride(0);
    int gStride1 = dst.GetStride(1);

    uint16_t nBurst = gShape1;
    uint32_t lenBurst = rows * c0Size;
    uint64_t burstDstStride = gStride1 * sizeof(typename TileData::DType);
    uint32_t burstSrcStride = TileData::Rows * c0Size;
    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    typename GlobalData::DType *dstAddr = dst.data();
    __ubuf__ typename TileData::DType *srcAddr = src.data();
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * tileStride + startDstAddr;
        copy_ubuf_to_gm_align_v2(dstGlobalAddr, srcTileAddr, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
    }
}

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN,
    Layout layoutType, int sfractalSize, int subBlockId>
__global__ __aicore__ void runTMOV_nz2nz(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(sfractalSize, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(M, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(N, sGCols_);
    
    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    using DynStridDim5 =
        pto::Stride< kGCols_ * kGRows_ * sGCols_ * sGRows_, kGRows_* sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStridDim5, layoutType>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, M, K, N>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N,
                            GetTileBLayout<layoutType>(),
                            validM, validN,
                            GetTileSLayout<layoutType>(), sfractalSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if (subBlockId == 0) {
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

    if (idx == subBlockId) {
        if (sfractalSize == 512) {
            TSTORE(dstGlobal, dstTileData);
        } else {
            UBCopyOut<outType, GlobalDataOut, DstTileData>(dstGlobal, dstTileData, M, N, 0);
        }
    }

#endif
    out = dstGlobal.data();
}

template <typename outType, typename aType, typename bType, int M, int K, int N, bool splitM>
__global__ __aicore__ void runSplitNTMOV_nz2nz(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    constexpr int mSize = splitM ? M / 2 : M;
    constexpr int nSize = splitM ? N : N / 2;
    constexpr int sFractalSize = std::is_same_v<outType, float> ? 1024 : 512;   // float:1024, other:512
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(sFractalSize, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(mSize, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(nSize, sGCols_);
    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;

    
    constexpr uint32_t gShape2 = CeilDiv<uint16_t>(M, sGRows_);
    constexpr uint32_t gShape3 = CeilDiv<uint16_t>(N, sGRows_);
    using DynStrideDim5 = pto::Stride< gShape3 * gShape2 * sGCols_ * sGRows_, 
                                       gShape2 * sGCols_ * sGRows_, 
                                       sGCols_ * sGRows_, 
                                       sGCols_, 
                                       1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStrideDim5, Layout::NZ>;
    GlobalDataOut dstGlobal1(out);
    constexpr int stride = splitM ? mSize * sGCols_ : M * nSize;
    GlobalDataOut dstGlobal2(out + stride);

    runMATMUL<aType, bType, M, K, N, M, K, N>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, M, N>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::ColMajor, 
                             mSize, nSize, SLayout::RowMajor, sFractalSize>;  //nz
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr int dualDstCtl = splitM ? 1 : 2;
    constexpr uint64_t mode = getMode<0, dualDstCtl>();
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
        if (sFractalSize == 512){
            TSTORE(dstGlobal1, dstTileData);
        } else {
            UBCopyOut<outType, GlobalDataOut, DstTileData>(dstGlobal1, dstTileData, M, N, 0);
        }
    } else {
        if (sFractalSize == 512){
            TSTORE(dstGlobal2, dstTileData);
        } else {
            UBCopyOut<outType, GlobalDataOut, DstTileData>(dstGlobal2, dstTileData, M, N, 0);
        }
    }
#endif
}

template <typename outType, typename aType, typename bType, int M, int K, int N, bool splitM>
__global__ __aicore__ void runSplitMTMOV_nz2nz(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    constexpr int mSize = splitM ? M / 2 : M;
    constexpr int nSize = splitM ? N : N / 2;
    constexpr int sFractalSize = std::is_same_v<outType, float> ? 1024 : 512;   // float:1024, other:512
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(sFractalSize, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(mSize, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(nSize, sGRows_);
    using DynShapeDim5 = Shape<1, 1, 1, sGRows_, sGCols_>;

    using DynStrideDim5 = pto::Stride< sGCols_ * sGRows_, 
                                       sGCols_ * sGRows_, 
                                       sGCols_ * sGRows_, 
                                       sGCols_, 
                                       1>;
    
    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStrideDim5, Layout::NZ>;
    constexpr int stride = splitM ? mSize * sGCols_ : M * nSize;

    using TileMatAData = Tile<Location::Mat, aType, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<Location::Mat, bType, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    runMATMUL<aType, bType, M, K, N, M, K, N>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, M, N>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    
    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::ColMajor, mSize, nSize, SLayout::RowMajor, sFractalSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr int dualDstCtl = splitM ? 1 : 2;
    constexpr uint64_t mode = getMode<0, dualDstCtl>();
    TMOV<DstTileData, AccTile, static_cast<L0cToUBMode>(mode)>(dstTileData, cTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    constexpr uint16_t sFractalColNum = CeilDiv<uint16_t>(nSize, sGRows_);
    if (idx == 0) {
        for (int i = 0 ; i < sFractalColNum ; i++){
            GlobalDataOut dstGlobal1(out + 2 * stride * i);
            if (sFractalSize == 512) {
                TSTORE(dstGlobal1, dstTileData);
            } else {
                uint16_t startDstAddr = 2*i*stride;
                UBCopyOut<outType, GlobalDataOut, DstTileData>(dstGlobal1, dstTileData, mSize, nSize, startDstAddr);
            }
        }
        
    } else {
        for (int i = 0 ; i < sFractalColNum ; i++){
            GlobalDataOut dstGlobal2(out + stride + 2 * stride * i);
            if (sFractalSize == 512) {
                TSTORE(dstGlobal2, dstTileData);
            } else {
                uint16_t startDstAddr = 2*i*stride;
                UBCopyOut<outType, GlobalDataOut, DstTileData>(dstGlobal2, dstTileData, mSize, nSize, startDstAddr);
            }
        }    
    }
#endif
}

template <typename outType, typename aType, typename bType, int M, int K, int N, bool splitM>
__global__ __aicore__ void runSplitTMOV(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    constexpr int mSize = splitM ? M / 2 : M;
    constexpr int nSize = splitM ? N : N / 2;
    using GlobalDataOut = GlobalTensor<outType, pto::Shape<1, 1, 1, mSize, nSize>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>, Layout::ND>;
    GlobalDataOut dstGlobal1(out);
    constexpr int stride = splitM ? mSize * nSize : nSize;
    GlobalDataOut dstGlobal2(out + stride);

    runMATMUL<aType, bType, M, K, N, M, K, N>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, M, N>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::RowMajor, mSize, nSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr int dualDstCtl = splitM ? 1 : 2;
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

template <typename outType, typename aType, typename bType, typename fbType, int M, int K, int N,
    int validM, int validK, int validN>
__global__ __aicore__ void runVectorQuantTMOV(
    __gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataOut =
        GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    runMATMULFB<aType, bType, fbType, M, K, N, validM, validK, validN>(src0, src1, src2);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using TileMatFbData = Tile<Location::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    TileMatFbData fbMatTile;
    TASSIGN(fbMatTile, 0x20000);
    using FbTile = Tile<Location::Scaling, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    FbTile fbTile;
    TASSIGN(fbTile, 0x0);

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::RowMajor, validM, validN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

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

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN>
__global__ __aicore__ void runScalarQuantTMOV(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, float scalar)
{
    using GlobalDataOut = 
        GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::RowMajor, validM, validN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    uint64_t preScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
    if (sizeof(outType) == 1) {
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

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN>
__global__ __aicore__ void runScalarQuantTMOVNz2Dn(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, float scalar)
{
    using GlobalDataOut = 
        GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, 1, M>, Layout::DN>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::ColMajor, validM, validN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    uint64_t preScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
    if (sizeof(outType) == 1) {
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

template <typename outType, typename aType, typename bType, typename fbType, int M, int K, int N,
    int validM, int validK, int validN>
__global__ __aicore__ void runVectorQuantTMOV_nz2nz(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(512, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(M, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(N, sGCols_);
    
    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    using DynStridDim5 = pto::Stride< kGCols_ * kGRows_ * sGCols_ * sGRows_, kGRows_* sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    GlobalDataOut dstGlobal(out);
    using TileMatFbData = Tile<Location::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>; 

    TileMatFbData fbMatTile;
    TASSIGN(fbMatTile, 0x20000);

    runMATMULFB<aType, bType, fbType, M, K, N, validM, validK, validN>(src0, src1, src2);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    using FbTile = Tile<Location::Scaling, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    FbTile fbTile;
    TASSIGN(fbTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::ColMajor,
                             validM, validN, SLayout::RowMajor, 512>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    TMOV(fbTile, fbMatTile);  // L1-> FB1

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

template <typename outType, typename aType, typename bType, int M, int K, int N,
    int validM, int validK, int validN>
__global__ __aicore__ void runScalarQuantTMOV_nz2nz(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, float scalar)
{
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(512, sGRows_ * sizeof(outType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(M, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(N, sGCols_);
    
    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
    using DynStridDim5 = pto::Stride< kGCols_ * kGRows_ * sGCols_ * sGRows_, kGRows_* sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<outType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::ColMajor,
                             validM, validN, SLayout::RowMajor, 512>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    uint64_t preQuantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scalar));
    if (sizeof(outType) == 1) {
        constexpr bool sign = (std::is_same_v<typename DstTileData::DType, int8_t>) ? true : false;
        preQuantScalar = (preQuantScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    TMOV<DstTileData, AccTile>(dstTileData, cTile, preQuantScalar);

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
        runTMOV_nz2nz<float, half, half, 16, 16, 16, 16, 16, 16, Layout::NZ, 512, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 2) {
        runTMOV_nz2nz<float, half, half, 128, 128, 64, 128, 128, 64, Layout::NZ, 512, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 3) {
        runTMOV_nz2nz<half, half, half, 128, 128, 64, 128, 128, 64, Layout::NZ, 512, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        runTMOV_nz2nz<float, half, half, 128, 128, 64, 128, 128, 64, Layout::NZ, 1024, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 5) {
        runTMOV_nz2nz<float, float, float, 128, 128, 64, 128, 128, 64, Layout::NZ, 512, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 6) {
        runTMOV_nz2nz<bfloat16_t, half, half, 128, 64, 128, 128, 64, 128, Layout::NZ, 512, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 7) {
        runSplitNTMOV_nz2nz<float, float, float, 32, 16, 32, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 8) {
        runSplitNTMOV_nz2nz<float, half, half, 128, 16, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 9) {
        runSplitMTMOV_nz2nz<float, half, half, 32, 16, 32, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 10) {
        runSplitMTMOV_nz2nz<float, float, float, 128, 128, 64, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    }
}

template void launchTMOVL0c2UBNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2NZ<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBVectorQuantNz(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1){
        runVectorQuantTMOV_nz2nz<int8_t, int8_t, int8_t, uint64_t, 32, 32, 128, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                 reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2){
        runVectorQuantTMOV_nz2nz<half, int8_t, int8_t, uint64_t, 128, 64, 128, 128, 64, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                 reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3){
        runVectorQuantTMOV_nz2nz<int8_t, float, float, uint64_t, 64, 32, 128, 64, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                 reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4){
        runVectorQuantTMOV_nz2nz<half, float, float, uint64_t, 64, 32, 64, 64, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0),
                                 reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(src2));
    }
}

template void launchTMOVL0c2UBVectorQuantNz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBVectorQuantNz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBVectorQuantNz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBVectorQuantNz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBSCQuantNz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream){
    if constexpr (tilingKey == 1){
         runScalarQuantTMOV_nz2nz<half, float, float, 128, 32, 64, 128, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0),
                                 reinterpret_cast<float *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        runScalarQuantTMOV_nz2nz<half, int8_t, int8_t, 32, 128, 64, 32, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                 reinterpret_cast<int8_t *>(src1), 4);
    } else if constexpr (tilingKey == 3) {
        runScalarQuantTMOV_nz2nz<int8_t, int8_t, int8_t, 32, 32, 128, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                 reinterpret_cast<int8_t *>(src1), 5);
    } else if constexpr (tilingKey == 4) {
        runScalarQuantTMOV_nz2nz<int8_t, float, float, 32, 32, 64, 32, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                 reinterpret_cast<float *>(src1), 7);
    }
}

template void launchTMOVL0c2UBSCQuantNz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuantNz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuantNz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuantNz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

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
        runScalarQuantTMOV<int8_t, float, float, 112, 96, 64, 112, 96, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 5);
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

template <typename outType, typename aType, typename bType, int M, int K, int N, int validM, int validK, int validN,
    int row, int col, int subBlockId, int sfractalSize = 512>
__global__ __aicore__ void runTMOV_nz2dn(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1)
{
    using GlobalDataOut = GlobalTensor<outType, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, 1, validM>, Layout::DN>;
    GlobalDataOut dstGlobal(out);

    runMATMUL<aType, bType, M, K, N, validM, validK, validN>(src0, src1);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<Location::Vec, outType, row, col, BLayout::ColMajor, validM, validN, SLayout::NoneBox, sfractalSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if (subBlockId == 0) {
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

    if (idx == subBlockId) {
        TSTORE(dstGlobal, dstTileData);
    }

#endif
    out = dstGlobal.data();
}

template <typename outType, typename aType, typename bType, typename fbType, int M, int K, int N,
    int validM, int validK, int validN>
__global__ __aicore__ void runVectorQuantTMOV_nz2dn(__gm__ outType *out, __gm__ aType *src0, __gm__ bType *src1, __gm__ fbType *src2)
{
    using GlobalDataOut =
        GlobalTensor<outType, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, 1, M>, Layout::DN>;
    GlobalDataOut dstGlobal(out);

    runMATMULFB<aType, bType, fbType, M, K, N, validM, validK, validN>(src0, src1, src2);

    using AccTile = TileAcc<CType<aType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using TileMatFbData = Tile<Location::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    TileMatFbData fbMatTile;
    TASSIGN(fbMatTile, 0x20000);
    using FbTile = Tile<Location::Scaling, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    FbTile fbTile;
    TASSIGN(fbTile, 0x0);

    using DstTileData = Tile<Location::Vec, outType, M, N, BLayout::ColMajor, validM, validN>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

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

template <int32_t tilingKey>
void launchTMOVL0c2UBNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runTMOV_nz2dn<float, float, float, 64, 128, 32, 64, 128, 32, 64, 32, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 2) {
        runTMOV_nz2dn<half, half, half, 128, 32, 64, 128, 32, 64, 128, 64, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 3) {
        runTMOV_nz2dn<bfloat16_t, half, half, 48, 32, 32, 48, 31, 31, 64, 32, 1><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        runTMOV_nz2dn<float, half, half, 64, 128, 128, 64, 128, 128, 64, 128, 0, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}

template void launchTMOVL0c2UBNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBVectorQuantDn(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1){
        runVectorQuantTMOV_nz2dn<int8_t, int8_t, int8_t, uint64_t, 128, 128, 64, 128, 128, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                 reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2){
        runVectorQuantTMOV_nz2dn<half, int8_t, int8_t, uint64_t, 32, 32, 128, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                 reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3){
        runVectorQuantTMOV_nz2dn<int8_t, half, half, uint64_t, 128, 64, 128, 128, 64, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                 reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4){
        runVectorQuantTMOV_nz2dn<half, half, half, uint64_t, 32, 32, 64, 32, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                 reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2));
    }
}

template void launchTMOVL0c2UBVectorQuantDn<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBVectorQuantDn<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBVectorQuantDn<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void launchTMOVL0c2UBVectorQuantDn<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMOVL0c2UBSCQuantDn(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        runScalarQuantTMOVNz2Dn<half, float, float, 128, 32, 64, 128, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        runScalarQuantTMOVNz2Dn<int8_t, float, float, 128, 96, 64, 128, 96, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 5);
    } else if constexpr (tilingKey == 3) {
        runScalarQuantTMOVNz2Dn<half, int8_t, int8_t, 32, 128, 64, 32, 128, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 3);
    } else if constexpr (tilingKey == 4) {
        runScalarQuantTMOVNz2Dn<int8_t, int8_t, int8_t, 32, 32, 32, 32, 32, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 1);
    }
}

template void launchTMOVL0c2UBSCQuantDn<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuantDn<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuantDn<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOVL0c2UBSCQuantDn<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);