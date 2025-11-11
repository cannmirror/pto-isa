#include <common/tile_tensor_impl.hpp>
#include <common/pto_tile.hpp>
#include <common/constants.hpp>
#include <davinci/v220/TMrgSort.hpp>
#include <davinci/v220/TLoad.hpp>
#include <davinci/v220/TStore.hpp>
#include <davinci/v220/TAssign.hpp>
#include <iostream>

using namespace std;
using namespace pto;

#define EXHAUSTED 1

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3, int TOPK, int LISTNUM>
__aicore__ void runTMrgsort( __gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1,
                                        __gm__ T __in__ *src2, __gm__ T __in__ *src3) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using TmpGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<Location::Vec, T, 1, kTCols_*LISTNUM, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(1, kTCols_);
    TileData src1Tile(1, kTCols_src1);
    TileData src2Tile(1, kTCols_src2);
    TileData src3Tile(1, kTCols_src3);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols_*LISTNUM);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(src2Tile, 0x8000);
    TASSIGN(src3Tile, 0xC000);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    for (unsigned row = 0; row < kTRows_; row ++) {
        int dataOffset = 0; // 0 * 128
        GlobalData src0Global(src0 + dataOffset);
        GlobalData src1Global(src1 + dataOffset);
        GlobalData src2Global(src2 + dataOffset);
        GlobalData src3Global(src3 + dataOffset);
        DstGlobalData dstGlobal(out + dataOffset*LISTNUM);

        // TSTORE(dstGlobal, src0Tile)
        MrgSortExecutedNumList executedNumList;
        // 4
        if constexpr (LISTNUM == 4) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            TLOAD(src2Tile, src2Global);
            TLOAD(src3Tile, src3Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, 0>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
        }
        // 3
        if constexpr (LISTNUM == 3) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            TLOAD(src2Tile, src2Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, 0>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
        }
        // 2
        if constexpr (LISTNUM == 2) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, 0>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, dstTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3, int TOPK, int LISTNUM>
__aicore__ void runTMrgsortExhausted( __gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1,
                                        __gm__ T __in__ *src2, __gm__ T __in__ *src3) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using TmpGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<Location::Vec, T, 1, kTCols_*LISTNUM, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(1, kTCols_);
    TileData src1Tile(1, kTCols_src1);
    TileData src2Tile(1, kTCols_src2);
    TileData src3Tile(1, kTCols_src3);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols_*LISTNUM);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(src2Tile, 0x8000);
    TASSIGN(src3Tile, 0xC000);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    int offset = 0; // (block_idx / 4) * （64 * 16) + (block_idx % 4) * 16;
    for (unsigned row = 0; row < kTRows_; row ++) {
        // ############################ round1
        int dataOffset = row*kGCols_;
        GlobalData src00Global(src0 + dataOffset);
        GlobalData src10Global(src1 + dataOffset);
        GlobalData src20Global(src2 + dataOffset);
        GlobalData src30Global(src3 + dataOffset);
        DstGlobalData dst0Global(out + dataOffset*LISTNUM);

        MrgSortExecutedNumList executedNumList;
        // 4
        if constexpr (LISTNUM == 4) {
            TLOAD(src0Tile, src00Global);
            TLOAD(src1Tile, src10Global);
            TLOAD(src2Tile, src20Global);
            TLOAD(src3Tile, src30Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
        }
        // 3
        if constexpr (LISTNUM == 3) {
            TLOAD(src0Tile, src00Global);
            TLOAD(src1Tile, src10Global);
            TLOAD(src2Tile, src20Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
        }
        // 2
        if constexpr (LISTNUM == 2) {
            TLOAD(src0Tile, src00Global);
            TLOAD(src1Tile, src10Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dst0Global, dstTile);
        //pipe_barrier(PIPE_V);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        // ##################### round2
        int curTotal = kTCols_ * LISTNUM;
        int offset0 = kTCols_;
        int offset1 = kTCols_;
        int offset2 = kTCols_;
        int offset3 = kTCols_;
        if (EXHAUSTED) {
            int numElem = 8 / sizeof(T); // 2 for float; 4 for half;
            curTotal = executedNumList.mrgSortList0 + executedNumList.mrgSortList1 + 
              executedNumList.mrgSortList2 + executedNumList.mrgSortList3;
            curTotal = curTotal * numElem;
            offset0 = executedNumList.mrgSortList0 * numElem;
            offset1 = executedNumList.mrgSortList1 * numElem;
            offset2 = executedNumList.mrgSortList2 * numElem;
            offset3 = executedNumList.mrgSortList3 * numElem;
        }
        GlobalData src0Global(src0 + dataOffset + offset0);
        GlobalData src1Global(src1 + dataOffset + offset1);
        GlobalData src2Global(src2 + dataOffset + offset2);
        GlobalData src3Global(src3 + dataOffset + offset3);
        DstGlobalData dstGlobal(out + dataOffset*LISTNUM + curTotal);


        // 4
        if constexpr (LISTNUM == 4) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            TLOAD(src2Tile, src2Global);
            TLOAD(src3Tile, src3Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
        }
        // 3
        if constexpr (LISTNUM == 3) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            TLOAD(src2Tile, src2Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
        }
        // 2
        if constexpr (LISTNUM == 2) {
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, TileData, TileData, EXHAUSTED>
                (dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, dstTile);
        //pipe_barrier(PIPE_V);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ inline void runTMrgsort_single(__gm__ T __out__ *out, __gm__ T __in__ *src0) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(kGRows_, kGCols_);
    DstTileData dstTile(kGRows_, kGCols_);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0xF000);

    int offset = 0;
    GlobalData src0Global(src0 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint32_t blockLen = 64;
    TMRGSORT<DstTileData, TileData>(dstTile, src0Tile, blockLen);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk, int totalNum>
__aicore__ inline void runTMrgsort_topk(__gm__ T __out__ *out, __gm__ T __in__ *src) {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<Location::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, kGRows_, topk>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, topk, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using TmpDynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using TmpDynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using TmpGlobalData = GlobalTensor<T, TmpDynShapeDim5, TmpDynStridDim5>;
    using DstTileData = Tile<Location::Vec, T, kTRows_, topk, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<Location::Vec, T, 1, kGCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, topk);
    TmpTileData tmpTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x4000);
    TASSIGN(tmpTile, 0x8000);

    GlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    uint32_t blockLen = 64;

    // 没4个合并，计算整块
    for(; blockLen * 4 <= totalNum; blockLen *= 4) {
        TLOAD(srcTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMRGSORT<TmpTileData, TileData>(tmpTile, srcTile, blockLen);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(srcGlobal, tmpTile);
        pipe_barrier(PIPE_ALL); 
    }

    // 合并尾块
    if (blockLen < totalNum) {
        TmpTileData tmp1Tile(kTRows_, kTCols_);
        TASSIGN(tmp1Tile, 0xC000);
        int32_t arrayCount = 0;
        int32_t mrgArray[15] = {0};
        int32_t tmpInner = totalNum;
        for (int32_t i = blockLen; i >= 64; i /= 4) {
            int32_t count;
            for (count = 0; count < tmpInner / i; count++) {
                mrgArray[arrayCount++] = i;
            }
            tmpInner -= count * i;
        }
        uint16_t mrgSortedLen = 0;
        GlobalData srcGlobal(src);
        MrgSortExecutedNumList executedNumList;
        for (int32_t i = 0; i < arrayCount - 1; ++i) {
            TLOAD(srcTile, srcGlobal);
            using Src0TileData = Tile<Location::Vec, T, kTRows_, totalNum, BLayout::RowMajor, -1, -1>;
            using Src1TileData = Tile<Location::Vec, T, kTRows_, totalNum, BLayout::RowMajor, -1, -1>;
            mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
            uint64_t tmpMrgSortedLen = mrgSortedLen;
            uint64_t tmpMrgArray = mrgArray[i + 1];
            if (tmpMrgSortedLen > topk) {
                tmpMrgSortedLen = topk;
            }
            if (tmpMrgArray > topk) {
                tmpMrgArray = topk;
            }
            Src0TileData src0Tile(kTRows_, tmpMrgSortedLen);
            Src1TileData src1Tile(kTRows_, tmpMrgArray);
            TASSIGN(src0Tile, 0x10000);
            TASSIGN(src1Tile, 0x20000);
            GlobalData src0Global(src);
            GlobalData src1Global(src + mrgSortedLen);
            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMRGSORT<DstTileData, TmpTileData, Src0TileData, Src1TileData, 0>(dstTile, executedNumList, tmp1Tile,
                                        src0Tile, src1Tile);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);  
            TSTORE(srcGlobal, dstTile);
            pipe_barrier(PIPE_ALL);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, dstTile);

            pipe_barrier(PIPE_ALL);
        }
    } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstGlobal, tmpTile);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ __aicore__ void launchTMRGSORT_multi_1(__gm__ float *out, __gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3)
{   constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 128;
    constexpr uint32_t SRC0COL = 128;
    constexpr uint32_t SRC1COL = 128;
    constexpr uint32_t SRC2COL = 128;
    constexpr uint32_t SRC3COL = 128;
    constexpr uint32_t TOPK = 512;
    constexpr uint32_t LISTMUM = 4;
    runTMrgsort<float, ROW, COL * LISTMUM, ROW, SRC0COL, SRC1COL, SRC2COL, SRC3COL, TOPK, LISTMUM>(out, src0, src1, src2, src3);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_multi_2(__gm__ float *out, __gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3)
{   constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 128;
    constexpr uint32_t SRC0COL = 128;
    constexpr uint32_t SRC1COL = 128;
    constexpr uint32_t SRC2COL = 128;
    constexpr uint32_t SRC3COL = 128;
    constexpr uint32_t TOPK = 512;
    constexpr uint32_t LISTMUM = 4;
    runTMrgsort<half, ROW, COL * LISTMUM, ROW, SRC0COL, SRC1COL, SRC2COL, SRC3COL, TOPK, LISTMUM>(reinterpret_cast<__gm__ half *>(out),
                                                                                        reinterpret_cast<__gm__ half *>(src0), 
                                                                                        reinterpret_cast<__gm__ half *>(src1), 
                                                                                        reinterpret_cast<__gm__ half *>(src2), 
                                                                                        reinterpret_cast<__gm__ half *>(src3));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_multi_3(__gm__ float *out, __gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3)
{   constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 128;
    constexpr uint32_t SRC0COL = 128;
    constexpr uint32_t SRC1COL = 128;
    constexpr uint32_t SRC2COL = 128;
    constexpr uint32_t SRC3COL = 64;
    constexpr uint32_t TOPK = 448;
    constexpr uint32_t LISTMUM = 4;
    runTMrgsort<float, ROW, COL * LISTMUM, ROW, SRC0COL, SRC1COL, SRC2COL, SRC3COL, TOPK, LISTMUM>(out, src0, src1, src2, src3);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_multi_4(__gm__ float *out, __gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3)
{   constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 128;
    constexpr uint32_t SRC0COL = 128;
    constexpr uint32_t SRC1COL = 128;
    constexpr uint32_t SRC2COL = 64;
    constexpr uint32_t SRC3COL = 0;
    constexpr uint32_t TOPK = 128;
    constexpr uint32_t LISTMUM = 3;
    runTMrgsort<float, ROW, COL * LISTMUM, ROW, SRC0COL, SRC1COL, SRC2COL, SRC3COL, TOPK, LISTMUM>(out, src0, src1, src2, src3);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_multi_exhausted_1(__gm__ float *out, __gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3)
{   constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 64;
    constexpr uint32_t SRC0COL = 64;
    constexpr uint32_t SRC1COL = 64;
    constexpr uint32_t SRC2COL = 0;
    constexpr uint32_t SRC3COL = 0;
    constexpr uint32_t TOPK = 128;
    constexpr uint32_t LISTMUM = 2;
    runTMrgsortExhausted<float, ROW, COL * LISTMUM, ROW, SRC0COL, SRC1COL, SRC2COL, SRC3COL, TOPK, LISTMUM>(out, src0, src1, src2, src3);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_multi_exhausted_2(__gm__ float *out, __gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3)
{   constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 256;
    constexpr uint32_t SRC0COL = 256;
    constexpr uint32_t SRC1COL = 256;
    constexpr uint32_t SRC2COL = 256;
    constexpr uint32_t SRC3COL = 0;
    constexpr uint32_t TOPK = 768;
    constexpr uint32_t LISTMUM = 3;
    runTMrgsortExhausted<half, ROW, COL * LISTMUM, ROW, SRC0COL, SRC1COL, SRC2COL, SRC3COL, TOPK, LISTMUM>(reinterpret_cast<__gm__ half *>(out),
                                                                                        reinterpret_cast<__gm__ half *>(src0), 
                                                                                        reinterpret_cast<__gm__ half *>(src1), 
                                                                                        reinterpret_cast<__gm__ half *>(src2), 
                                                                                        reinterpret_cast<__gm__ half *>(src3));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_1(__gm__ float *out, __gm__ float *src0)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 256;
    runTMrgsort_single<float, ROW, COL, ROW, COL>(out, src0);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_2(__gm__ float *out, __gm__ float *src0)
{   
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 320;
    runTMrgsort_single<float, ROW, COL, ROW, COL>(out, src0);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_3(__gm__ float *out, __gm__ float *src0)
{   
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 512;
    runTMrgsort_single<float, ROW, COL, ROW, COL>(out, src0);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_4(__gm__ float *out, __gm__ float *src0)
{   
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 640;
    runTMrgsort_single<float, ROW, COL, ROW, COL>(out, src0);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_5(__gm__ float *out, __gm__ float *src0)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 256;
    runTMrgsort_single<half, ROW, COL, ROW, COL>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src0));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_6(__gm__ float *out, __gm__ float *src0)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 320;
    runTMrgsort_single<half, ROW, COL, ROW, COL>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src0));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_7(__gm__ float *out, __gm__ float *src0)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 512;
    runTMrgsort_single<half, ROW, COL, ROW, COL>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src0));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_single_8(__gm__ float *out, __gm__ float *src0)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 640;
    runTMrgsort_single<half, ROW, COL, ROW, COL>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src0));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_topk_1(__gm__ float *out, __gm__ float *src)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 2048;
    constexpr uint32_t TOTAL_NUM = ROW * COL;
    constexpr uint32_t TOPK = 1024;
    runTMrgsort_topk<float, ROW, COL, ROW, COL, TOPK, TOTAL_NUM>(out, src);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_topk_2(__gm__ float *out, __gm__ float *src)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 2048;
    constexpr uint32_t TOTAL_NUM = ROW * COL;
    constexpr uint32_t TOPK = 2048;
    runTMrgsort_topk<float, ROW, COL, ROW, COL, TOPK, TOTAL_NUM>(out, src);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_topk_3(__gm__ float *out, __gm__ float *src)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 1280;
    constexpr uint32_t TOTAL_NUM = ROW * COL;
    constexpr uint32_t TOPK = 512;
    runTMrgsort_topk<float, ROW, COL, ROW, COL, TOPK, TOTAL_NUM>(out, src);
}

extern "C" __global__ __aicore__ void launchTMRGSORT_topk_4(__gm__ float *out, __gm__ float *src)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 2048;
    constexpr uint32_t TOTAL_NUM = ROW * COL;
    constexpr uint32_t TOPK = 1024;
    runTMrgsort_topk<half, ROW, COL, ROW, COL, TOPK, TOTAL_NUM>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_topk_5(__gm__ float *out, __gm__ float *src)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 2048;
    constexpr uint32_t TOTAL_NUM = ROW * COL;
    constexpr uint32_t TOPK = 2048;
    runTMrgsort_topk<half, ROW, COL, ROW, COL, TOPK, TOTAL_NUM>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ __aicore__ void launchTMRGSORT_topk_6(__gm__ float *out, __gm__ float *src)
{  
    constexpr uint32_t ROW = 1;
    constexpr uint32_t COL = 1280;
    constexpr uint32_t TOTAL_NUM = ROW * COL;
    constexpr uint32_t TOPK = 512;
    runTMrgsort_topk<half, ROW, COL, ROW, COL, TOPK, TOTAL_NUM>(reinterpret_cast<__gm__ half *>(out),
        reinterpret_cast<__gm__ half *>(src));
}

template <int32_t tilingKey>
void launchTMRGSORT_multi_demo(float *out, float *src0, float *src1, float *src2, float *src3, void* stream) {
    if constexpr(tilingKey == 1){
        launchTMRGSORT_multi_1<<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    } else if constexpr(tilingKey == 2){ 
        launchTMRGSORT_multi_2<<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    } else if constexpr(tilingKey == 3){ 
        launchTMRGSORT_multi_3<<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    } else if constexpr(tilingKey == 4){ 
        launchTMRGSORT_multi_4<<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    }
}

template <int32_t tilingKey>
void launchTMrgsort_demo_multi_exhausted(float *out, float *src0, float *src1, float *src2, float *src3, void* stream) {
    if constexpr(tilingKey == 1){
        launchTMRGSORT_multi_exhausted_1<<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    } else if constexpr(tilingKey == 2){ 
        launchTMRGSORT_multi_exhausted_2<<<1, nullptr, stream>>>(out, src0, src1, src2, src3);
    }
}

template <int32_t tilingKey>
void launchTMRGSORT_single_demo(float *out, float *src0, void* stream) 
{
    if constexpr(tilingKey == 1){
        launchTMRGSORT_single_1<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 2){ 
        launchTMRGSORT_single_2<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 3){ 
        launchTMRGSORT_single_3<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 4){ 
        launchTMRGSORT_single_4<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 5){ 
        launchTMRGSORT_single_5<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 6){ 
        launchTMRGSORT_single_6<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 7){ 
        launchTMRGSORT_single_7<<<1, nullptr, stream>>>(out, src0);
    } else if constexpr(tilingKey == 8){ 
        launchTMRGSORT_single_8<<<1, nullptr, stream>>>(out, src0);
    }
}

template <int32_t tilingKey>
void launchTMRGSORT_topk_demo(float *out, float *src, void* stream) 
{
    if constexpr(tilingKey == 1){
        launchTMRGSORT_topk_1<<<1, nullptr, stream>>>(out, src);
    } else if constexpr(tilingKey == 2){ 
        launchTMRGSORT_topk_2<<<1, nullptr, stream>>>(out, src);
    } else if constexpr(tilingKey == 3){ 
        launchTMRGSORT_topk_3<<<1, nullptr, stream>>>(out, src);
    } else if constexpr(tilingKey == 4){ 
        launchTMRGSORT_topk_4<<<1, nullptr, stream>>>(out, src);
    } else if constexpr(tilingKey == 5){ 
        launchTMRGSORT_topk_5<<<1, nullptr, stream>>>(out, src);
    } else if constexpr(tilingKey == 6){ 
        launchTMRGSORT_topk_6<<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTMRGSORT_multi_demo<1>(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template void launchTMRGSORT_multi_demo<2>(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template void launchTMRGSORT_multi_demo<3>(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template void launchTMRGSORT_multi_demo<4>(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template void launchTMrgsort_demo_multi_exhausted<1>(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template void launchTMrgsort_demo_multi_exhausted<2>(float *out, float *src0, float *src1, float *src2, float *src3, void* stream);
template void launchTMRGSORT_single_demo<1>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<2>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<3>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<4>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<5>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<6>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<7>(float *out, float *src0, void* stream);
template void launchTMRGSORT_single_demo<8>(float *out, float *src0, void* stream);
template void launchTMRGSORT_topk_demo<1>(float *out, float *src, void* stream);
template void launchTMRGSORT_topk_demo<2>(float *out, float *src, void* stream);
template void launchTMRGSORT_topk_demo<3>(float *out, float *src, void* stream);
template void launchTMRGSORT_topk_demo<4>(float *out, float *src, void* stream);
template void launchTMRGSORT_topk_demo<5>(float *out, float *src, void* stream);
template void launchTMRGSORT_topk_demo<6>(float *out, float *src, void* stream);