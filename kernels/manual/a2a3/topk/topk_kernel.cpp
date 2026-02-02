/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <int Cols>
PTO_INTERNAL int32_t FillMrgArray(int32_t *mrgArray, int blockLen)
{
    int32_t arrayCount = 0;
    int32_t tmpInner = Cols;
    for (int32_t i = blockLen; i >= 64; i /= 4) {
        int32_t count;
        for (count = 0; count < tmpInner / i; count++) {
            mrgArray[arrayCount++] = i;
        }
        tmpInner -= count * i;
    }
    return arrayCount;
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData, typename T, int Cols, int topk>
PTO_INTERNAL void SortTailBlock(DstTileData &dstTile, SrcTileData &srcTile, int blockLen, uint64_t tmpAddr)
{
    TmpTileData tmp1Tile(1, Cols);
    TASSIGN(tmp1Tile, tmpAddr);

    int32_t mrgArray[15] = {0};
    int32_t arrayCount = FillMrgArray<Cols>(mrgArray, blockLen);
    uint16_t mrgSortedLen = 0;
    MrgSortExecutedNumList executedNumList;
    for (int32_t i = 0; i < arrayCount - 1; ++i) {
        mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
        uint64_t tmpMrgSortedLen = mrgSortedLen;
        uint64_t tmpMrgArray = mrgArray[i + 1];
        if (tmpMrgSortedLen > topk) {
            tmpMrgSortedLen = topk;
        }
        if (tmpMrgArray > topk) {
            tmpMrgArray = topk;
        }

        SrcTileData src0Tile(1, tmpMrgSortedLen);
        SrcTileData src1Tile(1, tmpMrgArray);
        SrcTileData curDstTile(1, tmpMrgSortedLen + tmpMrgArray);
        TASSIGN(src0Tile, (uint64_t)srcTile.data());
        TASSIGN(src1Tile, (uint64_t)srcTile.data() + mrgSortedLen * sizeof(T));
        TASSIGN(curDstTile, (uint64_t)srcTile.data());
        TMRGSORT<DstTileData, TmpTileData, SrcTileData, SrcTileData, 0>(
            curDstTile, executedNumList, tmp1Tile, src0Tile, src1Tile);
        pipe_barrier(PIPE_V);
    }
}

template <typename DstTileData, typename SrcTileData, int kTRows_, int kTCols_, int valid_row, int valid_col, int dtopk>
PTO_INTERNAL void MrgsortSingleRow(DstTileData &dstTile, SrcTileData &srcTile, uint64_t tmpAddr)
{
    using T = typename SrcTileData::DType;
    constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
    uint32_t blockLen = 64 * TYPE_COEF;

    // Merge sort data for every 4 blockLen lengths.
    for (; blockLen * 4 <= valid_col; blockLen *= 4) {
        uint16_t cols = valid_col / (blockLen * 4) * (blockLen * 4);
        SrcTileData srcSortedTile(1, cols);
        SrcTileData tmpSortedTile(1, cols);
        TASSIGN(srcSortedTile, (uint64_t)srcTile.data());
        TASSIGN(tmpSortedTile, tmpAddr);
        TMRGSORT<SrcTileData, SrcTileData>(tmpSortedTile, srcSortedTile, blockLen);
        pipe_barrier(PIPE_V);
        TMOV(srcSortedTile, tmpSortedTile);
        pipe_barrier(PIPE_V);
    }

    // sort tail block
    if (blockLen < valid_col) {
        SortTailBlock<DstTileData, SrcTileData, SrcTileData, T, valid_col, dtopk>(srcTile, srcTile, blockLen, tmpAddr);
        SrcTileData tmpMovTile(1, dtopk);
        TASSIGN(tmpMovTile, (uint64_t)srcTile.data());
        pipe_barrier(PIPE_V);
        TMOV(dstTile, tmpMovTile);
    } else {
        SrcTileData tmpMovTile(1, dtopk);
        TASSIGN(tmpMovTile, (uint64_t)srcTile.data());
        pipe_barrier(PIPE_V);
        TMOV(dstTile, tmpMovTile);
    }
}

template <typename T, typename DstTileData, typename SrcTileData, typename RowTile, int kTRows_, int kTCols_,
    int validRow, int validCol, int topk>
PTO_INTERNAL void MrgsortSingleTile(DstTileData &dstTile, SrcTileData &srcTile, uint64_t tmpAddr)
{
    for (int i = 0; i < validRow; i++) {
        RowTile rowSrcTile(1, validCol);
        TASSIGN(rowSrcTile, (uint64_t)srcTile.data() + i * kTCols_ * sizeof(T));
        RowTile rowDstTile(1, validCol);
        TASSIGN(rowDstTile, (uint64_t)dstTile.data() + i * DstTileData::Cols * sizeof(T));
        MrgsortSingleRow<RowTile, RowTile, 1, kTCols_, 1, validCol, topk>(rowDstTile, rowSrcTile, tmpAddr);
    }
}

template <typename T, typename DstTileData, typename SrcTileData, typename IdxTileData, typename RowTile, int kTRows_, int kTCols_,
    int validRow, int validCol>
PTO_INTERNAL void SortEachGroup(DstTileData &dst, SrcTileData &src, IdxTileData &inIdx)
{
    using indexT = uint32_t;
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    for (size_t i = 0; i < validRow; ++i) {
        RowTile dstRowTile(1, validCol * 2 * TYPE_COEF);
        RowTile srcRowTile(1, validCol);
        RowTile tmpTile(1, validCol);
        TASSIGN(dstRowTile, (uint64_t)dst.data() + i * DstTileData::Cols * sizeof(T));
        TASSIGN(srcRowTile, (uint64_t)src.data() + i * SrcTileData::Cols * sizeof(T));
        TASSIGN(tmpTile, (uint64_t)inIdx.data() + kTCols_ * sizeof(indexT));
        TSORT32(dstRowTile, srcRowTile, inIdx, tmpTile);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename DstTileData, typename SrcTileData, typename RowTile, bool isIndex>
PTO_INTERNAL void ExtractDataOrIndex(DstTileData &dstTile, SrcTileData &srcTile)
{
    for (size_t i = 0; i < srcTile.GetValidRow(); ++i) {
        RowTile rowTile(1, srcTile.GetValidCol());
        TASSIGN(rowTile, (uint64_t)srcTile.data() + i * SrcTileData::Cols * sizeof(T));
        if constexpr (isIndex == false) {
            RowTile rowDTile(1, dstTile.GetValidCol());
            TASSIGN(rowDTile, (uint64_t)dstTile.data() + i * DstTileData::Cols * sizeof(T));
            if constexpr (std::is_same_v<T, half>) {
                TGATHER<RowTile, RowTile, MaskPattern::P0001>(rowDTile, rowTile);
            } else {
                TGATHER<RowTile, RowTile, MaskPattern::P0101>(rowDTile, rowTile);
            }
        } else {
            using indexT = uint32_t;
            using CopySrcTileData = Tile<TileType::Vec, indexT, 1, DstTileData::Cols * 2, BLayout::RowMajor, -1, -1>;
            CopySrcTileData copyTile(1, dstTile.GetValidCol() * 2);
            TASSIGN(copyTile, (uint64_t)rowTile.data());

            using IndexRowTileData = Tile<TileType::Vec, indexT, 1, DstTileData::Cols, BLayout::RowMajor, -1, -1>;
            IndexRowTileData rowITile(1, dstTile.GetValidCol());
            TASSIGN(rowITile, (uint64_t)dstTile.data() + i * DstTileData::Cols * sizeof(indexT));

            TGATHER<IndexRowTileData, CopySrcTileData, MaskPattern::P1010>(rowITile, copyTile);
        }
    }
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int blockDim>
AICORE inline void runTOPK(__gm__ T *origOut, __gm__ uint32_t *origIndex, __gm__ T *origSrc, __gm__ uint32_t *origInIdx)
{
    using indexT = uint32_t;

    constexpr int totalRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3 / blockDim;
    constexpr int validCol = gShape4;
    static_assert(totalRow % blockDim == 0, "expect totalRow % blockDim == 0");
    __gm__ T *src = origSrc + get_block_idx() * validRow * gWholeShape4;
    __gm__ T *out = origOut + get_block_idx() * validRow * topk;
    __gm__ uint32_t *index = origIndex + get_block_idx() * validRow * topk;
    __gm__ uint32_t *inIdx = origInIdx;
    constexpr int Rows = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 / blockDim;
    constexpr int Cols = gWholeShape4;

    constexpr int tileNumTwo = 2;
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    constexpr int singleLoopRow = 2;
    constexpr int dstCols = validCol * 2 * TYPE_COEF;
    using DstTileData = Tile<TileType::Vec, T, singleLoopRow, dstCols, BLayout::RowMajor, singleLoopRow, dstCols>;
    DstTileData sort32DstTile[tileNumTwo];

    using SrcTileData = Tile<TileType::Vec, T, singleLoopRow, validCol, BLayout::RowMajor, singleLoopRow, validCol>;
    SrcTileData srcTile[tileNumTwo];

    using SingleRowTileData = Tile<TileType::Vec, T, 1, dstCols, BLayout::RowMajor, -1, -1>;

    using IndexTileData = Tile<TileType::Vec, indexT, 1, validCol, BLayout::RowMajor, 1, validCol>;
    IndexTileData indexTile;
    using IndexShapeDim5 = Shape<1, 1, 1, 1, validCol>;
    using IndexStridDim5 = Stride<validCol, validCol, validCol, validCol, 1>;
    using IndexGlobalData = GlobalTensor<indexT, IndexShapeDim5, IndexStridDim5>;
    IndexGlobalData idxGlobal(inIdx);

    constexpr uint32_t loopNum = validRow / singleLoopRow;
    static_assert(validRow % (singleLoopRow * 2) == 0, "expect validRow % (singleLoopRow * 2) == 0.");

    DstTileData mrgDstTile[tileNumTwo];

    using DstDataTileData = Tile<TileType::Vec, T, singleLoopRow, topk, BLayout::RowMajor, singleLoopRow, topk>;
    DstDataTileData dTile[tileNumTwo];

    using DstIndexTileData = Tile<TileType::Vec, indexT, singleLoopRow, topk, BLayout::RowMajor, singleLoopRow, topk>;
    DstIndexTileData iTile[tileNumTwo];

    using DynShapeDim5 = Shape<1, 1, 1, singleLoopRow, validCol>;
    using DynStridDim5 = Stride<singleLoopRow * Cols, singleLoopRow * Cols, singleLoopRow * Cols, Cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, singleLoopRow, topk>;
    using DstStridDim5 = Stride<singleLoopRow * topk, singleLoopRow * topk, singleLoopRow * topk, topk, 1>;
    using DstDataGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5>;
    using DstIdxGlobalData = GlobalTensor<indexT, DstShapeDim5, DstStridDim5>;

    constexpr uint32_t sort32DstSize = singleLoopRow * dstCols * sizeof(T) * 2;
    TASSIGN(sort32DstTile[0], 0x0);
    TASSIGN(sort32DstTile[1], 0x0 + sort32DstSize / 2);
    TASSIGN(mrgDstTile[0], 0x0 + sort32DstSize);
    TASSIGN(mrgDstTile[1], 0x0 + sort32DstSize + sort32DstSize / 2);

    TASSIGN(indexTile, 0x0 + sort32DstSize * 2);
    uint64_t tmpAddr = 0x0 + sort32DstSize * 2 + validCol * sizeof(indexT);
    uint64_t nextTmpAddr = 0x0 + sort32DstSize * 2 + validCol * sizeof(indexT) * 3;
    uint64_t curAddr = sort32DstSize * 2 + validCol * sizeof(indexT) * 5;
    TASSIGN(dTile[0], curAddr);
    TASSIGN(dTile[1], curAddr + singleLoopRow * topk * sizeof(T));
    TASSIGN(iTile[0], curAddr + singleLoopRow * topk * sizeof(T) * 2);
    TASSIGN(iTile[1], curAddr + singleLoopRow * topk * sizeof(T) * 2 + singleLoopRow * topk * sizeof(indexT));
    constexpr uint32_t srcSize = singleLoopRow * validCol * sizeof(T) * 2;
    TASSIGN(srcTile[0], 0x0 + sort32DstSize * 3 + validCol * sizeof(indexT) * 5);
    TASSIGN(srcTile[1], 0x0 + sort32DstSize * 3 + validCol * sizeof(indexT) * 5 + srcSize / 2);

    static_assert(sort32DstSize * 3 + validCol * sizeof(indexT) * 5 + srcSize < 192 * 1024, "memory is exhausted.");
    TLOAD(indexTile, idxGlobal);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // reverse
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1); // reverse

    for (uint32_t i = 0; i < loopNum; i += 2) {
        GlobalData src0Global(src + i * singleLoopRow * Cols); // ND2ND
        GlobalData src1Global(src + i * singleLoopRow * Cols + singleLoopRow * Cols);
        DstDataGlobalData dst0DataGlobal(out + i * singleLoopRow * topk);
        DstDataGlobalData dst1DataGlobal(out + i * singleLoopRow * topk + singleLoopRow * topk);
        DstIdxGlobalData dst0IdxGlobal(index + i * singleLoopRow * topk);
        DstIdxGlobalData dst1IdxGlobal(index + i * singleLoopRow * topk + singleLoopRow * topk);

        constexpr int cur = 0;
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // reverse 0
        TLOAD(srcTile[cur], src0Global);

        set_flag(PIPE_MTE2, PIPE_V, (event_t)cur);
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)cur);

        SortEachGroup<T, DstTileData, SrcTileData, IndexTileData, SingleRowTileData, singleLoopRow, validCol,
            singleLoopRow, validCol>(sort32DstTile[cur], srcTile[cur], indexTile);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // reverse 0

        pipe_barrier(PIPE_V);
        MrgsortSingleTile<T, DstTileData, DstTileData, SingleRowTileData, singleLoopRow, dstCols, singleLoopRow,
            dstCols, topk * 2 * TYPE_COEF>(mrgDstTile[cur], sort32DstTile[cur], tmpAddr);

        pipe_barrier(PIPE_V);
        ExtractDataOrIndex<T, DstDataTileData, DstTileData, SingleRowTileData, 0>(dTile[cur], mrgDstTile[cur]);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)cur);

        pipe_barrier(PIPE_V);
        ExtractDataOrIndex<T, DstIndexTileData, DstTileData, SingleRowTileData, 1>(iTile[cur], mrgDstTile[cur]);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)(cur + 2));

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)cur);
        TSTORE(dst0DataGlobal, dTile[cur]);

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)(cur + 2));
        TSTORE(dst0IdxGlobal, iTile[cur]);

        ////////////////////////////////////////
        constexpr int next = 1;
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1); // reverse 1
        TLOAD(srcTile[next], src1Global);

        set_flag(PIPE_MTE2, PIPE_V, (event_t)next);
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)next);

        SortEachGroup<T, DstTileData, SrcTileData, IndexTileData, SingleRowTileData, singleLoopRow, validCol,
            singleLoopRow, validCol>(sort32DstTile[next], srcTile[next], indexTile);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1); // reverse 1

        pipe_barrier(PIPE_V);
        MrgsortSingleTile<T, DstTileData, DstTileData, SingleRowTileData, singleLoopRow, dstCols, singleLoopRow,
            dstCols, topk * 2 * TYPE_COEF>(mrgDstTile[next], sort32DstTile[next], nextTmpAddr);

        pipe_barrier(PIPE_V);
        ExtractDataOrIndex<T, DstDataTileData, DstTileData, SingleRowTileData, 0>(dTile[next], mrgDstTile[next]);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)next);

        pipe_barrier(PIPE_V);
        ExtractDataOrIndex<T, DstIndexTileData, DstTileData, SingleRowTileData, 1>(iTile[next], mrgDstTile[next]);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)(next + 2));

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)next);
        TSTORE(dst1DataGlobal, dTile[next]);

        wait_flag(PIPE_V, PIPE_MTE3, (event_t)(next + 2));
        TSTORE(dst1IdxGlobal, iTile[next]);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // reverse
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1); // reverse
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk, int blockDim>
__global__ AICORE void Topk(__gm__ uint8_t *out, __gm__ uint8_t *index, __gm__ uint8_t *src, __gm__ uint8_t *inIdx)
{
    using indexT = uint32_t;
    if constexpr (std::is_same_v<T, uint16_t>) {
        runTOPK<half, gShape0, gShape1, gShape2, gShape3, gShape4,
            gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4,
            topk, blockDim>(reinterpret_cast<__gm__ half *>(out),
            reinterpret_cast<__gm__ indexT *>(index), reinterpret_cast<__gm__ half *>(src), reinterpret_cast<__gm__ indexT *>(inIdx));
    } else {
        runTOPK<float, gShape0, gShape1, gShape2, gShape3, gShape4,
            gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4,
            topk, blockDim>(reinterpret_cast<__gm__ float *>(out),
            reinterpret_cast<__gm__ indexT *>(index), reinterpret_cast<__gm__ float *>(src), reinterpret_cast<__gm__ indexT *>(inIdx));
    }
}

template <typename T>
void launchTopk(uint8_t *out, uint8_t *index, uint8_t *src, uint8_t *inIdx, void *stream)
{
    constexpr int blockDim = 48;
    constexpr int gShape3 = 4800;
    constexpr int gShape4 = 1024;
    constexpr int gWholeShape3 = 4800;
    constexpr int gWholeShape4 = 1280;
    constexpr int topk = 1000;
    Topk<T, 1, 1, 1, gShape3, gShape4, 1, 1, 1, gWholeShape3, gWholeShape4,
        topk, blockDim><<<blockDim, nullptr, stream>>>(out, index, src, inIdx);
}

template void launchTopk<float>(uint8_t *out, uint8_t *index, uint8_t *src, uint8_t *inIdx, void *stream);