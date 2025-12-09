/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMRGSORT_HPP
#define TMRGSORT_HPP

#include <pto/common/constants.hpp>
#define TRUE 1
#define FALSE 0
#define STRUCTSIZE 8
#define UBSIZE 262144  // 256 * 1024 B
#define ELEMSIZE 4

namespace pto
{
    struct MrgSortExecutedNumList {
        uint16_t mrgSortList0;
        uint16_t mrgSortList1;
        uint16_t mrgSortList2;
        uint16_t mrgSortList3;
    };

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              typename Src2TileData, bool exhausted>
    PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1,
                                        Src2TileData &src2) {
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              bool exhausted>
    PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1) {
    }

    // blockLen大小包含值+索引，比如32个值+索引：blockLen=64
    template <typename DstTileData, typename SrcTileData>
    PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t blockLen) {
    }

    template <typename Src0TileData, typename Src1TileData, typename Src2TileData, typename Src3TileData>
    PTO_INTERNAL constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols + Src3TileData::Cols;
    }

    template <typename Src0TileData, typename Src1TileData, typename Src2TileData>
    PTO_INTERNAL constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols;
    }

    template <typename Src0TileData, typename Src1TileData>
    PTO_INTERNAL constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols;
    }
}
#endif