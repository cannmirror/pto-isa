/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef T_ROW_REDUCE_OPS_HPP
#define T_ROW_REDUCE_OPS_HPP
#include <common/utils.hpp>
#include <common/type.hpp>

#ifndef B16_REPEAT_MAX
#define B16_REPEAT_MAX 65535
#endif

namespace pto
{
  template <typename T, typename InstrOp>
  struct TRowReduceOp {
    __PTO_INSTR__ static void BinInstr(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t rptTimes,
      uint16_t dstRptStride, uint16_t src0RptStride, uint16_t src1RptStride) {
        InstrOp::BinInstrImpl(dst, src0, src1, rptTimes, dstRptStride, src0RptStride, src1RptStride);
    }

    __PTO_INSTR__ static void ReduceInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes,
      uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride) {
        InstrOp::ReduceInstrImpl(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }

    template <bool CntModeEn, int Cols, uint32_t DstStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    __PTO_INSTR__ static void ReduceInstrByMode(__ubuf__ T *dst, __ubuf__ T *src, unsigned rptTimes) {
      if constexpr (DstStride > B16_REPEAT_MAX) {
        for (int i = 0; i < rptTimes; i++) {
          ReduceInstr(dst + i * DstStride, src + i * Cols, 1, 0, 1, 0);
        }
      } else if constexpr (CntModeEn) {
        set_mask_count();
        set_vector_mask(0, (uint32_t)rptTimes * ElemPerRpt);
        ReduceInstr(dst, src, 0, DstStride, 1, SrcStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
      } else {
        ReduceInstr(dst, src, rptTimes, DstStride, 1, SrcStride);
      }
    }

    template <bool CntModeEn, int DstCols, int Src0Cols, int Src1Cols, uint32_t DstStride, uint32_t Src0RptStride,
      uint32_t Src1RptStride, uint8_t ElemPerRpt>
    __PTO_INSTR__ static void BinInstrByMode(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned rptTimes) {
      if constexpr (DstStride > REPEAT_MAX || Src0RptStride > REPEAT_MAX || Src1RptStride > REPEAT_MAX) {
        for (int i = 0; i < rptTimes; i++) {
          BinInstr(dst + i * DstCols, src0 + i * Src0Cols, src1 + i * Src1Cols, 1, 0, 0, 0);
        }
      } else if constexpr (CntModeEn) {
        set_mask_count();
        set_vector_mask(0, rptTimes * ElemPerRpt);
        BinInstr(dst, src0, src1, 0, DstStride, Src0RptStride, Src1RptStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
      } else {
        BinInstr(dst, src0, src1, rptTimes, DstStride, Src0RptStride, Src1RptStride);
      }
    }

  template <int TmpCols, int SrcCols, uint32_t TmpStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    __PTO_INSTR__ static void FillTmp(__ubuf__ T *tmp, __ubuf__ T *src, int srcRptPerRow, int validRow, int validCol) {
      if (validCol >= 2 * ElemPerRpt) {
        // validcol大于等于2次repeat，将完整的2次repeat比较后写入tmp
        BinInstrByMode<true, TmpCols, SrcCols, SrcCols, TmpStride, SrcStride, SrcStride, ElemPerRpt>
          (tmp, src, src + ElemPerRpt, validRow);
        pipe_barrier(PIPE_V);
      }
    }

    template <int TmpCols, int SrcCols, uint32_t TmpStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    __PTO_INSTR__ static void TmpProc(__ubuf__ T *tmp, __ubuf__ T *src, int srcRptPerRow, int validRow) {
      for (int i = 2; i < srcRptPerRow; ++i) {
        BinInstrByMode<true, TmpCols, TmpCols, SrcCols, TmpStride, TmpStride, SrcStride, ElemPerRpt>
          (tmp, tmp, src + i * ElemPerRpt, validRow);
        pipe_barrier(PIPE_V);
      }
    }
  };

  template <typename TileDataOut, typename TileDataIn>
  __PTO_INSTR__ void TRowReduceCheck(int validRow, int validCol, int dstValidRow) {
    static_assert(TileDataOut::Loc == pto::Location::Vec && TileDataIn::Loc == pto::Location::Vec,
      "This instruction only support Vec Tile");
    static_assert(TileDataIn::isRowMajor && TileDataIn::SFractal == SLayout::NoneBox,
      "This instruction only support Nd fractal Tile");
    static_assert(TileDataOut::isRowMajor && TileDataOut::SFractal == SLayout::NoneBox,
      "This instruction only support Nd fractal Tile");
    static_assert(std::is_same_v<typename TileDataIn::DType, half> ||
      std::is_same_v<typename TileDataIn::DType, float>,
      "The input data type is not supported by this instruction.");
    static_assert(std::is_same_v<typename TileDataOut::DType, typename TileDataIn::DType>,
      "The input data type must be consistent with the output data type.");
    PTO_ASSERT(validCol != 0 && validRow != 0, "The input shape is invalid, validCol or validRow is 0.");
    PTO_ASSERT(validRow == dstValidRow, "The input valid row must be consistent with the output valid row.");
  }

  template <typename InstrOp, typename T, uint32_t DstCols, uint32_t SrcCols, uint8_t elemPerRpt,
    uint32_t dstRptStride, uint32_t srcRptStride>
  __PTO_INSTR__ void OneRepeatProc(__ubuf__ T *dst, __ubuf__ T *src, int validCol, int validRow, int remain,
    int rowRptTimes) {
    if (validCol == elemPerRpt) {
      InstrOp::template ReduceInstrByMode<true, SrcCols, dstRptStride, srcRptStride, elemPerRpt>
        (dst, src, validRow);
      pipe_barrier(PIPE_V);
      return;
    }

    unsigned rptTimes;
    SetContinuousMask(remain);
    do {
      rptTimes = rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
      InstrOp::template ReduceInstrByMode<false, SrcCols, dstRptStride, srcRptStride, elemPerRpt>(dst, src, rptTimes);
      pipe_barrier(PIPE_V);
      rowRptTimes -= 1;
      dst += rptTimes * DstCols;
      src += rptTimes * SrcCols;
    } while (rowRptTimes >= 0);

    set_vector_mask(-1, -1);
  }

  template <typename InstrOp, typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
  __PTO_INSTR__ void TRowReduceInstr(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, int validCol, int validRow) {
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t dstRptStride = TileDataOut::Cols;
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;
    constexpr uint32_t tmpRptStride = TileDataTmp::Cols / elemPerBlock;
    int srcRptPerRow = validCol / elemPerRpt;
    int remain = validCol % elemPerRpt;
    int rowRptTimes = validRow / REPEAT_MAX;   // 需要处理的行若超过uint8_max, 则拆分为多次进行循环
    unsigned rptTimes;

    if (validCol <= elemPerRpt) {
      OneRepeatProc<InstrOp, T, TileDataOut::Cols, TileDataIn::Cols, elemPerRpt, dstRptStride, srcRptStride>
        (dst, src, validCol, validRow, remain, rowRptTimes);
      return;
    }

    if (validCol < 2 * elemPerRpt) {
      // 解决 ccec 编译检查问题； 如果删除会导致copy_ubuf_to_ubuf编译错误，提醒第六、七个参数的范围必须是[0, 65535]
      if constexpr ((srcRptStride < BLOCK_MAX_PER_REPEAT) || (tmpRptStride < BLOCK_MAX_PER_REPEAT)) {
        return;
      }
      // 将满足一次repeat部分copy到dst
      copy_ubuf_to_ubuf(tmp, src, 0, validRow, BLOCK_MAX_PER_REPEAT, srcRptStride - BLOCK_MAX_PER_REPEAT,
        tmpRptStride - BLOCK_MAX_PER_REPEAT);
      pipe_barrier(PIPE_V);
    }

    InstrOp::template FillTmp<TileDataTmp::Cols, TileDataIn::Cols, tmpRptStride, srcRptStride,
      elemPerRpt>(tmp, src, srcRptPerRow, validRow, validCol);

    // 不足一次repeat的部分设置mask与tmp计算, 此时tmp必定存在有效数据
    if (remain > 0) {
      __ubuf__ T *srcP = src;
      __ubuf__ T *tmpP = tmp;
      SetContinuousMask(remain);
      do {
        rptTimes = rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
        InstrOp::template BinInstrByMode<false, TileDataTmp::Cols, TileDataTmp::Cols, TileDataIn::Cols,
          tmpRptStride, tmpRptStride, srcRptStride, elemPerRpt>
          (tmpP, tmpP, srcP + srcRptPerRow * elemPerRpt, rptTimes);
        rowRptTimes -= 1;
        srcP += rptTimes * TileDataIn::Cols;
        tmpP += rptTimes * TileDataTmp::Cols;
      } while (rowRptTimes >= 0);
      set_vector_mask(-1, -1);
      pipe_barrier(PIPE_V);
    }

    InstrOp::template TmpProc<TileDataTmp::Cols, TileDataIn::Cols, tmpRptStride, srcRptStride, elemPerRpt>
      (tmp, src, srcRptPerRow, validRow);

    InstrOp::template ReduceInstrByMode<true, TileDataTmp::Cols, dstRptStride, tmpRptStride, elemPerRpt>
      (dst, tmp, validRow);
    pipe_barrier(PIPE_V);
  }

}

#endif