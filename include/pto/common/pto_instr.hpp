/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INSTR_HPP
#define PTO_INSTR_HPP

#include "pto/common/debug.h"
#include "pto/common/pto_instr_impl.hpp"

#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)

namespace pto {
template <typename T, typename AddrType>
PTO_INST void TASSIGN(T &obj, AddrType addr) {
  MAP_INSTR_IMPL(TASSIGN, obj, addr);
}

template <Op OpCode>
PTO_INST void TSYNC() {
  TSYNC_IMPL<OpCode>();
}

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents&... events) {
  waitAllEvents(events...);
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileData &dst, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TADD, dst, src0, src1);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TABS, dst, src);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUB(TileData &dst, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TSUB, dst, src0, src1);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileData &dst, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMUL, dst, src0, src1);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMIN(TileData &dst, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMIN, dst, src0, src1);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileData &dst, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMAX, dst, src0, src1);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TEXPANDS, dst, scalar);
  return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TLOAD, dst, src);
  return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename T, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc0 &src0, T src1, CmpMode cmpMode, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TCMPS, dst, src0, src1, cmpMode);
  return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1, CmpMode cmpMode,
  WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
  return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
  typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
  return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
  typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents&... events) {
  TSYNC(events...);
  TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src, preQuantScalar);
  return {};
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
  typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents&... events) {
  TSYNC(events...);
  TSTORE_IMPL<TileData, GlobalData, FpTileData, atomicType>(dst, src, fp);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileData &dst, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TDIV, dst, src0, src1);
  return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
  return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
  WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
  return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
  WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
  return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, typename Src3TileData,
          bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent
TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1,
         Src2TileData &src2, Src3TileData &src3, WaitEvents&... events) {
  TSYNC(events...);
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                Src2TileData, Src3TileData, exhausted>(
      dst, executedNumList, tmp, src0, src1, src2, src3);
  return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst,
                            MrgSortExecutedNumList &executedNumList,
                            TmpTileData &tmp, Src0TileData &src0,
                            Src1TileData &src1, Src2TileData &src2, WaitEvents&... events) {
  TSYNC(events...);
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                Src2TileData, exhausted>(dst, executedNumList, tmp, src0, src1,
                                         src2);
  return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent
TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                exhausted>(dst, executedNumList, tmp, src0, src1);
  return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, SrcTileData &src,
                            uint32_t blockLen, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen);
  return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src,
                            uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
  return {};
}

// TSORT32不自动实现wait, 需手动TSYNC(events...)
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx) {
  MAP_INSTR_IMPL(TSORT32, dst, src, idx);
  return {};
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData,
          typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp) {
  MAP_INSTR_IMPL(TSORT32, dst, src, idx, tmp);
  return {};
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TGATHER, dst, src0, src1);
  return {};
}

template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T S, WaitEvents&... events) {
  TSYNC(events...);
  TCI_IMPL<TileData, T, descending>(dst, S);
  return {};
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents&... events) {
  TSYNC(events...);
  TGATHER_IMPL<DstTileData, SrcTileData, maskPattern>(dst, src);
  return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TPARTADD, dst, src0, src1);
  return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TPARTMAX, dst, src0, src1);
  return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TPARTMIN, dst, src0, src1);
  return {};
}

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TCVT, dst, src, mode);
  return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMOV, dst, src);
  return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents&... events) {
  TSYNC(events...);
  TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
  return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
  typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents&... events) {
  TSYNC(events...);
  TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src);
  return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
  typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents&... events) {
  TSYNC(events...);
  TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp);
  return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents&... events) {
  TSYNC(events...);
  TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp);
  return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
  typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents&... events) {
  TSYNC(events...);
  TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar);
  return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
  typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents&... events) {
  TSYNC(events...);
  TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar);
  return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
  return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary,
  WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary);
  return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut &dst, TileDataIn &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TCOLMAX, dst, src);
  return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWMAX, dst, src, tmp);
  return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWMIN, dst, src, tmp);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileData &dst, TileData &src0, TileData &src1, uint8_t selectMode, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TSELS, dst, src0, src1, selectMode);
  return {};
}

template <typename TileData, typename MaskTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1);
  return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
  return {};
}

template <typename TileData, typename T, typename... WaitEvents>
PTO_INST RecordEvent TMINS(TileData &dst, TileData &src, T scalar, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMINS, dst, src, scalar);
  return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWEXPAND, dst, src);
  return {};
}

template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWEXPANDDIV, dst, src0, src1);
  return {};
}

template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1);
  return {};
}

template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TRSQRT(TileData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TRSQRT, dst, src);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TSQRT, dst, src);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TEXP, dst, src);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST void TLOG(TileData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TLOG, dst, src);
}

template <typename TileData, typename... WaitEvents>
PTO_INST void TRECIP(TileData &dst, TileData &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TRECIP, dst, src);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TGATHERB, dst, src, offset);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDS(TileData &dst, TileData &src0, typename TileData::DType scalar, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileData &dst, TileData &src0, typename TileData::DType scalar, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TDIVS, dst, src0, scalar);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMULS(TileData &dst, TileData &src0, typename TileData::DType scalar, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
  return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileData &dst, typename TileData::DType scalar, TileData &src0, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TDIVS, dst, scalar, src0);
  return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDataOut &dst, TileDataIn &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TCOLMIN, dst, src);
  return {};
}

template <typename TileData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileData &dst, TileData &src, TileInd &indexes, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TSCATTER, dst, src, indexes);
  return {};
}

} // namespace pto
#endif