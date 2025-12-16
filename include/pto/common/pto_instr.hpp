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

template <typename TileData>
PTO_INST void TADD(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TADD, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TSUB(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TSUB, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TMUL(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TMUL, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TMIN(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TMIN, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TMAX(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TMAX, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TEXPANDS(TileData &dst, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TEXPANDS, dst, scalar);
}

template <typename TileData, typename GlobalData>
PTO_INST void TLOAD(TileData &dst, GlobalData &src) {
  MAP_INSTR_IMPL(TLOAD, dst, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename T>
PTO_INST void TCMPS(TileDataDst &dst, TileDataSrc0 &src0, T src1, CmpMode cmpMode) {
  MAP_INSTR_IMPL(TCMPS, dst, src0, src1, cmpMode);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1, CmpMode cmpMode) {
  MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INST void TSTORE(GlobalData &dst, TileData &src) {
  TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INST void TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar) {
  TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src, preQuantScalar);
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INST void TSTORE(GlobalData &dst, TileData &src, FpTileData &fp) {
  TSTORE_IMPL<TileData, GlobalData, FpTileData, atomicType>(dst, src, fp);
}

template <typename TileData>
PTO_INST void TDIV(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TDIV, dst, src0, src1);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INST void TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix) {
  MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INST void TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix) {
  MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INST void TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData) {
  MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, typename Src3TileData,
          bool exhausted>
PTO_INST void
TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1,
         Src2TileData &src2, Src3TileData &src3) {
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                Src2TileData, Src3TileData, exhausted>(
      dst, executedNumList, tmp, src0, src1, src2, src3);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, bool exhausted>
PTO_INST void TMRGSORT(DstTileData &dst,
                            MrgSortExecutedNumList &executedNumList,
                            TmpTileData &tmp, Src0TileData &src0,
                            Src1TileData &src1, Src2TileData &src2) {
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                Src2TileData, exhausted>(dst, executedNumList, tmp, src0, src1,
                                         src2);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, bool exhausted>
PTO_INST void
TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1) {
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                exhausted>(dst, executedNumList, tmp, src0, src1);
}

template <typename DstTileData, typename SrcTileData>
PTO_INST void TMRGSORT(DstTileData &dst, SrcTileData &src,
                            uint32_t blockLen) {
  MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen);
}

template <typename DstTileData, typename SrcTileData>
PTO_INST void TEXTRACT(DstTileData &dst, SrcTileData &src,
                            uint16_t indexRow = 0, uint16_t indexCol = 0) {
  MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST void TSORT32(DstTileData &dst, SrcTileData &src,
                           IdxTileData &idx) {
  MAP_INSTR_IMPL(TSORT32, dst, src, idx);
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData,
          typename TmpTileData>
PTO_INST void TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx,
                           TmpTileData &tmp) {
  MAP_INSTR_IMPL(TSORT32, dst, src, idx, tmp);
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
PTO_INST void TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1) {
  MAP_INSTR_IMPL(TGATHER, dst, src0, src1);
}

template <typename TileData, typename T, int descending>
PTO_INST void TCI(TileData &dst, T S) {
  TCI_IMPL<TileData, T, descending>(dst, S);
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
PTO_INST void TGATHER(DstTileData &dst, SrcTileData &src) {
  TGATHER_IMPL<DstTileData, SrcTileData, maskPattern>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TPARTADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TPARTADD, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TPARTMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TPARTMAX, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TPARTMIN, dst, src0, src1);
}

template <typename TileDataD, typename TileDataS>
PTO_INST void TCVT(TileDataD &dst, TileDataS &src, RoundMode mode) {
  MAP_INSTR_IMPL(TCVT, dst, src, mode);
}

template <typename DstTileData, typename SrcTileData>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src) {
  MAP_INSTR_IMPL(TMOV, dst, src);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src) {
  TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src) {
  TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp) {
  TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp) {
  TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar) {
  TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar) {
  TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TROWSUM(TileDataOut &dst, TileDataIn &src,
                           TileDataTmp &tmp) {
  MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary) {
  MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INST void TCOLMAX(TileDataOut &dst, TileDataIn &src) {
  MAP_INSTR_IMPL(TCOLMAX, dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TROWMAX(TileDataOut &dst, TileDataIn &src,
                           TileDataTmp &tmp) {
  MAP_INSTR_IMPL(TROWMAX, dst, src, tmp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TROWMIN(TileDataOut &dst, TileDataIn &src,
                           TileDataTmp &tmp) {
  MAP_INSTR_IMPL(TROWMIN, dst, src, tmp);
}

template <typename TileData>
PTO_INST void TSELS(TileData &dst, TileData &src0, TileData &src1, uint8_t selectMode) {
  MAP_INSTR_IMPL(TSELS, dst, src0, src1, selectMode);
}

template <typename TileData, typename MaskTile>
PTO_INST void TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INST void TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp) {
  MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
}

template <typename TileData, typename T>
PTO_INST void TMINS(TileData &dst, TileData &src, T scalar) {
  MAP_INSTR_IMPL(TMINS, dst, src, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TROWEXPAND(TileDataDst &dst, TileDataSrc &src) {
  MAP_INSTR_IMPL(TROWEXPAND, dst, src);
}

template <typename TileDataDst, typename TileDataSrc1>
PTO_INST void TROWEXPANDDIV(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TROWEXPANDDIV, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc1>
PTO_INST void TROWEXPANDMUL(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc1>
PTO_INST void TROWEXPANDSUB(TileDataDst &dst, TileDataDst &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TRSQRT(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TRSQRT, dst, src);
}

template <typename TileData>
PTO_INST void TSQRT(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TSQRT, dst, src);
}

template <typename TileData>
PTO_INST void TEXP(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TEXP, dst, src);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset>
PTO_INST void TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset) {
  MAP_INSTR_IMPL(TGATHERB, dst, src, offset);
}

template <typename TileData>
PTO_INST void TADDS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TDIVS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TDIVS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TMULS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TDIVS(TileData &dst, typename TileData::DType scalar, TileData &src0) {
  MAP_INSTR_IMPL(TDIVS, dst, scalar, src0);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INST void TCOLMIN(TileDataOut &dst, TileDataIn &src) {
  MAP_INSTR_IMPL(TCOLMIN, dst, src);
}

template <typename TileData, typename TileInd>
PTO_INST void TSCATTER(TileData &dst, TileData &src, TileInd &indexes) {
  MAP_INSTR_IMPL(TSCATTER, dst, src, indexes);
}

} // namespace pto
#endif