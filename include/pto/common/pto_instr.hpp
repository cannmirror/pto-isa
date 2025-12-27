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

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TADD, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TABS(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TABS, dst, src);
}

template <typename TileData>
PTO_INST void TSUB(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TSUB, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TSUB, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TMUL(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TMUL, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TMUL, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TMIN(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TMIN, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TMIN, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TMAX(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TMAX, dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
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

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INST void TDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
  MAP_INSTR_IMPL(TDIV, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TREM(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TREM, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TSHL(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TSHL, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TSHR(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TSHR, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TAND(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TAND, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TOR(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TOR, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TXOR(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TXOR, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TLOG(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TLOG, dst, src);
}

template <typename TileData>
PTO_INST void TNEG(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TNEG, dst, src);
}

template <typename TileData>
PTO_INST void TNOT(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TNOT, dst, src);
}

template <typename TileData>
PTO_INST void TRECIP(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TRECIP, dst, src);
}

template <typename TileData>
PTO_INST void TRELU(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TRELU, dst, src);
}

template <typename TileData>
PTO_INST void TPRELU(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TPRELU, dst, src0, src1);
}

template <typename TileData>
PTO_INST void TADDC(TileData &dst, TileData &src0, TileData &src1, TileData &src2) {
  MAP_INSTR_IMPL(TADDC, dst, src0, src1, src2);
}

template <typename TileData>
PTO_INST void TSUBC(TileData &dst, TileData &src0, TileData &src1, TileData &src2) {
  MAP_INSTR_IMPL(TSUBC, dst, src0, src1, src2);
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
PTO_INST void TCI(TileData &dst, T start) {
  TCI_IMPL<TileData, T, descending>(dst, start);
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

template <typename DstTileData, typename SrcTileData, AccToVecMode mode>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src) {
  TMOV_IMPL<DstTileData, SrcTileData, mode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode = AccToVecMode::SingleModeVec0>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar) {
  TMOV_IMPL<DstTileData, SrcTileData, mode>(dst, src, preQuantScalar);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode = AccToVecMode::SingleModeVec0>
PTO_INST void TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp) {
  TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode>(dst, src, fp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TROWSUM(TileDataOut &dst, TileDataIn &src,
                           TileDataTmp &tmp) {
  MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INST void TCOLSUM(TileDataOut &dst, TileDataIn &src) {
  MAP_INSTR_IMPL(TCOLSUM, dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INST void TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary
  ) {
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

template <typename TileDataOut, typename TileDataIn>
PTO_INST void TRESHAPE(TileDataOut &dst, TileDataIn &src) {
  MAP_INSTR_IMPL(TRESHAPE, dst, src);
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

template <typename TileData>
PTO_INST void TMINS(TileData &dst, TileData &src, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TMINS, dst, src, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar) {
  MAP_INSTR_IMPL(TMINS, dst, src, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TROWEXPAND(TileDataDst &dst, TileDataSrc &src) {
  TROWEXPAND_IMPL<TileDataDst, TileDataSrc>(dst, src);
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

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TRSQRT(TileDataDst &dst, TileDataSrc &src) {
  MAP_INSTR_IMPL(TRSQRT, dst, src);
}

template <typename TileData>
PTO_INST void TSQRT(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TSQRT, dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TSQRT(TileDataDst &dst, TileDataSrc &src) {
  MAP_INSTR_IMPL(TSQRT, dst, src);
}

template <typename TileData>
PTO_INST void TEXP(TileData &dst, TileData &src) {
  MAP_INSTR_IMPL(TEXP, dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TEXP(TileDataDst &dst, TileDataSrc &src) {
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

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TADDS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar) {
  MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TSUBS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TSUBS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TDIVS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TDIVS, dst, src0, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar) {
  MAP_INSTR_IMPL(TDIVS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TREMS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TREMS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TMAXS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TMAXS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TANDS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TANDS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TORS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TORS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TXORS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TXORS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TLRELU(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TLRELU, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TMULS(TileData &dst, TileData &src0, typename TileData::DType scalar) {
  MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TMULS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar) {
  MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
}

template <typename TileData>
PTO_INST void TADDSC(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1) {
  MAP_INSTR_IMPL(TADDSC, dst, src0, scalar, src1);
}

template <typename TileData>
PTO_INST void TSUBSC(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1) {
  MAP_INSTR_IMPL(TSUBSC, dst, src0, scalar, src1);
}

template <typename TileData>
PTO_INST void TDIVS(TileData &dst, typename TileData::DType scalar, TileData &src0) {
  MAP_INSTR_IMPL(TDIVS, dst, scalar, src0);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDIVS(TileDataDst &dst, typename TileDataSrc::DType scalar, TileDataSrc &src0) {
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

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TCOLEXPAND(TileDataDst &dst, TileDataSrc &src) {
  MAP_INSTR_IMPL(TCOLEXPAND, dst, src);
}

template <typename TileDst, typename GlobalData, typename TileInd>
PTO_INST void MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes) {

  MAP_INSTR_IMPL(MGATHER, dst, src, indexes);
}

template <typename TileData, typename TileDataIdx>
PTO_INST void MSCATTER(TileData &src0, typename TileData::TileDType data, TileDataIdx &src1) {

  MAP_INSTR_IMPL(MSCATTER, src0, data, src1);
}

} // namespace pto
#endif
