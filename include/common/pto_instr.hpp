#ifndef PTO_INSTR_HPP
#define PTO_INSTR_HPP

#include "common/debug.h"
#include "common/pto_instr_impl.hpp"

#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__);

namespace pto {
template <typename TileData>
__PTO_INSTR__ void TASSIGN(TileData &tile, uint32_t addr) {
  MAP_INSTR_IMPL(TASSIGN, tile, addr)
}

template <typename TileData>
__PTO_INSTR__ void TADD(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TADD, dst, src0, src1)
}

template <typename TileData>
__PTO_INSTR__ void TSUB(TileData &dst, TileData &src0, TileData &src1) {
  MAP_INSTR_IMPL(TSUB, dst, src0, src1)
}

template <typename TileData, typename GlobalData>
__PTO_INSTR__ void TLOAD(TileData &dst, GlobalData &src) {
  MAP_INSTR_IMPL(TLOAD, dst, src)
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
__PTO_INSTR__ void TSTORE(GlobalData &dst, TileData &src) {
  TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
__PTO_INSTR__ void TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar) {
  TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src, preQuantScalar);
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone>
__PTO_INSTR__ void TSTORE(GlobalData &dst, TileData &src, FpTileData &fp) {
  TSTORE_IMPL<TileData, GlobalData, FpTileData, atomicType>(dst, src, fp);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__PTO_INSTR__ void TMATMUL(TileAcc &cMatrix, TileLeft &aMatrix,
                           TileRight &bMatrix) {
  MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix)
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__PTO_INSTR__ void TMATMUL_ACC(TileAcc &cOutMatrix, TileAcc &cInMatrix,
                               TileLeft &aMatrix, TileRight &bMatrix) {
  MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix)
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
__PTO_INSTR__ void TMATMUL_BIAS(TileAcc &cMatrix, TileLeft &aMatrix,
                                TileRight &bMatrix, TileBias &biasData) {
  MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData)
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, typename Src3TileData,
          bool exhausted>
__PTO_INSTR__ void
TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1,
         Src2TileData &src2, Src3TileData &src3) {
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                Src2TileData, Src3TileData, exhausted>(
      dst, executedNumList, tmp, src0, src1, src2, src3);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, typename Src2TileData, bool exhausted>
__PTO_INSTR__ void TMRGSORT(DstTileData &dst,
                            MrgSortExecutedNumList &executedNumList,
                            TmpTileData &tmp, Src0TileData &src0,
                            Src1TileData &src1, Src2TileData &src2) {
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                Src2TileData, exhausted>(dst, executedNumList, tmp, src0, src1,
                                         src2);
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData,
          typename Src1TileData, bool exhausted>
__PTO_INSTR__ void
TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList,
         TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1) {
  TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData,
                exhausted>(dst, executedNumList, tmp, src0, src1);
}

template <typename DstTileData, typename SrcTileData>
__PTO_INSTR__ void TMRGSORT(DstTileData &dst, SrcTileData &src,
                            uint32_t blockLen) {
  MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen)
}

template <typename DstTileData, typename SrcTileData>
__PTO_INSTR__ void TEXTRACT(DstTileData &dst, SrcTileData &src,
                            uint16_t indexRow = 0, uint16_t indexCol = 0) {
  MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol)
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData>
__PTO_INSTR__ void TSORT32(DstTileData &dst, SrcTileData &src,
                           IdxTileData &idx) {
  MAP_INSTR_IMPL(TSORT32, dst, src, idx)
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData,
          typename TmpTileData>
__PTO_INSTR__ void TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx,
                           TmpTileData &tmp) {
  MAP_INSTR_IMPL(TSORT32, dst, src, idx, tmp)
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__PTO_INSTR__ void TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1) {
  MAP_INSTR_IMPL(TGATHER, dst, src0, src1)
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
__PTO_INSTR__ void TGATHER(DstTileData &dst, SrcTileData &src) {
  TGATHER_IMPL<DstTileData, SrcTileData, maskPattern>(dst, src);
}

// TODO: uncomment if TCOPY supported for v310
// template <typename TileDataDst, typename TileDataSrc, TCopyMode copyMode>
// __PTO_INSTR__ void TCOPY(TileDataDst &dst, TileDataSrc &src) {
//   TCOPY_IMPL<TileDataDst, TileDataSrc, copyMode>(dst, src);
// }

// TODO: uncomment if TPARTADD supported for v310
// template <typename TileData>
// __PTO_INSTR__ void TPARTADD(TileData &dst, TileData &src0, TileData &src1) {
//   MAP_INSTR_IMPL(TPARTADD, dst, src0, src1)
// }

template <typename TileDataD, typename TileDataS>
__PTO_INSTR__ void TCVT(TileDataD &dst, TileDataS &src, RoundMode mode) {
  MAP_INSTR_IMPL(TCVT, dst, src, mode)
}

template <typename DstTileData, typename SrcTileData>
__PTO_INSTR__ void TMOV(DstTileData &dst, SrcTileData &src) {
  MAP_INSTR_IMPL(TMOV, dst, src)
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode>
__PTO_INSTR__ void TMOV(DstTileData &dst, SrcTileData &src) {
  TMOV_IMPL<DstTileData, SrcTileData, mode>(dst, src);
}

template <typename DstTileData, typename SrcTileData, L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
__PTO_INSTR__ void TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar) {
  TMOV_IMPL<DstTileData, SrcTileData, mode>(dst, src, preQuantScalar);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, L0cToUBMode mode = L0cToUBMode::SingleModeUB0>
__PTO_INSTR__ void TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp) {
  TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode>(dst, src, fp);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__PTO_INSTR__ void TROWSUM(TileDataOut &dst, TileDataIn &src,
                           TileDataTmp &tmp) {
  MAP_INSTR_IMPL(TROWSUM, dst, src, tmp)
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__PTO_INSTR__ void TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary) {
  MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary)
}

template <typename TileDataOut, typename TileDataIn>
__PTO_INSTR__ void TCOLMAX(TileDataOut &dst, TileDataIn &src) {
  MAP_INSTR_IMPL(TCOLMAX, dst, src)
}

template <typename TileDataDst, typename TileDataSrc>
__PTO_INSTR__ void TTRANS(TileDataDst &dst, TileDataSrc &src) {
  MAP_INSTR_IMPL(TTRANS, dst, src)
}

template <typename TileDataDst, typename TileDataSrc>
__PTO_INSTR__ void TROWEXPAND(TileDataDst &dst, TileDataSrc &src) {
  MAP_INSTR_IMPL(TROWEXPAND, dst, src)
}
} // namespace pto

#endif