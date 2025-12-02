#ifndef TMATMUL_HPP
#define TMATMUL_HPP

namespace pto {
template <typename TileAcc, typename TileLeft, typename TileRight, bool cmatrixSource, bool CmatrixInitVal>
__tf__ __aicore__ void TMatmul(typename TileAcc::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n) {
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;

    __cc__ CType *c = (__cc__ CType *)(cMatrix);
    __ca__ AType *a = (__ca__ AType *)(aMatrix);
    __cb__ BType *b = (__cb__ BType *)(bMatrix);

    bool kDirectionAlign = false; // only for f322f32
    if constexpr ((std::is_same<AType, float>::value) && (std::is_same<CType, float>::value)) {
        if constexpr (TileLeft::isRowMajor && TileLeft::SFractal == SLayout::ColMajor) {
            kDirectionAlign = true;
        }
    }

    uint8_t unitFlag = 0; // unit flag control bits

    mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, CmatrixInitVal);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__aicore__ PTO_INLINE void CheckStaticMad() {
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;
    static_assert(((std::is_same<CType, int32_t>::value) && (std::is_same<AType, int8_t>::value) &&
                      (std::is_same<BType, int8_t>::value)) ||
                      ((std::is_same<CType, float>::value) && (std::is_same<AType, half>::value) &&
                          (std::is_same<BType, half>::value)) ||
                      ((std::is_same<CType, float>::value) && (std::is_same<AType, float>::value) &&
                          (std::is_same<BType, float>::value)) ||
                      ((std::is_same<CType, float>::value) && (std::is_same<AType, bfloat16_t>::value) &&
                          (std::is_same<BType, bfloat16_t>::value)),
        "The data type is not supported.");

    static_assert(
        (TileLeft::Rows == TileAcc::Rows) && (TileLeft::Cols == TileRight::Rows) && (TileRight::Cols == TileAcc::Cols),
        "Inconsistent number of m, k, n");

    static_assert(TileLeft::Loc == Location::Left, "TileLeft location must be set to Location::Left.");
    static_assert(TileRight::Loc == Location::Right, "TileRight location must be set to Location::Right.");
    static_assert(TileAcc::Loc == Location::Acc, "TileAcc location must be set to Location::Acc.");
}

__aicore__ PTO_INLINE void CheckDynamicMad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol) {
    constexpr uint16_t elementSize = 4095;
    PTO_ASSERT(aMatrixRow >= 1 && aMatrixRow <= elementSize, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(aMatrixCol >= 1 && aMatrixCol <= elementSize, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(bMatrixCol >= 1 && bMatrixCol <= elementSize, "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__aicore__ PTO_INLINE void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix) {
    CheckStaticMad<TileAcc, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMad(m, k, n);
    TMatmul<TileAcc, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__aicore__ PTO_INLINE void TMATMUL_ACC_IMPL(
    TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix) {
    CheckStaticMad<TileAcc, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMad(m, k, n);
    TMatmul<TileAcc, TileLeft, TileRight, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
__aicore__ PTO_INLINE void TMATMUL_BIAS_IMPL(
    TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData) {
    CheckStaticMad<TileAcc, TileLeft, TileRight>();
    using CType = typename TileAcc::DType;
    using BiasType = typename TileBias::DType;
    static_assert(std::is_same_v<CType, BiasType>, "No supported bias data type");
    static_assert((TileBias::Loc == Location::Bias) && (TileBias::Rows == 1), "TileBias must be single row.");
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMad(m, k, n);

    __cc__ CType *c = (__cc__ CType *)(cMatrix.data());
    uint64_t bias = biasData.data();
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);

    TMatmul<TileAcc, TileLeft, TileRight, true, false>((__cc__ CType *)xd, aMatrix.data(), bMatrix.data(), m, k, n);
}
} // namespace pto
#endif