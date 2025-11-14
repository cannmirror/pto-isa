#ifndef TMATMUL_HPP
#define TMATMUL_HPP

namespace pto {
template <typename TileAcc, typename TileLeft, typename TileRight, bool gemvCtrl, bool CmatrixSource, bool CmatrixInitVal>
__tf__ __aicore__ void TMatmul(typename TileAcc::TileDType __out__ cMatrix,
    typename TileLeft::TileDType __in__ aMatrix, typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n)
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;

    __cc__ CType *c = (__cc__ CType *)(cMatrix);
    __ca__ AType *a = (__ca__ AType *)(aMatrix);
    __cb__ BType *b = (__cb__ BType *)(bMatrix);

    mad(c, a, b, m, k, n, 0, gemvCtrl, CmatrixSource, CmatrixInitVal);
}

template <typename TileLeft>
__aicore__ inline constexpr bool GetGemvCtrl()
{
    return TileLeft::Rows != 1;
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__aicore__ inline constexpr bool IsTypeValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;
    if constexpr ((std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t> && std::is_same_v<CType, int32_t>) ||   // s8
        (std::is_same_v<AType, half> && std::is_same_v<BType, half> && std::is_same_v<CType, float>) ||     // f162f32
        (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t> && std::is_same_v<CType, float>) || // bf162f32
        (std::is_same_v<AType, float> && std::is_same_v<BType, float> && std::is_same_v<CType, float>) ||   // f322f32
        (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e4m3_t> && std::is_same_v<CType, float>) ||   // e4m3e4m3
        (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e5m2_t> && std::is_same_v<CType, float>) ||   // e4m3e5m2
        (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e4m3_t> && std::is_same_v<CType, float>) ||   // e5m2e4m3
        (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e5m2_t> && std::is_same_v<CType, float>) ||   // e5m2e5m2
        (std::is_same_v<AType, hifloat8_t> && std::is_same_v<BType, hifloat8_t> && std::is_same_v<CType, float>)) {
        {
            return true;
        }
    }
    return false;
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__aicore__ inline void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    constexpr bool gemvCtrl = GetGemvCtrl<TileLeft>();
    constexpr bool isTypeValid = IsTypeValid<TileAcc, TileLeft, TileRight>();
    static_assert(isTypeValid, "No supported data type");

    TMatmul<TileAcc, TileLeft, TileRight, gemvCtrl, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
__aicore__ inline void TMATMUL_ACC_IMPL(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    constexpr bool gemvCtrl = GetGemvCtrl<TileLeft>();
    constexpr bool isTypeValid = IsTypeValid<TileAcc, TileLeft, TileRight>();
    static_assert(isTypeValid, "No supported data type");

    TMatmul<TileAcc, TileLeft, TileRight, gemvCtrl, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
__aicore__ inline void TMATMUL_BIAS_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    // CmatrixSource control matrix source, 0: C matrix is in L0C, 1: C matrix is in C2
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    using CType = typename TileAcc::DType;
    using BiasType = typename TileBias::DType;
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    using CType = typename TileAcc::DType;
    __cc__ CType *c = (__cc__ CType *)(cMatrix.data());
    uint64_t bias = biasData.data();

    constexpr bool gemvCtrl = GetGemvCtrl<TileLeft>();
    constexpr bool isTypeValid = IsTypeValid<TileAcc, TileLeft, TileRight>();
    static_assert(isTypeValid, "No supported data type");
    constexpr bool isBiasValid = std::is_same_v<CType, BiasType>;
    static_assert(isBiasValid, "No supported bias data type");

    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    TMatmul<TileAcc, TileLeft, TileRight, gemvCtrl, true, false>((__cc__ CType*)xd, aMatrix.data(), bMatrix.data(),m, k, n);
}
}
#endif