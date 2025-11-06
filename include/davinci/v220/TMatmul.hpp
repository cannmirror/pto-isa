#ifndef TMATMUL_HPP
#define TMATMUL_HPP

namespace pto {
    template <typename TileAcc, typename TileLeft, typename TileRight, bool cmatrixSource, bool cmatrixInitVal>
    __tf__ __aicore__ void TMatmul(typename TileAcc::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
        typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n)
    {
        using AType = typename TileLeft::DType;
        using BType = typename TileRight::DType;
        using CType = typename TileAcc::DType;

        __cc__ CType *c = (__cc__ CType *)(cMatrix);
        __ca__ AType *a = (__ca__ AType *)(aMatrix);
        __cb__ BType *b = (__cb__ BType *)(bMatrix);

        bool kDirectionAlign = false;  // only for f322f32
        if constexpr ((std::is_same<AType, float>::value) && (std::is_same<CType, float>::value)) {
            if constexpr(TileLeft::isRowMajor && TileLeft::SFractal == SLayout::ColMajor) {  // zn A transpose
                kDirectionAlign = true;
            }
        }

        // Indicates the Cmatrix source, 1: the C matrix is in bias table buffer, 0: the C matrix is in L0C
        uint8_t unitFlag = 0;  // unit flag control bits

        mad(c, a, b, m, k, n, unitFlag, KDirectionAlign, camatrixSource, CmatrixInitVal);
    }

    template <typename TileAcc, typename TileLeft, typename TileRight>
    __aicore__ PTO_INLINE void CheckStatic() 
    {
        static_assert(((std::is_same<typename TileAcc::DType, int32_t>::value) && (std::is_same<typename TileLeft::DType, int8_t>::value) && (std::is_same<typename TileRight::DType, int8_t>::value)) ||
                      ((std::is_same<typename TileAcc::DType, float>::value) && (std::is_same<typename TileLeft::DType, half>::value) && (std::is_same<typename TileRight::DType, half>::value)) ||
                      ((std::is_same<typename TileAcc::DType, float>::value) && (std::is_same<typename TileLeft::DType, float>::value) && (std::is_same<typename TileRight::DType, float>::value)) ||
                      ((std::is_same<typename TileAcc::DType, float>::value) && (std::is_same<typename TileLeft::DType, bfloat16_t>::value) && (std::is_same<typename TileRight::DType, bfloat16_t>::value)),
                      "The data type is not supported.");
        constexpr uint16_t m = TileAcc::Rows;
        constexpr uint16_t k = TileAcc::Cols;
        constexpr uint16_t n = TileLeft::Cols;
        constexpr size_t leftSize = (m * k) * sizeof(typename TileLeft::DType);
        constexpr size_t rightSize = (k * n) * sizeof(typename TileRight::DType);
        constexpr size_t accSize =(m * n) * sizeof(typename TileAcc::DType);
        static_assert(leftSize <= 64 * 1024, "The size of left matrix is out of range.");
        static_assert(rightSize <= 64 * 1024, "The size of right matrix is out of range.");
        static_assert(accSize <= 256 * 1024, "The size of acc matrix is out of range.");

        static_assert(TileLeft::Loc == Location::Left, "TileLeft location must be Left!");
        static_assert(TileRight::Loc == Location::Right, "TileRight location must be Right!");
        static_assert(TileAcc::Loc == Location::Acc, "TileAcc location must be Acc!");
    }

    template <typename TileAcc, typename TileLeft, typename TileRight>
    __aicore__ PTO_INLINE void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        CheckStatic<TileAcc, TileLeft, TileRight>();
        // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0: use the real number in C matrix
        uint16_t m = aMatrix.GetValidRow();
        uint16_t k = aMatrix.GetValidCol();
        uint16_t n = bMatrix.GetValidCol();
        TMatmul<TileAcc, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
    }

    template <typename TileAcc, typename TileLeft, typename TileRight>
    __aicore__ PTO_INLINE void TMATMUL_ACC_IMPL(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        CheckStatic<TileAcc, TileLeft, TileRight>();
        // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0: use the real number in C matrix
        uint16_t m = aMatrix.GetValidRow();
        uint16_t k = aMatrix.GetValidCol();
        uint16_t n = bMatrix.GetValidCol();
        TMatmul<TileAcc, TileLeft, TileRight, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
    }

    template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
    __aicore__ PTO_INLINE void TMATMUL_BIAS_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
    {
        CheckStatic<TileAcc, TileLeft, TileRight>();
        using CType = typename TileAcc::DType;
        using BiasType = typename TileBias::DType;
        constexpr bool isBiasValid = std::is_same_v<CType, BiasType>;
        static_assert(isBiasValid, "No supported bias data type");
        static_assert(TileBias::Rows == 1, "TileBias must be single row.");
        static_assert((TileBias::Rows) * sizeof(typename TileAcc::DType) <= 1024, "The size of bias matrix is out of range.");
        uint16_t m = aMatrix.GetValidRow();
        uint16_t k = aMatrix.GetValidCol();
        uint16_t n = bMatrix.GetValidCol();

        __cc__ CType *c = (__cc__ CType *)(cMatrix.data());
        uint64_t bias = biasMatrix.data();
        uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);

        TMatmul<TileAcc, TileLeft, TileRight, true, false>((__cc__ CType *)xd, aMatrix.data(), bMatrix.data(), m, k, n);
    }
}
#endif