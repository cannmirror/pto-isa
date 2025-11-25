#ifndef TMATMUL_HPP
#define TMATMUL_HPP

namespace pto {
    template <typename TileAcc, typename TileLeft, typename TileRight>
    void TMatmulNzZn(typename TileAcc::TileDType dst,
                       typename TileAcc::TileDType acc,
                       typename TileLeft::TileDType src0,
                       typename TileRight::TileDType src1,
                       uint16_t M, uint16_t N, uint16_t K)
    {
        static constexpr int innRowsA = TileLeft::InnerRows;
        static constexpr int innColsA = TileLeft::InnerCols;
        static constexpr int rowsA = TileLeft::Rows;
        static constexpr int innSizeA = TileLeft::InnerNumel;

        static constexpr int innRowsB = TileRight::InnerRows;
        static constexpr int innColsB = TileRight::InnerCols;
        static constexpr int colsB = TileRight::Cols;
        static constexpr int innSizeB = TileRight::InnerNumel;

        static constexpr int innRowsC = TileAcc::InnerRows;
        static constexpr int innColsC = TileAcc::InnerCols;
        static constexpr int rowsC = TileAcc::Rows;
        static constexpr int innSizeC = TileAcc::InnerNumel;

        // What is the correct usage of 'innSizeX'???
        for (uint16_t i = 0; i < M; i++) {
            for (uint16_t j = 0; j < N; j++) {
                typename TileAcc::DType mul_acc = 0;

                for (uint16_t k = 0; k < K; k++) {
                    uint16_t src0Idx =
                        (i / innRowsA) * innSizeA + (i % innRowsA) * innColsA +
                        (k / innColsA) * innColsA * rowsA + (k % innColsA);
                    uint16_t src1Idx =
                        (j / innColsB) * innSizeB + (j % innColsB) * innRowsB +
                        (k / innRowsB) * innRowsB * colsB + (k % innRowsB);

                    mul_acc += src0[src0Idx] * src1[src1Idx];
                }

                uint16_t dstIdx =
                        (i / innRowsC) * innSizeC + (i % innRowsC) * innColsC +
                        (j / innColsC) * innColsC * rowsC + (j % innColsC);
                dst[dstIdx] = acc ? acc[dstIdx] + mul_acc : mul_acc;
            }
        }
    }

    template <typename TileAcc, typename TileLeft, typename TileRight>
    __aicore__ PTO_INLINE void CheckMadValid()
    {
        using AType = typename TileLeft::DType;
        using BType = typename TileRight::DType;
        using CType = typename TileAcc::DType;
        static_assert(
            (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t> && std::is_same_v<CType, int32_t>) ||  // s8
                (std::is_same_v<AType, half> && std::is_same_v<BType, half> && std::is_same_v<CType, float>) ||  // f162f32
                (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t> &&
                    std::is_same_v<CType, float>) ||  // bf162f32
                (std::is_same_v<AType, float> && std::is_same_v<BType, float> &&
                    std::is_same_v<CType, float>)  // f322f32
            , "Not supported data type");
        static_assert(
            (TileLeft::Rows == TileAcc::Rows) && (TileLeft::Cols == TileRight::Rows) && (TileRight::Cols == TileAcc::Cols),
            "Inconsistent number of m, k, n");
        static_assert(
            ((TileLeft::Loc == Location::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
                ((TileRight::Loc == Location::Right) && (TileRight::isRowMajor) &&
                    (TileRight::SFractal == SLayout::ColMajor)) &&
                ((TileAcc::Loc == Location::Acc) && (!TileAcc::isRowMajor) && (TileAcc::SFractal == SLayout::RowMajor)),
            "Non-conforming matrix fractal");
    }

    template <typename TileAcc, typename TileBias>
    __aicore__ PTO_INLINE void CheckBiasValid()
    {
        using CType = typename TileAcc::DType;
        using BiasType = typename TileBias::DType;
        static_assert(std::is_same_v<CType, BiasType>, "No supported bias data type");
        static_assert((TileBias::Loc == Location::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
            "Non-conforming bias fractal");
    }

    template <typename TileAcc, typename TileLeft, typename TileRight>
    __aicore__ PTO_INLINE void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        CheckMadValid<TileAcc, TileLeft, TileRight>();

        uint16_t m = aMatrix.GetValidRow();
        uint16_t k = aMatrix.GetValidCol();
        uint16_t n = bMatrix.GetValidCol();

        TMatmulNzZn<TileAcc, TileLeft, TileRight>(cMatrix.data(), nullptr, aMatrix.data(), bMatrix.data(), m, n, k);
    }

    template <typename TileAcc, typename TileLeft, typename TileRight>
    __aicore__ PTO_INLINE void TMATMUL_ACC_IMPL(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        CheckMadValid<TileAcc, TileLeft, TileRight>();

        uint16_t m = aMatrix.GetValidRow();
        uint16_t k = aMatrix.GetValidCol();
        uint16_t n = bMatrix.GetValidCol();

        TMatmulNzZn<TileAcc, TileLeft, TileRight>(cOutMatrix.data(), cInMatrix.data(), aMatrix.data(), bMatrix.data(), m, n, k);
    }

    template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
    __aicore__ PTO_INLINE void TMATMUL_BIAS_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
    {
        CheckMadValid<TileAcc, TileLeft, TileRight>();
        CheckBiasValid<TileAcc, TileBias>();

        uint16_t m = aMatrix.GetValidRow();
        uint16_t k = aMatrix.GetValidCol();
        uint16_t n = bMatrix.GetValidCol();

        static_assert(false, "Not implemented yet");
    }
}
#endif