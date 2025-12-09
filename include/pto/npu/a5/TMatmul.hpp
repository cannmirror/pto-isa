/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

namespace pto {
template <typename TileLeft>
__aicore__ PTO_INLINE constexpr bool GetGemvCtrl()
{
    return TileLeft::Rows != 1;
}

template <typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource, bool cmatrixInitVal>
__tf__ __aicore__ void TMatmul(typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n)
{
    constexpr bool gemvCtrl = GetGemvCtrl<TileLeft>();

    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);

    mad(c, a, b, m, k, n, 0, gemvCtrl, cmatrixSource, cmatrixInitVal);
}

template <typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource, bool cmatrixInitVal>
__tf__ __aicore__ void TMatmulBias(typename TileRes::TileDType __out__ cMatrix,
    typename TileLeft::TileDType __in__ aMatrix, typename TileRight::TileDType __in__ bMatrix, uint64_t bias,
    uint16_t m, uint16_t k, uint16_t n)
{
    constexpr bool gemvCtrl = GetGemvCtrl<TileLeft>();

    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType *)xd;

    mad(c, a, b, m, k, n, 0, gemvCtrl, cmatrixSource, cmatrixInitVal);
}

template <typename TileRes, typename TileLeft, typename TileRight>
__aicore__ PTO_INLINE void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    static_assert(std::is_same_v<CType, int32_t> || std::is_same_v<CType, float>, "Acc Type support int32_t or float.");
    if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert(std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>,
            "Left Type and Rigth Type must be int8_t when Acc Type is int32_t.");
    } else if constexpr (std::is_same_v<CType, float>) {
        static_assert((std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
                          (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t>) ||
                          (std::is_same_v<AType, float> && std::is_same_v<BType, float>) ||
                          (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                          (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e5m2_t>) ||
                          (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                          (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e5m2_t>) ||
                          (std::is_same_v<AType, hifloat8_t> && std::is_same_v<BType, hifloat8_t>),
            "No supported data type when Acc Type is float.");
    }
    static_assert(
        (TileLeft::Rows == TileRes::Rows) && (TileLeft::Cols == TileRight::Rows) && (TileRight::Cols == TileRes::Cols),
        "Inconsistent number of m, k, n.");
    static_assert(
        ((TileLeft::Loc == Location::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == Location::Right) && (TileRight::isRowMajor) &&
                (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileRes::Loc == Location::Acc) && (!TileRes::isRowMajor) && (TileRes::SFractal == SLayout::RowMajor)),
        "Non-conforming matrix fractal.");
}

template <typename TileRes, typename TileLeft, typename TileRight>
__aicore__ PTO_INLINE void TMATMUL_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    TMatmul<TileRes, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <typename TileRes, typename TileLeft, typename TileRight>
__aicore__ PTO_INLINE void TMATMUL_ACC_IMPL(
    TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    TMatmul<TileRes, TileLeft, TileRight, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
__aicore__ PTO_INLINE void TMATMUL_BIAS_IMPL(
    TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    // cmatrixSource control matrix source, 0: C matrix is in L0C, 1: C matrix is in C2
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert((TileBias::Loc == Location::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
        "Non-conforming bias fractal.");

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();

    TMatmulBias<TileRes, TileLeft, TileRight, true, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n);
}
} // namespace pto
#endif