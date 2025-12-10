/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TUNARYOP_HPP
#define TUNARYOP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

enum class UnaryOpsImpl : uint8_t {
  UnaryOpsIMPL_DEFAULT = 0,
  UnaryOpsIMPL_1D_NO_POST_UPDATE = 1,
  UnaryOpsIMPL_2D_NO_POST_UPDATE = 2,
  UnaryOpsIMPL_1D_POST_UPDATE = 3,
  UnaryOpsIMPL_2D_POST_UPDATE = 4,
};

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TUnaryOps_1D_NoPostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *srcPtr,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidRows * kValidCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        MaskReg  preg;

        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = kValidRows * kValidCols;
        #pragma clang loop unroll(disable)
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, srcPtr, i*elementsPerRepeat, NORM);
            Op::UnaryInstr(vreg1, vreg0, preg);
            vsts(vreg1, dstPtr, i*elementsPerRepeat, distValue, preg);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TUnaryOps_1D_PostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *srcPtr,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidRows * kValidCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU;
        RegTensor<T> vreg1_PU;
        MaskReg  preg;

        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = kValidRows * kValidCols;
        #pragma clang loop unroll(disable)
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0_PU, srcPtr, elementsPerRepeat, NORM, POST_UPDATE);
            Op::UnaryInstr(vreg1_PU, vreg0_PU, preg);
            vsts(vreg1_PU, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
        }
    } 
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TUnaryOps_2D_NoPostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *srcPtr,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        MaskReg preg;
        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
            uint32_t sreg = (uint32_t)(kValidCols);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, srcPtr, i * rowStride + j * elementsPerRepeat, NORM);
                Op::UnaryInstr(vreg1, vreg0, preg);
                vsts(vreg1, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TUnaryOps_2D_PostUpdate(__ubuf__ typename TileData::DType *dstPtr, 
                    __ubuf__ typename TileData::DType *srcPtr,
                    unsigned kValidRows,
                    unsigned kValidCols) {
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU;
        RegTensor<T> vreg1_PU;
        MaskReg preg;
        constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                vlds(vreg0_PU, srcPtr, i * rowStride + j * elementsPerRepeat, NORM);
                uint32_t count = ((j + 1) * elementsPerRepeat >= kValidCols ? kValidCols - j * elementsPerRepeat : elementsPerRepeat);
                preg = CreatePredicate<T>(count);
                Op::UnaryInstr(vreg1_PU, vreg0_PU, preg);
                vsts(vreg1_PU, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void TUnaryOps_1D_Switch(__ubuf__ typename TileData::DType *dstPtr,
                    __ubuf__ typename TileData::DType *srcPtr,
                    unsigned kValidRows,
                    unsigned kValidCols,
                    UnaryOpsImpl version) {
    switch (version) {
    case UnaryOpsImpl::UnaryOpsIMPL_1D_NO_POST_UPDATE:
    case UnaryOpsImpl::UnaryOpsIMPL_2D_NO_POST_UPDATE:
        TUnaryOps_1D_NoPostUpdate<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, srcPtr, kValidRows, kValidCols);
        break;
    case UnaryOpsImpl::UnaryOpsIMPL_DEFAULT:
    case UnaryOpsImpl::UnaryOpsIMPL_1D_POST_UPDATE:
    case UnaryOpsImpl::UnaryOpsIMPL_2D_POST_UPDATE:
    default:
        TUnaryOps_1D_PostUpdate<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, srcPtr, kValidRows, kValidCols);
        break;
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL
void UnaryInstr(typename TileData::TileDType __out__ dst,
                            typename TileData::TileDType __in__ src,
                            unsigned kValidRows,
                            unsigned kValidCols,
                            UnaryOpsImpl version) {
    if constexpr (TileData::ValidCol == TileData::Cols) {
        TUnaryOps_1D_Switch<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src, kValidRows, kValidCols, version);
    } else {
        if (TileData::Cols == kValidCols) {
            TUnaryOps_1D_Switch<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src, kValidRows, kValidCols, version);
        } else {
            switch (version) {
            case UnaryOpsImpl::UnaryOpsIMPL_1D_NO_POST_UPDATE:
            case UnaryOpsImpl::UnaryOpsIMPL_2D_NO_POST_UPDATE:
                TUnaryOps_2D_NoPostUpdate<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src, kValidRows, kValidCols);
                break;
            case UnaryOpsImpl::UnaryOpsIMPL_DEFAULT:
            case UnaryOpsImpl::UnaryOpsIMPL_1D_POST_UPDATE:
            case UnaryOpsImpl::UnaryOpsIMPL_2D_POST_UPDATE:
            default:
                TUnaryOps_2D_PostUpdate<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst, src, kValidRows, kValidCols);
                break;
            }
        }
    }
}


template<typename T> using unaryFuncPtr = void (*)(RegTensor<T> &reg_dst, RegTensor<T> &reg_src, MaskReg &preg);

template <typename T, unaryFuncPtr<T> funcPtr> struct UnaryOperation {
    PTO_INTERNAL static void UnaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src, MaskReg &preg)
    {
        funcPtr(reg_dst, reg_src, preg);
    }
};

template <typename TileData, unaryFuncPtr<typename TileData::DType> funcPtr, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ PTO_INTERNAL
void TUnaryOp(typename TileData::TileDType __out__ dst, 
                            typename TileData::TileDType __in__ src,
                            unsigned kValidRows,
                            unsigned kValidCols,
                            UnaryOpsImpl version = UnaryOpsImpl::UnaryOpsIMPL_DEFAULT) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    UnaryInstr<UnaryOperation<T, funcPtr>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(
                dstPtr, srcPtr, kValidRows, kValidCols, version);
}

/* TEXP */
template<typename T> AICORE void _vexp(RegTensor<T> &reg_dst, RegTensor<T> &reg_src, MaskReg &preg) {
    vexp(reg_dst, reg_src, preg, MODE_ZEROING);
}

template <typename TileData>
AICORE void TEXP_IMPL(TileData &dst, TileData &src) {
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    static_assert(TileData::isRowMajor, "TEXP: not supported Layout type");
    static_assert(std::is_same_v<typename TileData::DType, float> || std::is_same_v<typename TileData::DType, half>, "TEXP: not supported Layout type");

    TUnaryOp<TileData, _vexp, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
}

/* TSQRT */
template<typename T> AICORE void _vsqrt(RegTensor<T> &reg_dst, RegTensor<T> &reg_src, MaskReg &preg) {
    vsqrt(reg_dst, reg_src, preg, MODE_ZEROING);
}

template <typename TileData>
AICORE void TSQRT_IMPL(TileData &dst, TileData &src) {
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    static_assert(TileData::isRowMajor, "TSQRT: not supported Layout type");
    static_assert(std::is_same_v<typename TileData::DType, float> || std::is_same_v<typename TileData::DType, half>, "TSQRT: not supported Layout type");

    TUnaryOp<TileData, _vsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
}

/* TRSQRT */
template <typename TileData>
__tf__ AICORE void TRsqrtCustom(typename TileData::TileDType __out__ dst,
                                    typename TileData::TileDType __in__ src,
                                    unsigned validCol, unsigned validRow) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        RegTensor<T> vreg2;
        RegTensor<T> vreg3;
        uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileData::DType));
        uint16_t loop_num = CEIL(validCol, batch_size);
        uint32_t count = (batch_size >= validCol ? validCol : batch_size);
        MaskReg preg = CreatePredicate<T>(count);
        vdup(vreg2, (T)1.0, preg, MODE_MERGING);
        for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
            for(uint16_t j = 0; j < loop_num; ++j) {
                vlds(vreg0, srcPtr, (i * TileData::Cols + j * batch_size), NORM);
                count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                preg = CreatePredicate<T>(count);
                vsqrt(vreg1, vreg0, preg, MODE_ZEROING);
                vdiv(vreg3, vreg2, vreg1, preg);
                
                vsts(vreg3, dstPtr, (i * TileData::Cols + j * batch_size), NORM_B32, preg);
            }
        }
    }
}

template <typename TileData>
AICORE void TRSQRT_IMPL(TileData &dst, TileData &src) {
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    static_assert(TileData::isRowMajor, "TRSQRT: not supported Layout type");
    static_assert(std::is_same_v<typename TileData::DType, float> || std::is_same_v<typename TileData::DType, half>, "TRSQRT: not supported Layout type");

    TRsqrtCustom<TileData>(dst.data(), src.data(), validRow, validCol);
}

}
#endif