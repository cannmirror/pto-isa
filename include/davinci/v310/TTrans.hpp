/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRANS_HPP
#define TTRANS_HPP

#include "common/utils.hpp"
#include "common/constants.hpp"
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {
    
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem>
    __tf__ __aicore__ PTO_INLINE void TTransB32(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, float>) {
            __VEC_SCOPE__{
                RegTensor<uint32_t> vreg0;
                RegTensor<T> vreg1;
                uint16_t aligned_Rows = CeilDivision(TileData::Rows, blockSizeElem) * blockSizeElem;
                uint32_t sreg = (uint32_t)(aligned_Rows * TileData::Cols);
                constexpr uint32_t sregLower = elementsPerRepeat;
                MaskReg preg = CreatePredicate<T>(sreg);
                uint16_t repeatTimes = CeilDivision(aligned_Rows, sregLower);
                
                constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM_B32>())>();
                for (uint16_t col = 0; col < (uint16_t)TileData::Cols; ++col) {
                    for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                        vci((RegTensor<int32_t> &)vreg0, (int32_t)(chunk * sregLower), INC_ORDER);
                        vmuls(vreg0, vreg0, TileData::Cols, preg);
                        vadds(vreg0, vreg0, col, preg);
                        vgather2(vreg1, srcPtr, (RegTensor<uint32_t> &)vreg0, preg);
                        vsts(vreg1, dstPtr, (col * aligned_Rows + chunk * sregLower), distValue, preg);
                    }
                }
            }
        } else {
            static_assert(sizeof(T) == 4, "TTRANS: Invalid data type.");
        }
    }

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem>
    __tf__ __aicore__ PTO_INLINE void TTransB16(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            __VEC_SCOPE__{
                RegTensor<uint16_t> vreg0;
                RegTensor<T> vreg1;
                uint16_t aligned_Rows = CeilDivision(TileData::Rows, blockSizeElem) * blockSizeElem;
                uint32_t sreg = (uint32_t)(aligned_Rows * TileData::Cols);
                constexpr uint32_t sregLower = elementsPerRepeat;
                MaskReg preg = CreatePredicate<T>(sreg);
                uint16_t repeatTimes = CeilDivision(aligned_Rows, sregLower);
                constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM_B16>())>();
                    for (uint16_t col = 0; col < (uint16_t)TileData::Cols; ++col) {
                        for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                            vci((RegTensor<int16_t> &)vreg0, (int16_t)(chunk * sregLower), INC_ORDER);
                            vmuls(vreg0, vreg0, TileData::Cols, preg);
                            vadds(vreg0, vreg0, col, preg);
                            vgather2(vreg1, srcPtr, (RegTensor<uint16_t> &)vreg0, preg);
                            vsts(vreg1, dstPtr, (col * aligned_Rows + chunk * sregLower), distValue, preg);
                        }
                    }
            }
        } else {
            static_assert(sizeof(T) == 2, "TTRANS: Invalid data type.");
        }
    }

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem>
    __tf__ __aicore__ PTO_INLINE void TTransB8(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
            __VEC_SCOPE__{
                RegTensor<uint16_t> vreg0;
                RegTensor<T> vreg1;
                uint16_t aligned_Rows = CeilDivision(TileData::Rows, blockSizeElem) * blockSizeElem;
                uint32_t sreg = (uint32_t)(aligned_Rows * TileData::Cols);
                constexpr uint32_t sregLower = elementsPerRepeat >> 1;
                MaskReg preg = CreatePredicate<T>(sreg);
                uint16_t repeatTimes = CeilDivision(aligned_Rows, sregLower);
                constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_PK_B16>())>();
                    for (uint16_t col = 0; col < (uint16_t)TileData::Cols; ++col) {
                        for (uint16_t chunk = 0; chunk < repeatTimes; ++chunk) {
                            vci((RegTensor<int16_t> &)vreg0, (int16_t)(chunk * sregLower), INC_ORDER);
                            vmuls(vreg0, vreg0, TileData::Cols, preg);
                            vadds(vreg0, vreg0, col, preg);
                            vgather2((RegTensor<uint16_t> &)vreg1, (__ubuf__ uint8_t *)srcPtr, (RegTensor<uint16_t> &)vreg0, preg);
                            vsts(vreg1, dstPtr, (col * aligned_Rows + chunk * sregLower), distValue, preg);
                        }
                    }
            }
        } else {
            static_assert(sizeof(T) == 1, "TTRANS: Invalid data type.");
        }
    }

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src) {

        using T = typename TileDataSrc::DType; 

        static_assert(sizeof(T) == sizeof(typename TileDataDst::DType), "TTRANS: Inconsistent source and destination data types.");

        if constexpr (TileDataSrc::isRowMajor) {
            static_assert(TileDataSrc::Cols * sizeof(typename TileDataSrc::DType) % 32 == 0, "TTRANS: Inconsistent Input Shape.");
            static_assert(TileDataDst::Cols * sizeof(typename TileDataDst::DType) % 32 == 0, "TTRANS: Inconsistent Output Shape.");
        } else {
            static_assert(TileDataSrc::Rows * sizeof(typename TileDataSrc::DType) % 32 == 0, "TTRANS: Inconsistent Input Shape.");
            static_assert(TileDataDst::Rows * sizeof(typename TileDataDst::DType) % 32 == 0, "TTRANS: Inconsistent Output Shape.");
        }

        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);//REPEAT_BYTE = 256
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);

        if constexpr (sizeof(T) == 4) {
            TTransB32<TileDataSrc, elementsPerRepeat, blockSizeElem>(dst.data(), src.data());
        } else if constexpr (sizeof(T) == 2) {
            TTransB16<TileDataSrc, elementsPerRepeat, blockSizeElem>(dst.data(), src.data());
        } else if constexpr (sizeof(T) == 1) {
            TTransB8<TileDataSrc, elementsPerRepeat, blockSizeElem>(dst.data(), src.data());
        } else {
            static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TTRANS: Invalid data type.");
        }
    }

} // namespace pto

#endif // TTRANS_HPP