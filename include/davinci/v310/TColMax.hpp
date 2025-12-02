/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLMAX_HPP
#define TCOLMAX_HPP

#include "common/constants.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace pto {
    template <typename T, typename TileDataOut, typename TileDataIn>
    __tf__ __PTO_INSTR__ void TColMax(typename TileDataOut::TileDType __out__ dstData,
        typename TileDataIn::TileDType __in__ srcData, uint16_t validRow, int validCol) {
        __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
        __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);

        constexpr unsigned srcRowStride = TileDataIn::Cols;
        constexpr unsigned dstRowStride = TileDataOut::Cols;
        constexpr unsigned nElmPerRepeat = CCE_VL / sizeof(T);  // 每次repeat涉及多少个元素
        uint16_t repeatTimes = CeilDivision(validCol, nElmPerRepeat);

        if (validRow == 1) {
            __VEC_SCOPE__
            {
                RegTensor<T> VReg;
                MaskReg preg;
                uint32_t sreg = validCol;
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    preg = CreatePredicate<T>(sreg);
                    vlds(VReg, src, nElmPerRepeat, NORM, POST_UPDATE);
                    vsts(VReg, dst, nElmPerRepeat, distValue, preg, POST_UPDATE);
                }
            }
            return;
        }

        __ubuf__ T *dstP = dst;
        __ubuf__ T *srcP = src;
        __VEC_SCOPE__
        {
            RegTensor<T> src0VReg;
            RegTensor<T> src1VReg;
            RegTensor<T> tmpVReg;
            RegTensor<T> dstVReg;
            MaskReg preg;
            uint32_t sreg = validCol;
            uint16_t nLoop = (validRow - 1) / 2;    // 第一行copy 故-1
            bool remain = (validRow - 1) % 2;
            constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>
                (GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                // sreg在每次执行CreatePredicate之后会累减nElmPerRepeat，直至0
                preg = CreatePredicate<T>(sreg);

                // 将src的第一行存入dst寄存器
                vlds(dstVReg, srcP, 0, NORM);

                // 读取第二行及以后的每行数据存入src寄存器，与dst寄存器相加后存入dst寄存器
                for (uint16_t j = 0; j < nLoop; ++j) {
                    vlds(src0VReg, srcP, (2 * j + 1) * srcRowStride, NORM);
                    vlds(src1VReg, srcP, (2 * j + 2) * srcRowStride, NORM);
                    vmax(tmpVReg, src0VReg, src1VReg, preg, MODE_ZEROING);
                    vmax(dstVReg, dstVReg, tmpVReg, preg, MODE_ZEROING);
                }
                if (remain) {
                    vlds(src0VReg, srcP, (validRow - 1) * srcRowStride, NORM);
                    vmax(dstVReg, dstVReg, src0VReg, preg, MODE_ZEROING);
                }
                vsts(dstVReg, dstP, nElmPerRepeat, distValue, preg, POST_UPDATE);   // dstP每次累加nElmPerRepeat
                srcP += nElmPerRepeat;
            }
        } // end VF
    }

    template <typename TileDataOut, typename TileDataIn>
    __PTO_INSTR__ void TCOLMAX_IMPL(TileDataOut &dst, TileDataIn &src) {
        using T = typename TileDataIn::DType;
        constexpr bool isTargetType =
            std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int8_t> ||
            std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> ||
            std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, bfloat16_t>;
        static_assert(isTargetType, "The input data type is not supported by this instruction.");

        int validCol = src.GetValidCol();
        int validRow = src.GetValidRow();
        if (validCol == 0 || validRow == 0) {
            return;
        }
        TColMax<T, TileDataOut, TileDataIn>(dst.data(), src.data(), validRow, validCol);

    }
}
#endif
