/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWSUM_HPP
#define TROWSUM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {
    const uint32_t ADDR_ALIGN = 32;
    template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat, unsigned blockSizeElem>
    __tf__ AICORE void TRowSum(typename TileDataOut::TileDType __out__ dst,
                                   typename TileDataIn::TileDType __in__ src, uint32_t rows, uint32_t cols) {
        using TOUT = typename TileDataOut::DType;
        using TIN = typename TileDataIn::DType;
        __ubuf__ TOUT *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
        __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);
        if constexpr (std::is_same_v<TIN, float> || std::is_same_v<TIN, bfloat16_t>) {
            constexpr uint32_t regItems = std::is_same_v<TIN, float> ? ELE_CNT_B32 : std::is_same_v<TIN, bfloat16_t> ? ELE_CNT_B16 : ELE_CNT_B8;
            __VEC_SCOPE__
            {
                RegTensor<TIN> vreg0;
                RegTensor<TIN> vreg1;
                RegTensor<TIN> vregdst;
                uint16_t repeatTimes = CeilDivision(cols, regItems);
                constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TIN, DistVST::DIST_NORM>())>();
                uint32_t destItems = 1;
                MaskReg pregdst = CreatePredicate<TIN>(destItems);
                for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                    vbr(vregdst, 0);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                        vlds(vreg0, srcPtr + i * TileDataIn::Cols, j * regItems, NORM);
                        uint32_t availItems = min(regItems, cols - j * regItems);
                        MaskReg preg = CreatePredicate<TIN>(availItems);
                        vcadd(vreg1, vreg0, preg, MODE_ZEROING);
                        vadd(vregdst, vregdst, vreg1, pregdst, MODE_ZEROING);
                    }
                    vsts(vregdst, dstPtr + i * TileDataOut::Cols, 0, distValue, pregdst);
                }
            } // end VF
        }
    }

    template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
    AICORE void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataIn::DType);  // 每个block涉及多少个元素
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataIn::DType);  // 每次repeat涉及多少个元素

        // 静态场景下优化合并处理
        TRowSum<TileDataOut, TileDataIn, elementsPerRepeat, blockSizeElem>(dst.data(), src.data(),
            src.GetValidRow(), src.GetValidCol());
    }
}
#endif
