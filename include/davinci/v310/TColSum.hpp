/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLSUM_HPP
#define TCOLSUM_HPP

#include "common/constants.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace pto {
    template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp, bool isBinary>
    __tf__ __aicore__ PTO_INLINE void TColSum(typename TileDataOut::TileDType __out__ dstData,
                                              typename TileDataIn::TileDType __in__ srcData,
                                              typename TileDataIn::TileDType __in__ tmpData,
                                              uint16_t validRow, int validCol) {
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
        if constexpr (isBinary) {
            __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
            __ubuf__ T *tmpP = tmp;
            constexpr unsigned tmpRowStride = TileDataTmp::Cols;
            __VEC_SCOPE__
            {
                RegTensor<T> src0VReg;
                RegTensor<T> src1VReg;
                RegTensor<T> tmpVReg;
                MaskReg preg;
                uint32_t sreg = validCol;
                uint16_t nLoop;
                uint16_t i, j, k;
                uint16_t BinaryAccLoopTimes;
                bool remain;
                constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>
                    (GetDistVst<T, DistVST::DIST_NORM>())>();

                for (i = 0; i < repeatTimes; ++i) {
                    // sreg在每次执行CreatePredicate之后会累减nElmPerRepeat，直至0
                    preg = CreatePredicate<T>(sreg);

                    // 相邻两行相加放入temp, nLoop为tmp有效数据行数
                    nLoop = validRow / 2;
                    remain = validRow % 2;
                    for (j = 0; j < nLoop; ++j) {
                        vlds(src0VReg, srcP, (2 * j) * srcRowStride, NORM);
                        vlds(src1VReg, srcP, (2 * j + 1) * srcRowStride, NORM);
                        vadd(tmpVReg, src0VReg, src1VReg, preg, MODE_ZEROING);
                        vsts(tmpVReg, tmpP, j * tmpRowStride, distValue, preg);
                    }

                    if (remain) {
                        // 最后剩余奇数行加入tmp最后一行
                        vlds(src0VReg, srcP, (validRow - 1) * srcRowStride, NORM);
                        vlds(src1VReg, tmpP, (nLoop - 1) * tmpRowStride, NORM);
                        vadd(tmpVReg, src0VReg, src1VReg, preg, MODE_ZEROING);
                        vsts(tmpVReg, tmpP, (nLoop - 1) * tmpRowStride, distValue, preg);
                    }

                    // 获取nLoop的 最高比特位-1 为循环次数, for(BinaryAccLoopTimes)等价于while(nLoop > 1)
                    BinaryAccLoopTimes = nLoop > 0 ? 63 - __builtin_clzll(nLoop) : 0;
                    for (j = 0; j < BinaryAccLoopTimes; ++j) {
                        remain = nLoop % 2;
                        nLoop = nLoop / 2;
                        for (k = 0; k < nLoop; ++k) {
                            vlds(src0VReg, tmpP, (2 * k) * tmpRowStride, NORM);
                            vlds(src1VReg, tmpP, (2 * k + 1) * tmpRowStride, NORM);
                            vadd(tmpVReg, src0VReg, src1VReg, preg, MODE_ZEROING);
                            vsts(tmpVReg, tmpP, k * tmpRowStride, distValue, preg);
                        }

                        if (remain) {
                            vlds(src0VReg, tmpP, (nLoop - 1) * tmpRowStride, NORM);
                            vlds(src1VReg, tmpP, (2 * nLoop) * tmpRowStride, NORM);
                            vadd(tmpVReg, src0VReg, src1VReg, preg, MODE_ZEROING);
                            vsts(tmpVReg, tmpP, (nLoop - 1) * tmpRowStride, distValue, preg);
                        }
                    }

                    // 最后一步vsts(tmpVReg, tmp)其实无作用, tmpVReg已经保存最终结果
                    vsts(tmpVReg, dstP, nElmPerRepeat, distValue, preg, POST_UPDATE);   // dstP每次累加nElmPerRepeat
                    srcP += nElmPerRepeat;
                    tmpP += nElmPerRepeat;
                }
            } // end VF
        } else {
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
                        vadd(tmpVReg, src0VReg, src1VReg, preg, MODE_ZEROING);
                        vadd(dstVReg, dstVReg, tmpVReg, preg, MODE_ZEROING);
                    }
                    if (remain) {
                        vlds(src0VReg, srcP, (validRow - 1) * srcRowStride, NORM);
                        vadd(dstVReg, dstVReg, src0VReg, preg, MODE_ZEROING);
                    }
                    vsts(dstVReg, dstP, nElmPerRepeat, distValue, preg, POST_UPDATE);   // dstP每次累加nElmPerRepeat
                    srcP += nElmPerRepeat;
                }
            } // end VF
        }
    }

    template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
    __aicore__ PTO_INLINE void TCOLSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary) {
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

        if (isBinary) {
            TColSum<T, TileDataOut, TileDataIn, TileDataTmp, true>(dst.data(), src.data(), tmp.data(),
                validRow, validCol);
        } else {
            TColSum<T, TileDataOut, TileDataIn, TileDataTmp, false>(dst.data(), src.data(), tmp.data(),
                validRow, validCol);
        }
    }
}
#endif
