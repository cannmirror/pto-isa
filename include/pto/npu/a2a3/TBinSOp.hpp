/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINS_HPP
#define TBINS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
namespace pto
{
    #define SMALL_RPT (4)
    template <typename Op, typename T>
    PTO_INTERNAL void BinS1LCountMode(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        set_mask_count();
        SetVectorCount(validRow * validCol);
        Op::BinSInstr(dst, src0, src1, 0);
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }
    template <typename Op, typename T, unsigned rowStride>
    PTO_INTERNAL void BinS2LCountMode(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        set_mask_count();
        SetVectorCount(validCol);
        for (unsigned i = 0; i < validRow; i++) {
            unsigned offset = i * rowStride; 
            Op::BinSInstr(dst + offset, src0 + offset, src1, 0);
        }
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }
    template <typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride, unsigned Cols>
    PTO_INTERNAL void BinS1LNormMode(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        unsigned numElements = validRow * validCol;
        unsigned headRepeats = numElements / elementsPerRepeat;
        unsigned tailElements = numElements % elementsPerRepeat;
        Op::BinSInstr(dst, src0, src1, headRepeats);
        if (tailElements) [[unlikely]] {
            unsigned offset = headRepeats * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::BinSInstr(dst + offset, src0 + offset, src1, 1);
            SetFullVecMaskByDType<T>();
        }
    }
    template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
    PTO_INTERNAL void BinS2LNormModeColVLAlign(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        unsigned headRepeats = validCol / elementsPerRepeat;
        for (uint32_t i = 0; i < validRow; i++) {
            unsigned offset = i * rowStride;
            Op::BinSInstr(dst + offset, src0 + offset, src1, headRepeats);
        }
    }
    template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem , unsigned stride>
    PTO_INTERNAL void BinS2LNormModeHead(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned numRepeatPerLine) {
        if (numRepeatPerLine > 0) {
                unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
                unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
                for (int i = 0; i < validRow; i++) {
                    if (numLoop) [[unlikely]] {
                        for (int j = 0; j < numLoop; j++) {
                            unsigned offset = i * stride + j * elementsPerRepeat * REPEAT_MAX;
                            Op::BinSInstr(dst + offset, src0 + offset, src1, REPEAT_MAX);
                        }
                    }
                    if (remainAfterLoop) {
                        unsigned offset = i * stride + numLoop * elementsPerRepeat * REPEAT_MAX;
                        Op::BinSInstr(dst + offset, src0 + offset, src1, remainAfterLoop);
                    }   
                }
            }
    }
    
    template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem , unsigned stride>
    PTO_INTERNAL void BinS2LNormModeTail(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned numRemainPerLine) {
            unsigned numLoop = 0;
            unsigned remainAfterLoop = validRow;
            const bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContMaskByDType<T>(numRemainPerLine);
            if constexpr (Rows > pto::REPEAT_MAX) {
                numLoop = validRow / REPEAT_MAX;
                for (int i = 0; i < numLoop; i++) {
                    if constexpr (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            unsigned offset = i * REPEAT_MAX * stride + j * stride;
                            Op::BinSInstr(dst + offset, src0 + offset, src1, 1, 1, 1);
                        }
                    } else {
                        unsigned offset = i * REPEAT_MAX * stride;
                        uint8_t repeatStride = stride / blockSizeElem;
                        Op::BinSInstr(dst + offset, src0 + offset, src1, REPEAT_MAX, repeatStride, repeatStride);
                    }
                }
                remainAfterLoop = validRow % REPEAT_MAX;
            }
            
            if (remainAfterLoop) {
                if constexpr (strideOverFlag) {
                    for (unsigned j = 0; j < remainAfterLoop; j++) {
                        unsigned offset = numLoop * REPEAT_MAX * stride + j * stride;
                        Op::BinSInstr(dst + offset, src0 + offset, src1, 1, 1, 1);
                    }
                } else {
                    unsigned offset = numLoop * REPEAT_MAX * stride;
                    uint8_t repeatStride = stride / blockSizeElem;
                    Op::BinSInstr(dst + offset, src0 + offset, src1, remainAfterLoop, repeatStride, repeatStride);
                }
            }
            SetFullVecMaskByDType<T>();
    }

    template <typename Op, typename T, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem , unsigned stride>
    PTO_INTERNAL void BinS2LNormModeRowRpt(__ubuf__ T* dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        constexpr unsigned repeatStride = stride / blockSizeElem;
        constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && repeatStride <= (REPEAT_STRIDE_MAX ));
        if constexpr (condRowRpt) {
            unsigned numLoop = validCol / elementsPerRepeat;
            unsigned tailElements = validCol % elementsPerRepeat;
            for (unsigned i = 0; i < numLoop; i++) {
                unsigned offset = i * elementsPerRepeat;
                Op::BinSInstr(dst + offset, src0 + offset, src1, validRow, repeatStride, repeatStride);
            }

            if (tailElements) {
                unsigned offset = numLoop * elementsPerRepeat;
                SetContMaskByDType<T>(tailElements);
                Op::BinSInstr(dst + offset, src0 + offset, src1, validRow, repeatStride, repeatStride);
                SetFullVecMaskByDType<T>();
            }
        } else {
            unsigned numRemainPerLine = validCol;
            if constexpr (Rows > elementsPerRepeat) {
                unsigned numRepeatPerLine = validCol / elementsPerRepeat;
                numRemainPerLine = validCol % elementsPerRepeat;
                BinS2LNormModeHead<Op, T, Rows, elementsPerRepeat, blockSizeElem, stride>(dst, src0, src1, validRow, numRepeatPerLine);
                unsigned offset = numRepeatPerLine * elementsPerRepeat;
                dst += offset; 
                src0 += offset; 
            }
            if (numRemainPerLine) {
                BinS2LNormModeTail<Op, T, Rows, elementsPerRepeat, blockSizeElem, stride>(dst, src0, src1, validRow, numRemainPerLine);
            }
        }
    }
    template <typename Op,typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    PTO_INTERNAL void TBinSInstr(__ubuf__ typename TileData::DType __out__ *dst,
                                  __ubuf__ typename TileData::DType __in__ *src0,
                                  typename TileData::DType __in__ src1,
                                  unsigned validRow,
                                  unsigned validCol)
    {
        using T = typename TileData::DType;
        if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
            constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
            constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
            if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
                BinS1LCountMode<Op, T>(dst, src0, src1, validRow, validCol);
            } else {
                BinS1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(dst, src0, src1, validRow, TileData::Cols);
            }
        } else {
            if ((TileData::Cols == validCol) || (validRow == 1)) [[likely]] {
                unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
                bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
                if (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) [[unlikely]] {
                    BinS1LCountMode<Op, T>(dst, src0, src1, validRow, validCol);
                } else {
                    BinS1LNormMode<Op, T, elementsPerRepeat, blockSizeElem, rowStride, TileData::Cols>(dst, src0, src1, validRow, validCol);
                }
            } else {
                constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
                if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT)) {
                    BinS2LCountMode<Op, T, rowStride>(dst, src0, src1, validRow, validCol);
                } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
                    unsigned tailElements = validCol % elementsPerRepeat;
                    if (tailElements) {
                        BinS2LCountMode<Op, T, rowStride>(dst, src0, src1, validRow, validCol);    
                    } else {
                        BinS2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(dst, src0, src1, validRow, validCol);
                    }
                } else {
                    BinS2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(dst, src0, src1, validRow, validCol);
                }
            }
        }
    }
} //namespace pto
#endif
