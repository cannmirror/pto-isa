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

namespace pto {

    #define SMALL_RPT (4)

    template <typename Op, typename T>
    PTO_INTERNAL void Unary1LCountMode(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol) {
        set_mask_count();
        SetVectorCount(validRow * validCol);
        Op::UnaryInstr(dstPtr, srcPtr, 0);
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }

    template <typename Op, typename T, unsigned rowStride>
    PTO_INTERNAL void Unary2LCountMode(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol) {
        set_mask_count();
        SetVectorCount(validCol);
        for (uint32_t i = 0; i < validRow; i++) {
            uint32_t offset = i * rowStride;
            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 0);
        }
        set_mask_norm();
        SetFullVecMaskByDType<T>();
    }

    template <typename Op, typename T, unsigned elementsPerRepeat>
    PTO_INTERNAL void Unary1LNormMode(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol) {
        unsigned numElements = validRow * validCol;
        unsigned headRepeats = numElements / elementsPerRepeat;
        unsigned tailElements = numElements % elementsPerRepeat;

        Op::UnaryInstr(dstPtr, srcPtr, headRepeats);
        if (tailElements) {
            unsigned offset = headRepeats * elementsPerRepeat;
            SetContMaskByDType<T>(tailElements);
            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 1);
            SetFullVecMaskByDType<T>();
        }
    }

     template <typename Op, typename T, unsigned elementsPerRepeat, unsigned rowStride>
    PTO_INTERNAL void Unary2LNormModeColVLAlign(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol) {
        unsigned headRepeats = validCol / elementsPerRepeat;
        for (uint32_t i = 0; i < validRow; i++) {
            uint32_t offset = i * rowStride;
            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, headRepeats);
        }
    }

    template <typename Op, typename T, unsigned rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    PTO_INTERNAL void Unary2LNormModeHead(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned numRepeatPerLine) {
        if (numRepeatPerLine) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (unsigned i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (unsigned j = 0; j < numLoop; j++) {
                        unsigned offset = i * rowStride + j * elementsPerRepeat * REPEAT_MAX;
                        Op::UnaryInstr(dstPtr + offset, srcPtr + offset, REPEAT_MAX);
                    }
                }
                if (remainAfterLoop) {
                    unsigned offset = i * rowStride + numLoop * elementsPerRepeat * REPEAT_MAX;
                    Op::UnaryInstr(dstPtr + offset, srcPtr + offset, remainAfterLoop);
                }
            }
        }
    }

    template <typename Op, typename T, unsigned rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    PTO_INTERNAL void Unary2LNormModeTail(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned numRemainPerLine) {
        unsigned numLoop = 0;
        unsigned remainAfterLoop = validRow;
        constexpr bool strideOverFlag = (rowStride / blockSizeElem > REPEAT_STRIDE_MAX);
        SetContMaskByDType<T>(numRemainPerLine);
        if constexpr (rows > pto::REPEAT_MAX) {
            numLoop = validRow / REPEAT_MAX;
            if (numLoop) {
                for (uint32_t i = 0; i < numLoop; i++) {
                    if constexpr (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            unsigned offset = i * REPEAT_MAX * rowStride + j * rowStride;
                            Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 1, 1, 1);
                        }
                    } else {
                        unsigned offset = i * REPEAT_MAX * rowStride;
                        uint8_t repeatStride = rowStride / blockSizeElem;
                        Op::UnaryInstr(dstPtr + offset, srcPtr + offset, REPEAT_MAX, repeatStride, repeatStride);
                    }
                }
            }
            remainAfterLoop = validRow % REPEAT_MAX;
        }
        if (remainAfterLoop) {
            if constexpr (strideOverFlag) {
                for (uint32_t j = 0; j < remainAfterLoop; j++) {
                    unsigned offset = numLoop * REPEAT_MAX * rowStride + j * rowStride;
                    Op::UnaryInstr(dstPtr + offset, srcPtr + offset, 1, 1, 1);
                }
            } else {
                unsigned offset = numLoop * REPEAT_MAX * rowStride;
                uint8_t repeatStride = rowStride / blockSizeElem;
                Op::UnaryInstr(dstPtr + offset, srcPtr + offset, remainAfterLoop, repeatStride, repeatStride);
            }
        }
        SetFullVecMaskByDType<T>();
    }

    template <typename Op, typename T, unsigned rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    PTO_INTERNAL void Unary2LNormModeRowRpt(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned validRow, unsigned validCol) {
        constexpr unsigned repeatStride = rowStride / blockSizeElem;
        constexpr bool condRowRpt = ((rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
        if constexpr (condRowRpt) {
            unsigned numLoop = validCol / elementsPerRepeat;
            unsigned tailElements = validCol % elementsPerRepeat;
            for (uint32_t i = 0; i < numLoop; i++) {
                unsigned offset = i * elementsPerRepeat;
                Op::UnaryInstr(dstPtr + offset, srcPtr + offset, validRow, repeatStride, repeatStride);
            }

            if (tailElements) {
                unsigned offset = numLoop * elementsPerRepeat;
                SetContMaskByDType<T>(tailElements);
                Op::UnaryInstr(dstPtr + offset, srcPtr + offset, validRow, repeatStride, repeatStride);
                SetFullVecMaskByDType<T>();
            }
        } else {
            unsigned numRemainPerLine = validCol;
            if constexpr (rows > elementsPerRepeat) {
                unsigned numRepeatPerLine = validCol / elementsPerRepeat;
                numRemainPerLine = validCol % elementsPerRepeat;
                Unary2LNormModeHead<Op, T, rows, elementsPerRepeat, blockSizeElem, rowStride>
                    (dstPtr, srcPtr, validRow, numRepeatPerLine);
                unsigned offset = numRepeatPerLine * elementsPerRepeat;
                dstPtr += offset;
                srcPtr += offset;
            }
            if (numRemainPerLine) {
                Unary2LNormModeTail<Op, T, rows, elementsPerRepeat, blockSizeElem, rowStride>
                    (dstPtr, srcPtr, validRow, numRemainPerLine);
            }
        }
    }

    template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    PTO_INTERNAL void UnaryInstr(__ubuf__ typename TileData::DType *dstPtr,
                                  __ubuf__ typename TileData::DType *srcPtr,
                                  unsigned validRow, unsigned validCol) {
        using T = typename TileData::DType;

        if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
            constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
            if constexpr (totalRepeats > pto::REPEAT_MAX) {
                Unary1LCountMode<Op, T>(dstPtr, srcPtr, validRow, validCol);
            } else {
                Unary1LNormMode<Op, T, elementsPerRepeat>(dstPtr, srcPtr, validRow, TileData::Cols);
            }
        } else {
            if ((TileData::Cols == validCol) || (validRow == 1)) {
                unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
                if (totalRepeats > pto::REPEAT_MAX) {
                    Unary1LCountMode<Op, T>(dstPtr, srcPtr, validRow, validCol);
                } else {
                    Unary1LNormMode<Op, T, elementsPerRepeat>(dstPtr, srcPtr, validRow, validCol);
                }
            } else {
                constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
                if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT)) {
                    Unary2LCountMode<Op, T, rowStride>(dstPtr, srcPtr, validRow, validCol);
                } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
                    unsigned tailElements = validCol % elementsPerRepeat;
                    if (tailElements) {
                        Unary2LCountMode<Op, T, rowStride>(dstPtr, srcPtr, validRow, validCol);
                    } else {
                        Unary2LNormModeColVLAlign<Op, T, elementsPerRepeat, rowStride>(dstPtr, srcPtr, validRow, validCol);
                    }
                } else {
                    Unary2LNormModeRowRpt<Op, T, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(
                        dstPtr, srcPtr, validRow, validCol);
                }
            }
        }
    }

    template <typename T> using unaryFuncPtr = void (*)(__ubuf__ T*, __ubuf__ T*, uint8_t, uint16_t, uint16_t, uint8_t, uint8_t);

    template <typename T, unaryFuncPtr<T> funcPtr> struct UnaryOperation {
        PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeats) {
            funcPtr(dst, src, repeats, 1, 1, 8, 8);
        }
        PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeats,
                                             uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
            funcPtr(dst, src, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
        }
    };

    template <typename TileData, unaryFuncPtr<typename TileData::DType> funcPtr, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ AICORE void TUnaryOp(typename TileData::TileDType __out__ dst,
                                    typename TileData::TileDType __in__ src,
                                    unsigned validRow,
                                    unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);

        UnaryInstr<UnaryOperation<typename TileData::DType, funcPtr>, TileData, elementsPerRepeat, blockSizeElem, rowStride>(dstPtr, srcPtr, validRow, validCol);
    }

    /* RSQRT */

    template <typename TileData>
    __tf__ AICORE void TRsqrtCustom(typename TileData::TileDType __out__ dst,
                                        typename TileData::TileDType __in__ src,
                                        unsigned validRow,
                                        unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);

        unsigned TShape0 = TileData::Rows;
        unsigned TShape1 = TileData::Cols;

        __ubuf__ typename TileData::DType *ones = reinterpret_cast<__ubuf__ typename TileData::DType*>(static_cast<std::uintptr_t>(0x2fc00));
        vector_dup(ones, (typename TileData::DType)(1.0), 1, 1, 1, 8, 8);

        set_mask_count();
        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < validRow; ++i) {
            vsqrt((dstPtr + i * TShape1), (srcPtr + i * TShape1), 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);

        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < validRow; ++i) {
            vdiv((dstPtr + i * TShape1), (ones), (dstPtr + i * TShape1), 1, 1, 1, 1, 8, 0, 8); 
        }
        pipe_barrier(PIPE_V);

        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template<typename DataType>
    AICORE void _vrsqrt(__ubuf__ DataType* dst, __ubuf__ DataType* src,
                            uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                            uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vrsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TRSQRT_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TRSQRT: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TRSQRT: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TRSQRT: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TRSQRT: Number of valid rows must not be greater than number of tile rows.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TRSQRT: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TRSQRT: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;

#ifdef ACCURATE_RSQRT
        TRsqrtCustom<TileData>(dst.data(), src.data(), validRow, validCol);
#else
        TUnaryOp<TileData, _vrsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
#endif
    }

    /* SQRT */

    template<typename DataType>
    AICORE void _vsqrt(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                            uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                            uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TSQRT_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TSQRT: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TSQRT: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TSQRT: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TSQRT: Number of valid rows must not be greater than number of tile rows.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TSQRT: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TSQRT: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vsqrt, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* EXP */

    template<typename DataType>
    AICORE void _vexp(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vexp(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TEXP_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TEXP: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TEXP: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TEXP: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TEXP: Number of valid rows must not be greater than number of tile rows.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TEXP: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TEXP: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vexp, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* ABS */

    template<typename DataType>
    AICORE void _vabs(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vabs(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TABS_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TABS: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TABS: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TABS: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TABS: Number of valid rows must not be greater than number of tile rows.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TABS: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TABS: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vabs, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* LOG */

    template<typename DataType>
    AICORE void _vlog(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vln(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TLOG_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TLOG: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TLOG: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TLOG: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TLOG: Number of valid rows must not be greater than number of tile rows.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TLOG: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TLOG: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vlog, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }

    /* RECIP */

    template<typename DataType>
    AICORE void _vrecip(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vrec(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    AICORE void TRECIP_IMPL(TileData &dst, TileData &src) {
        static_assert(std::is_same<typename TileData::DType, float32_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value,
                      "TRECIP: Invalid data type");
        static_assert(TileData::Loc == TileType::Vec, "TRECIP: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "TRECIP: Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "TRECIP: Number of valid rows must not be greater than number of tile rows.");

        PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TRECIP: Number of columns of src and dst must be the same.");
        PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TRECIP: Number of rows of src and dst must be the same.");

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        TUnaryOp<TileData, _vrecip, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src.data(), validRow, validCol);
    }
}

#endif