/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIVS_HPP
#define TDIVS_HPP

#include <pto/common/constants.hpp>
#include "TBinSOp.hpp"

namespace pto
{
    template <typename T>
    struct SDivOp {
        PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *src0, T src1, uint8_t repeats) {
            if constexpr (std::is_same<T, int32_t>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_s322f32(reinterpret_cast<__ubuf__ float *>(dst), dst, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_s322f32(reinterpret_cast<__ubuf__ float *>(src0), src0, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vdiv(reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(src0), repeats, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float *>(dst), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, int16_t>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_s162f16(reinterpret_cast<__ubuf__ half *>(dst), dst, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_s162f16(reinterpret_cast<__ubuf__ half *>(src0), src0, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vdiv(reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(src0), repeats, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f162s16z(dst, reinterpret_cast<__ubuf__ half *>(dst), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, float>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vdiv(dst, dst, src0, repeats, 1, 1, 1, 8, 8, 8);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vdiv(dst, dst, src0, repeats, 1, 1, 1, 8, 8, 8);
            }
        }
        PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
            if constexpr (std::is_same<T, int32_t>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_s322f32(reinterpret_cast<__ubuf__ float *>(dst), dst, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_s322f32(reinterpret_cast<__ubuf__ float *>(src0), src0, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vdiv(reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(src0), repeats, 1, 1, 1, dstRepeatStride, srcRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float *>(dst), repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, int16_t>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_s162f16(reinterpret_cast<__ubuf__ half *>(dst), dst, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_s162f16(reinterpret_cast<__ubuf__ half *>(src0), src0, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vdiv(reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(src0), repeats, 1, 1, 1, dstRepeatStride, srcRepeatStride, dstRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_f162s16z(dst, reinterpret_cast<__ubuf__ half *>(dst), repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, float>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vdiv(dst, dst, src0, repeats, 1, 1, 1, dstRepeatStride, srcRepeatStride, dstRepeatStride);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vdiv(dst, dst, src0, repeats, 1, 1, 1, dstRepeatStride, srcRepeatStride, dstRepeatStride);
            }
        }
    };
    
    template <typename T> 
    struct DivSOp {
        PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *src0, T src1, uint8_t repeats) {
            float divider = static_cast<float>(src1);
            if (divider != 0.0f)
            {
                divider = 1.0f / divider;
            }
            else
            {
                divider = 1.0 / 0.0;
            }
            if constexpr (std::is_same<T, int32_t>::value)
            {
                vconv_s322f32(reinterpret_cast<__ubuf__ float *>(dst), src0, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vmuls(reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(dst), divider, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float *>(dst), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, int16_t>::value)
            {
                vconv_s162f16(reinterpret_cast<__ubuf__ half *>(dst), src0, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vmuls(reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(dst), static_cast<half>(divider), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vconv_f162s16z(dst, reinterpret_cast<__ubuf__ half *>(dst), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vdiv(dst, src0, dst, repeats, 1, 1, 1, 8, 8, 8);
            }
            else
            {
                vmuls(dst, src0, divider, repeats, 1, 1, 8, 8);
            }
        }
        PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
            float divider = static_cast<float>(src1);
            if (divider != 0.0f)
            {
                divider = 1.0f / divider;
            }
            else
            {
                divider = 1.0 / 0.0;
            }
            if constexpr (std::is_same<T, int32_t>::value)
            {
                vconv_s322f32(reinterpret_cast<__ubuf__ float *>(dst), src0, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vmuls(reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(dst), divider, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float *>(dst), repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, int16_t>::value)
            {
                vconv_s162f16(reinterpret_cast<__ubuf__ half *>(dst), src0, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vmuls(reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(dst), static_cast<half>(divider), repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vconv_f162s16z(dst, reinterpret_cast<__ubuf__ half *>(dst), repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                vector_dup(dst, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
                pipe_barrier(PIPE_V);
                vdiv(dst, src0, dst, repeats, 1, 1, 1, dstRepeatStride, srcRepeatStride, dstRepeatStride);
            }
            else
            {
                vmuls(dst, src0, divider, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
            }
        }
    };
    template <typename T, unsigned Cols>
    PTO_INTERNAL void TDivs_naive(__ubuf__ T *dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int row = 0; row < validRow; row++) {
            for (int col = 0; col < validCol; col++) {
                int idx = row * Cols + col;
                dst[idx] = src0[idx] / src1;
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }

    template <typename T, unsigned Cols>
    PTO_INTERNAL void TSDiv_naive(__ubuf__ T *dst, __ubuf__ T* src0, T src1, unsigned validRow, unsigned validCol) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int row = 0; row < validRow; row++) {
            for (int col = 0; col < validCol; col++) {
                int idx = row * Cols + col;
                dst[idx] = src1 / src0[idx];
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }

    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ PTO_INTERNAL void TDivS(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::DType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);

        if constexpr(std::is_same<T, int16_t>::value) {
            TDivs_naive<T, TileData::Cols>(dst, src0, src1, validRow, validCol);
        }
        else {
            TBinSInstr<DivSOp<typename TileData::DType>, TileData, elementsPerRepeat, blockSizeElem, stride>(
                dstPtr, src0Ptr, src1, validRow, validCol);
        }
    }
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ PTO_INTERNAL void TSDiv(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::DType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        if constexpr(std::is_same<T, int16_t>::value) {
            TSDiv_naive<T, TileData::Cols>(dst, src0, src1, validRow, validCol);
        }
        else {
            TBinSInstr<SDivOp<typename TileData::DType>, TileData, elementsPerRepeat, blockSizeElem, stride>(
                dstPtr, src0Ptr, src1, validRow, validCol);
        }
    }
    template <typename TileData>
    AICORE void TDIVS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar) {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TDIVS: Invalid data type");

        static_assert(TileData::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "Number of valid rows must not be greater than number of tile rows.");
        
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TDivS<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), scalar, validRow, validCol);
    }
    
    template <typename TileData>
    PTO_INTERNAL void TDIVS_IMPL(TileData &dst, typename TileData::DType scalar, TileData &src0) {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TDIVS: Invalid data type");

        static_assert(TileData::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
        static_assert(TileData::ValidCol <= TileData::Cols, "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows, "Number of valid rows must not be greater than number of tile rows.");
        
        PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
        PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
        
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TSDiv<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), scalar, validRow, validCol);
    }
}

#endif