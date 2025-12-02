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

#include "constants.hpp"

namespace pto {
    template <typename TileData, typename Func>
    __tf__ __aicore__ void TUnaryOp(typename TileData::TileDType __out__ dst,
                                    typename TileData::TileDType __in__ src,
                                    unsigned validCol,
                                    Func currentFunc) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);

        unsigned TShape0 = TileData::Rows;
        unsigned TShape1 = TileData::Cols;

        set_mask_count();
        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < TShape0; ++i) {
            currentFunc((dstPtr + i * TShape1), (srcPtr + i * TShape1), 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    /* RSQRT */

    template <typename TileData>
    __tf__ __aicore__ void TRsqrtCustom(typename TileData::TileDType __out__ dst,
                                        typename TileData::TileDType __in__ src,
                                        unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *srcPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);

        unsigned TShape0 = TileData::Rows;
        unsigned TShape1 = TileData::Cols;

        set_mask_count();
        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < TShape0; ++i) {
            vsqrt((srcPtr + i * TShape1), (srcPtr + i * TShape1), 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);

        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < TShape0; ++i) {
            vector_dup((dstPtr + i * TShape1), (typename TileData::DType)(1.0), 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);

        set_vector_mask(0, validCol);
        for (uint32_t i = 0; i < TShape0; ++i) {
            vdiv((dstPtr + i * TShape1), (dstPtr + i * TShape1), (srcPtr + i * TShape1), 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    template<typename DataType>
    __aicore__ void _vrsqrt(__ubuf__ DataType* dst, __ubuf__ DataType* src,
                            uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                            uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vrsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

#define ACCURATE_RSQRT
    template <typename TileData>
    __aicore__ void TRSQRT_IMPL(TileData &dst, TileData &src) {
        unsigned validCol = dst.GetValidCol();
#ifdef ACCURATE_RSQRT
        TRsqrtCustom<TileData>(dst.data(), src.data(), validCol);
#else
        auto funcPtr = _vrsqrt<typename TileData::DType>;
        TUnaryOp<TileData, decltype(funcPtr)>(dst.data(), src.data(), validCol, funcPtr);
#endif
    }

    /* SQRT */

    template<typename DataType>
    __aicore__ void _vsqrt(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                            uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                            uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vsqrt(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    __aicore__ void TSQRT_IMPL(TileData &dst, TileData &src) {
        unsigned validCol = dst.GetValidCol();
        auto funcPtr = _vsqrt<typename TileData::DType>;
        TUnaryOp<TileData, decltype(funcPtr)>(dst.data(), src.data(), validCol, funcPtr);
    }

    /* EXP */

    template<typename DataType>
    __aicore__ void _vexp(__ubuf__ DataType* dst, __ubuf__ DataType* src, 
                          uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                          uint8_t dstRepeatStride, uint8_t srcRepeatStride) {
        vexp(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename TileData>
    __aicore__ void TEXP_IMPL(TileData &dst, TileData &src) {
        unsigned validCol = dst.GetValidCol();
        auto funcPtr = _vexp<typename TileData::DType>;
        TUnaryOp<TileData, decltype(funcPtr)>(dst.data(), src.data(), validCol, funcPtr);
    }
}

#endif