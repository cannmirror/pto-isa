/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

template <typename T>
// Tensor-tensor remainder implemented via vdiv, vmul and vsub for floating types.
struct RemOp {
    PTO_INTERNAL static void REMF32Instr(
        __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ T *tmpPtr, __ubuf__ T *prod) {
        // qf = s0 / s1
        vdiv(tmpPtr, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // float32 path: convert float quotient -> int32 (truncate), then back to float
        // Convert float -> int32 with truncation
        __ubuf__ int32_t *qi = reinterpret_cast<__ubuf__ int32_t *>(tmpPtr);
        vconv_f322s32z(qi, tmpPtr, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // convert int32 back to float
        vconv_s322f32(tmpPtr, qi, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // prod = qf * s1
        vmul(prod, tmpPtr, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // dst = s0 - prod
        vsub(dst, src0, prod, 1, 1, 1, 1, 8, 8, 8);
    }

    PTO_INTERNAL static void REMF16Instr(
        __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ T *tmpPtr, __ubuf__ T *prod) {
        // qf = s0 / s1
        vdiv(tmpPtr, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // half path: convert half quotient -> int32 (via f16->s32) then back
        __ubuf__ int32_t *qi = reinterpret_cast<__ubuf__ int32_t *>(tmpPtr);
        // qf currently half; convert half->int32 trunc
        vconv_f162s32z(qi, tmpPtr, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // convert int32 -> float32 into reinterpret tmp area (as float)
        __ubuf__ float *qf32 = reinterpret_cast<__ubuf__ float *>(tmpPtr);
        vconv_s322f32(qf32, qi, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // convert float32 -> half into qf
        vconv_f322f16(tmpPtr, qf32, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // prod = qf * s1
        vmul(prod, tmpPtr, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // dst = s0 - prod
        vsub(dst, src0, prod, 1, 1, 1, 1, 8, 8, 8);
    }

    PTO_INTERNAL static void REMInt32Instr(
        __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ T *tmpPtr, unsigned validCols) {
        // SIMD实现：vcvt, vdiv, vmul, vsub
        // 假设tmpPtr和prod已分配足够空间，类型为float
        // 1. int->float
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        __ubuf__ float *src0_f = reinterpret_cast<__ubuf__ float *>(tmpPtr);
        __ubuf__ float *src1_f = src0_f + validCols;
        __ubuf__ float *qf = src1_f + validCols;
        __ubuf__ float *prod = qf + validCols;
        // int->float
        vconv_s322f32(src0_f, (__ubuf__ int32_t *)src0, 1, 1, 1, 8, 8);
        vconv_s322f32(src1_f, (__ubuf__ int32_t *)src1, 1, 1, 1, 8, 8);
        // 2. qf = src0_f / src1_f
        vdiv(qf, src0_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        // 3. qf_int = trunc(qf)
        __ubuf__ int32_t *qf_int = reinterpret_cast<__ubuf__ int32_t *>(prod + validCols); // reuse prod buffer
        vconv_f322s32z(qf_int, qf, 1, 1, 1, 8, 8);
        // 4. qf_trunc_f = float(qf_int)
        __ubuf__ float *qf_trunc_f = reinterpret_cast<__ubuf__ float *>(qf_int + validCols);
        vconv_s322f32(qf_trunc_f, qf_int, 1, 1, 1, 8, 8);
        // 5. prod = qf_trunc_f * src1_f
        vmul(prod, qf_trunc_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        // 6. rem_f = src0_f - prod
        __ubuf__ float *rem_f = prod + validCols;
        vsub(rem_f, src0_f, prod, 1, 1, 1, 1, 8, 8, 8);
        // 7. float->int
        // if constexpr (std::is_same_v<T, int32_t>) {
        vconv_f322s32z((__ubuf__ int32_t *)dst, rem_f, 1, 1, 1, 8, 8);
    }

    PTO_INTERNAL static void REMInt16Instr(
        __ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, __ubuf__ T *tmpPtr, unsigned validCols) {
        // SIMD实现：vconv, vdiv, vmul, vsub for int16
        // 1. int16 -> float32
        __ubuf__ half *src0_f = reinterpret_cast<__ubuf__ half *>(tmpPtr);
        __ubuf__ half *src1_f = src0_f + validCols;
        __ubuf__ half *qf = src1_f + validCols;
        __ubuf__ half *prod = qf + validCols;
        vconv_s162f16(src0_f, (__ubuf__ int16_t *)src0, 1, 1, 1, 8, 8);
        vconv_s162f16(src1_f, (__ubuf__ int16_t *)src1, 1, 1, 1, 8, 8);
        // 2. qf = src0_f / src1_f
        vdiv(qf, src0_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        // 3. qf_int = trunc(qf)
        __ubuf__ int16_t *qf_int = reinterpret_cast<__ubuf__ int16_t *>(prod + validCols); // reuse prod buffer
        vconv_f162s16z(qf_int, qf, 1, 1, 1, 8, 8);
        // 4. qf_trunc_f = float(qf_int)
        __ubuf__ half *qf_trunc_f = reinterpret_cast<__ubuf__ half *>(qf_int + validCols);
        vconv_s162f16(qf_trunc_f, qf_int, 1, 1, 1, 8, 8);
        // 5. prod = qf_trunc_f * src1_f
        vmul(prod, qf_trunc_f, src1_f, 1, 1, 1, 1, 8, 8, 8);
        // 6. rem_f = src0_f - prod
        __ubuf__ half *rem_f = prod + validCols;
        vsub(rem_f, src0_f, prod, 1, 1, 1, 1, 8, 8, 8);
        // 7. float->int16
        vconv_f162s16z((__ubuf__ int16_t *)dst, rem_f, 1, 1, 1, 8, 8);
    }
};

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
    unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL void TRem(typename TileData::TileDType __out__ dst, typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1, unsigned validRows, unsigned validCols) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET); // tmp buffer
    __ubuf__ T *prod = tmpPtr + validCols;              // product buffer

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < validRows; ++i) {
        unsigned colsRemaining = validCols;
        __ubuf__ T *dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T *s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T *s1Next = src1Ptr + i * src1RowStride;
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemOp<T>::REMF32Instr(dstNext, s0Next, s1Next, tmpPtr, prod);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float16_t>) {
            RemOp<T>::REMF16Instr(dstNext, s0Next, s1Next, tmpPtr, prod);
        } else if constexpr (std::is_integral_v<T> || sizeof(T) == 4) {
            RemOp<T>::REMInt32Instr(dstNext, s0Next, s1Next, tmpPtr, validCols);
        } else if constexpr (std::is_integral_v<T> && sizeof(T) == 2) {
            RemOp<T>::REMInt16Instr(dstNext, s0Next, s1Next, tmpPtr, validCols);
        } else {
            static_assert(sizeof(T) == 4 || sizeof(T) == 2, "Fix: TREM has unsupported dtype size");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TRemCheck(const TileDataDst &dst, const TileDataSrc0 &src0, const TileDataSrc1 &src1) {
    static_assert(std::is_same<T, half>::value || std::is_same<T, float>::value || std::is_same<T, float32_t>::value ||
                      std::is_same<T, int32_t>::value,
        "Fix: TREM currently supports half/float and 16/32-bit integer data types.");
    static_assert(TileDataDst::isRowMajor || TileDataSrc0::isRowMajor || TileDataSrc1::isRowMajor,
        "Fix: TREM only support row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TREM_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) {
    using T = typename TileDataDst::DType;
    TRemCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    TRem<TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}

} // namespace pto

#endif
