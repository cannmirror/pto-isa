/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_HPP
#define COMMON_HPP

#include "datatype.hpp"
#include <pto/common/type.hpp>

namespace pto {

    template <typename T>
    __aicore__ PTO_INLINE uint32_t GetByteSize(const uint32_t value) {
        if constexpr (std::is_same<T, float4_e1m2x2_t>::value || std::is_same<T, float4_e2m1x2_t>::value) {
            return value >> 1; // fp4 4bits
        }
        return sizeof(T) * value;
    }

    template <typename T, int U, int... Args> __aicore__ constexpr bool SupportBytes()
    {
        if constexpr (sizeof...(Args) > 0) {
            return sizeof(T) == U || SupportBytes<T, Args...>();
        }
        return sizeof(T) == U;
    }

    using MaskReg = vector_bool;
    using UnalignReg = vector_align;
    using AddrReg = vector_address;

    template <typename T> __aicore__ PTO_INLINE MaskReg CreatePredicateImpl(uint32_t &scalar)
    {
        MaskReg reg;
        if constexpr (sizeof(T) == 1) {
            reg = plt_b8(scalar, POST_UPDATE);
        } else if constexpr (sizeof(T) == 2) {
            reg = plt_b16(scalar, POST_UPDATE);
        } else if constexpr (sizeof(T) == 4) {
            reg = plt_b32(scalar, POST_UPDATE);
        }
        return reg;
    }

    template <typename T>
    __aicore__ PTO_INLINE MaskReg CreatePredicate(uint32_t &scalar)
    {
        return CreatePredicateImpl<T>(scalar);
    }

    template <typename T> struct RegTensor {
        __aicore__ PTO_INLINE RegTensor(){};
        using RegType = typename TypeGet<T>::T;
        RegType reg;

        __aicore__ PTO_INLINE operator RegType &()
        {
            return reg;
        }
        __aicore__ void Print() const;
    };

    template <typename SrcType, typename DstType>
    __aicore__ PTO_INLINE constexpr QuantMode_t GetCastPreQuantMode()
    {
        QuantMode_t quantPre = QuantMode_t::NoQuant;
        if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        }
        return quantPre;
    }

    template <typename SrcType, typename DstType>
    __aicore__ PTO_INLINE constexpr QuantMode_t GetScalarPreQuantMode()
    {
        QuantMode_t quantPre = QuantMode_t::NoQuant;
        if constexpr (std::is_same<SrcType, float>::value) {
            if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
                quantPre = QuantMode_t::QF322B8_PRE;
            } else if constexpr (std::is_same<DstType, hifloat8_t>::value) {
                quantPre = QuantMode_t::QF322HIF8_PRE;
            } else if constexpr (std::is_same<DstType, half>::value) {
                quantPre = QuantMode_t::QF322F16_PRE;
            } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
                quantPre = QuantMode_t::QF322BF16_PRE;
            }
        } else if constexpr (std::is_same<SrcType, int32_t>::value) {
            if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
                quantPre = QuantMode_t::REQ8;
            } else if constexpr (std::is_same<DstType, half>::value) {
                quantPre = QuantMode_t::DEQF16;
            } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
                quantPre = QuantMode_t::QS322BF16_PRE;
            }
        }
        return quantPre;
    }

    template <typename SrcType, typename DstType>
    __aicore__ PTO_INLINE constexpr QuantMode_t GetVectorPreQuantMode()
    {
        QuantMode_t quantPre = QuantMode_t::NoQuant;
        if constexpr (std::is_same<SrcType, float>::value) {
            if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
                quantPre = QuantMode_t::VQF322B8_PRE;
            } else if constexpr (std::is_same<DstType, hifloat8_t>::value) {
                quantPre = QuantMode_t::VQF322HIF8_PRE;
            } else if constexpr (std::is_same<DstType, half>::value) {
                quantPre = QuantMode_t::VQF322F16_PRE;
            } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
                quantPre = QuantMode_t::VQF322BF16_PRE;
            }
        } else if constexpr (std::is_same<SrcType, int32_t>::value) {
            if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
                quantPre = QuantMode_t::VREQ8;
            } else if constexpr (std::is_same<DstType, half>::value) {
                quantPre = QuantMode_t::VDEQF16;
            } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
                quantPre = QuantMode_t::VQS322BF16_PRE;
            }
        }
        return quantPre;
    }
}

#endif