#ifndef COMMON_HPP
#define COMMON_HPP

#include "datatype.hpp"
#include <common/type.hpp>

namespace pto {
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

}

#endif