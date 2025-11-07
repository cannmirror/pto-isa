#ifndef __UIILS_HPP__
#define __UIILS_HPP__

#include "constants.hpp"
#pragma once

namespace pto {
    __aicore__ PTO_INLINE void SetContinuousMask(unsigned n) {
        set_vector_mask(static_cast<uint64_t>(
                            (n > MASK_LEN) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n - MASK_LEN)) - 1) : 0),
            static_cast<uint64_t>(
                (n >= MASK_LEN) ? 0xffffffffffffffff : (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n)) - 1)));
    }

    __aicore__ PTO_INLINE int32_t CeilDivision(int32_t num1, int32_t num2) {
        if (num2 == 0) {
            return 0;
        }
        return (num1 + num2 - 1) / num2;
    }
}


#endif