#ifndef PTO_UTILS_H
#define PTO_UTILS_H

#include <common/type.hpp>

namespace pto{
    const uint32_t VECTOR_REG_WIDTH = 256;
    const uint32_t VECTOR_REG_WIDTH_2XVL = 512;

    enum class DistVST {
        DIST_NORM_B8,
        DIST_NORM_B16,
        DIST_NORM_B32,
        DIST_ONEPT_B8,
        DIST_ONEPT_B16,
        DIST_ONEPT_B32,
        DIST_PK_B16,
        DIST_PK_B32,
        DIST_INTLV_B8,
        DIST_INTLV_B16,
        DIST_PK_B64,
        DIST_INTLV_B32,
        DIST_PK4_B32,
        DIST_MRG4CHN_B8,
        DIST_MRG2CHN_B8,
        DIST_MRG2CHN_B16,
        DIST_NORM
    };
    
    template <typename T, DistVST dist> __aicore__ PTO_INLINE constexpr DistVST GetDistVst()
    {
        if constexpr (dist == DistVST::DIST_NORM) {
            static_assert(SupportBytes<T, 1, 2, 4>(), "DistVST DIST_NORM only support type b8/b16/b32 on current device");
            if constexpr (sizeof(T) == 1) {
                return DistVST::DIST_NORM_B8;
            } else if constexpr (sizeof(T) == 2) {
                return DistVST::DIST_NORM_B16;
            } else if constexpr (sizeof(T) == 4) {
                return DistVST::DIST_NORM_B32;
            }
        }
        return dist;
    }
} // end pto

#endif
