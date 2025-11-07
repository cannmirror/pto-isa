#ifndef TCOLSUM_HPP
#define TCOLSUM_HPP

#include "common/constants.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace pto {
    template <typename T, typename TileDataOut, typename TileDataIn, unsigned nElmPerRepeat>
    __tf__ __aicore__ void TColSum(typename TileDataOut::TileDType __out__ dstData,
        typename TileDataIn::TileDType __in__ srcData, uint16_t validRow, int validCol, uint16_t repeatTimes) {
        
        __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
        __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
        __VEC_SCOPE__
        {
            RegTensor<T> srcVReg;
            RegTensor<T> dstVReg;
            uint32_t sreg = validCol;
            MaskReg preg;

            constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>
                (GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < repeatTimes; ++i) {
    
                // sreg在每次执行CreatePredicate之后会累减nElmPerRepeat，直至0
                preg = CreatePredicate<T>(sreg);

                // 将src的第一行存入dst寄存器
                vlds(dstVReg, src, i * nElmPerRepeat, NORM);

                // 读取第二行及以后的每行数据存入src寄存器，与dst寄存器相加后存入dst寄存器
                for (uint16_t j = 1; j < validRow; ++j) {
                    vlds(srcVReg, src + i * nElmPerRepeat, j * TileDataIn::Cols, NORM);
                    vadd(dstVReg, dstVReg, srcVReg, preg, MODE_ZEROING);
                }
                vsts(dstVReg, dst, i * nElmPerRepeat, distValue, preg);
            }
        } // end VF
    }

    template <typename TileDataOut, typename TileDataIn>
    __aicore__ void TCOLSUM(TileDataOut &dst, TileDataIn &src) {
        using T = typename TileDataIn::DType;
        constexpr bool isTargetType =
            std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int8_t> ||
            std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> ||
            std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, bfloat16_t>;
        static_assert(isTargetType, "The input data type is not supported by this instruction.");

        int validCol = src.GetValidCol();
        int validRow = src.GetValidRow();
        constexpr unsigned nElmPerRepeat = CCE_VL / sizeof(T);  // 每次repeat涉及多少个元素
        int repeatTimes = CeilDivision(validCol, nElmPerRepeat);

        TColSum<T, TileDataOut, TileDataIn, nElmPerRepeat>(dst.data(), src.data(), validRow, validCol, repeatTimes);
    }
}
#endif
