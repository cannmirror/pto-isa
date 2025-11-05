#ifndef TADD_HPP
#define TADD_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {
template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
__tf__ __aicore__ PTO_INLINE void TAdd(typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0, typename TileData::TileDType __in__ src1, unsigned validRow,
    unsigned validCol)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint16_t> ||
                  std::is_same_v<T, int16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> ||
                  std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
        if constexpr (TileData::PadVal == PadValue::Zero) {
            __VEC_SCOPE__
            {
                MaskReg preg;
                RegTensor<T> vreg0, vreg1, vreg2;                
                uint32_t sreg = (uint32_t)(validRow * TileData::Cols);
                uint16_t repeatTimes = CeilDivision(validRow * TileData::Cols, elementsPerRepeat);
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                    preg = CreatePredicate<T>(sreg);
                    vlds(vreg0, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    vlds(vreg1, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                    vsts(vreg2, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                }
            }     // end of VF
        } else {  // -INF(MIN) or INF(MAX)
            __VEC_SCOPE__
            {                
                MaskReg preg;
                RegTensor<T> vreg0, vreg1, vreg2;                   
                uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
                constexpr auto distValue =
                    std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
                    uint32_t sreg = (uint32_t)(validCol);
                    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vreg0, src0Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                        vlds(vreg1, src1Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                        vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
                        vsts(vreg2, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                    }
                }
            }  // end VF
        }
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TADD: Invalid data type.");
    }
}

template <typename TileData>
__aicore__ void TADD_IMPL(TileData &dst, TileData &src0, TileData &src1)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    TAdd<TileData, elementsPerRepeat, blockSizeElem, rowStride>(
        dst.data(), src0.data(), src1.data(), validRow, validCol);
}
}  // namespace pto
#endif
