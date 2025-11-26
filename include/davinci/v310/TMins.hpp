#ifndef TMINS_HPP
#define TMINS_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {
template <typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ __aicore__ PTO_INLINE void TMinsImpl(
    typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0,
    ScalarType scalar,
    unsigned validRow
) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg2;
        MaskReg preg;
        uint32_t sreg = (uint32_t)(validRow * TileData::Cols);
        uint16_t repeatTimes = CeilDivision(validRow * TileData::Cols, elementsPerRepeat);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
            vmins(vreg2, vreg0, scalar, preg, MODE_ZEROING);
            vsts(vreg2, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
        }
    } // end of VF
}

template <typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ __aicore__ PTO_INLINE void TMinsPadImpl(
    typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0,
    ScalarType scalar,
    unsigned validRow,
    unsigned validCol
) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg2;
        MaskReg preg;
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr + i * TileData::Cols, j * elementsPerRepeat, NORM);
                vmins(vreg2, vreg0, scalar, preg, MODE_ZEROING);
                vsts(vreg2, dstPtr + i * TileData::Cols, j * elementsPerRepeat, distValue, preg);
            }
        }
    } // end VF
}

template <typename TileData>
__aicore__ void TMINS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    using T = typename TileData::DType;
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TMINS: Invalid data type.");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    if constexpr (TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero) {
        TMinsImpl<TileData, T, elementsPerRepeat, blockSizeElem>(dst.data(), src0.data(), scalar, validRow);
    } else { // -INF(MIN) or INF(MAX)
        TMinsPadImpl<TileData, T, elementsPerRepeat, blockSizeElem>(dst.data(), src0.data(), scalar, validRow, validCol);
    }
}
}  // namespace pto
#endif
