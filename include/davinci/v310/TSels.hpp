#ifndef TSELS_HPP
#define TSELS_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto {
template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ __aicore__ PTO_INLINE void TSelsImpl(
    typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1,
    uint8_t selectMode,
    unsigned validRow
) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    __VEC_SCOPE__
    {
        MaskReg maskReg;
        MaskReg preg;
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        RegTensor<T> vreg2;
        if (selectMode == 1) {
            maskReg = pset_b8(PAT_ALL);
        } else {
            maskReg = pset_b8(PAT_ALLF);
        }
        uint32_t sreg = (uint32_t)(validRow * TileData::Cols);
        uint16_t repeatTimes = CeilDivision(validRow * TileData::Cols, elementsPerRepeat);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
            vlds(vreg1, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
            vsel(vreg2, vreg0, vreg1, maskReg);
            vsts(vreg2, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
        }
    } // end of VF
}

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem>
__tf__ __aicore__ PTO_INLINE void TSelsPadImpl(
    typename TileData::TileDType __out__ dst,
    typename TileData::TileDType __in__ src0,
    typename TileData::TileDType __in__ src1,
    uint8_t selectMode,
    unsigned validRow,
    unsigned validCol
) {
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    __VEC_SCOPE__
    {
        MaskReg maskReg;
        MaskReg preg;
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        RegTensor<T> vreg2;
        if (selectMode == 1) {
            maskReg = pset_b8(PAT_ALL);
        } else {
            maskReg = pset_b8(PAT_ALLF);
        }
        uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            uint32_t sreg = (uint32_t)(validCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr + i * TileData::Cols, j * elementsPerRepeat, NORM);
                vlds(vreg1, src1Ptr + i * TileData::Cols, j * elementsPerRepeat, NORM);
                vsel(vreg2, vreg0, vreg1, maskReg);
                vsts(vreg2, dstPtr + i * TileData::Cols, j * elementsPerRepeat, distValue, preg);
            }
        }
    } // end VF
}

template <typename TileData>
__aicore__ void TSELS_IMPL(TileData &dst, TileData &src0, TileData &src1, uint8_t selectMode)
{
    using T = typename TileData::DType;
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TSELS: Invalid data type.");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    if constexpr (TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero) {
        TSelsImpl<TileData, elementsPerRepeat, blockSizeElem>(dst.data(), src0.data(), src1.data(), selectMode, validRow);
    } else { // -INF(MIN) or INF(MAX)
        TSelsPadImpl<TileData, elementsPerRepeat, blockSizeElem>(dst.data(), src0.data(), src1.data(), selectMode, validRow, validCol);
    }
}
}  // namespace pto
#endif
