#ifndef TADDS_HPP
#define TADDS_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common.hpp"
#include "utils.hpp"


namespace pto
{
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ __aicore__ PTO_INLINE void TAddS(typename TileData::TileDType __out__ dst,
                                            typename TileData::TileDType __in__ src0, typename TileData::DType __in__ src1, unsigned validRow,
                                            unsigned validCol)
    {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint16_t> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>)
        {
            if constexpr (TileData::PadVal == PadValue::Zero)
            {
                __VEC_SCOPE__
                {
                    MaskReg preg;
                    RegTensor<T> vregsrc, vregdst;
                    uint32_t sreg = (uint32_t)(validRow * TileData::Cols);
                    uint16_t repeatTimes = CeilDivision(validRow * TileData::Cols, elementsPerRepeat);
                    constexpr auto distValue =
                        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i)
                    {
                        preg = CreatePredicate<T>(sreg);
                        vlds(vregsrc, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                        vadds(vregdst, vregsrc, src1, preg);
                        vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                    }
                } // end of VF
            }
            else
            { // -INF(MIN) or INF(MAX)
                __VEC_SCOPE__
                {
                    MaskReg preg;
                    RegTensor<T> vregdst, vregsrc;
                    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
                    constexpr auto distValue =
                        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
                    for (uint16_t i = 0; i < (uint16_t)(validRow); ++i)
                    {
                        uint32_t sreg = (uint32_t)(validCol);
                        for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j)
                        {
                            preg = CreatePredicate<T>(sreg);
                            vlds(vregsrc, src0Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                            vadds(vregdst, vregsrc, src1, preg);
                            vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                        }
                    }
                } // end VF
            }
        }
        else
        {
            static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TADD: Invalid data type.");
        }
    }
    template <typename TileData>
    __aicore__ void TADDS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned rowStride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TAddS<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), scalar, validRow, validCol);
    }
} // namespace pto
#endif
