#ifndef TDIVS_HPP
#define TDIVS_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common.hpp"
#include "utils.hpp"

using namespace pto;
using namespace std;

namespace pto
{
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ __aicore__ PTO_INLINE void TDivS(typename TileData::TileDType __out__ dst,
                                            typename TileData::TileDType __in__ src0, typename TileData::DType __in__ src1, unsigned validRow,
                                            unsigned validCol)
    {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        float divider = static_cast<float>(src1);
        if (divider != 0)
        {
            divider = 1.0f / divider;
        }
        else
        {
            divider = 1.0 / 0.0;
        }
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
                        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
                        {
                            vlds(vregsrc, src0Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                            vmuls(vregdst, vregsrc, divider, preg);
                            vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                        }
                        else if constexpr (std::is_same<T, int32_t>::value)
                        {
                            vlds(vregsrc, src0Ptr, elementsPerRepeat, NORM);
                            RegTensor<float> tempDst;
                            vcvt(tempDst, vregsrc, preg, RoundRType());
                            vmuls(tempDst, tempDst, divider, preg);
                            vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                            vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg);
                        }
                        else if constexpr (std::is_same<T, int16_t>::value)
                        {
                            vlds(vregsrc, src0Ptr, elementsPerRepeat, NORM);
                            RegTensor<half> tempDst;
                            vcvt(tempDst, vregsrc, preg, RoundRType());
                            vmuls(tempDst, tempDst, divider, preg);
                            vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                            vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg);
                        }
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
                            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
                            {
                                vlds(vregsrc, src0Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                                vmuls(vregdst, vregsrc, divider, preg);
                                vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                            }
                            else if constexpr (std::is_same<T, int32_t>::value)
                            {
                                vlds(vregsrc, src0Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                                RegTensor<float> tempDst;
                                vcvt(tempDst, vregsrc, preg, RoundRType());
                                vmuls(tempDst, tempDst, divider, preg);
                                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                                vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                            }
                            else if constexpr (std::is_same<T, int16_t>::value)
                            {
                                vlds(vregsrc, src0Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                                RegTensor<half> tempDst;
                                vcvt(tempDst, vregsrc, preg, RoundRType());
                                vmuls(tempDst, tempDst, divider, preg);
                                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                                vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                            }
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
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
    __tf__ __aicore__ PTO_INLINE void TDivS(typename TileData::TileDType __out__ dst,
                                            typename TileData::DType __in__ src0, typename TileData::TileDType __in__ src1, unsigned validRow,
                                            unsigned validCol)
    {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
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
                        if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
                        {
                            vdup(vregdst, src0, preg, MODE_ZEROING);
                            vlds(vregsrc, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                            vdiv(vregdst, vregdst, vregsrc, preg);
                            vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                        }
                        else if constexpr (std::is_same<T, int32_t>::value)
                        {
                            RegTensor<float> tempDst;
                            RegTensor<float> tempSrc;
                            vdup(vregdst, src0, preg, MODE_ZEROING);
                            vlds(vregsrc, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                            vcvt(tempDst, vregdst, preg, RoundRType());
                            vcvt(tempSrc, vregsrc, preg, RoundRType());
                            vdiv(tempDst, tempDst, tempSrc, preg);
                            vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                            vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                        }
                        else if constexpr (std::is_same<T, int16_t>::value)
                        {
                            RegTensor<half> tempDst;
                            RegTensor<half> tempSrc;
                            vdup(vregdst, src0, preg, MODE_ZEROING);
                            vlds(vregsrc, src1Ptr, elementsPerRepeat, NORM, POST_UPDATE);
                            vcvt(tempDst, vregdst, preg, RoundRType());
                            vcvt(tempSrc, vregsrc, preg, RoundRType());
                            vdiv(tempDst, tempDst, tempSrc, preg);
                            vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                            vsts(vregdst, dstPtr, elementsPerRepeat, distValue, preg, POST_UPDATE);
                        }
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
                            if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
                            {
                                vdup(vregdst, src0, preg, MODE_ZEROING);
                                vlds(vregsrc, src1Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                                vdiv(vregdst, vregdst, vregsrc, preg);
                                vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                            }
                            else if constexpr (std::is_same<T, int32_t>::value)
                            {
                                RegTensor<float> tempDst;
                                RegTensor<float> tempSrc;
                                vdup(vregdst, src0, preg, MODE_ZEROING);
                                vlds(vregsrc, src1Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                                vcvt(tempDst, vregdst, preg, RoundRType());
                                vcvt(tempSrc, vregsrc, preg, RoundRType());
                                vdiv(tempDst, tempDst, tempSrc, preg);
                                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                                vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                            }
                            else if constexpr (std::is_same<T, int16_t>::value)
                            {
                                RegTensor<half> tempDst;
                                RegTensor<half> tempSrc;
                                vdup(vregdst, src0, preg, MODE_ZEROING);
                                vlds(vregsrc, src1Ptr + i * rowStride, j * elementsPerRepeat, NORM);
                                vcvt(tempDst, vregdst, preg, RoundRType());
                                vcvt(tempSrc, vregsrc, preg, RoundRType());
                                vdiv(tempDst, tempDst, tempSrc, preg);
                                vcvt(vregdst, tempDst, preg, RoundZType(), RS_ENABLE);
                                vsts(vregdst, dstPtr + i * rowStride, j * elementsPerRepeat, distValue, preg);
                            }
                        }
                        
                    }
                }
            } // end VF
        }
        else
        {
            static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TADD: Invalid data type.");
        }
} 
template <typename TileData>
__aicore__ void TDIVS_IMPL(TileData &dst, TileData &src0, typename TileData::DType scalar)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    TDivS<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), src0.data(), scalar, validRow, validCol);
}
template <typename TileData>
__aicore__ void TDIVS_IMPL(TileData &dst, typename TileData::DType scalar, TileData &src0)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    TDivS<TileData, elementsPerRepeat, blockSizeElem, rowStride>(dst.data(), scalar, src0.data(), validRow, validCol);
}
} // namespace pto
#endif
