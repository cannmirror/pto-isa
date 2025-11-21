#ifndef TUNARYOP_HPP
#define TUNARYOP_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common.hpp"
#include "utils.hpp"

using namespace pto;

namespace pto {
    template <typename TileData>
    __tf__ __aicore__ void TRsqrtCustom(typename TileData::TileDType __out__ dst,
                                        typename TileData::TileDType __in__ src,
                                        unsigned validCol, unsigned validRow) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;
            RegTensor<T> vreg2;
            RegTensor<T> vreg3;
            uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileData::DType));
            uint16_t loop_num = CEIL(validCol, batch_size);
            uint32_t count = (batch_size >= validCol ? validCol : batch_size);
            MaskReg preg = CreatePredicate<T>(count);
            vdup(vreg2, (T)1.0, preg, MODE_MERGING);
            for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                for(uint16_t j = 0; j < loop_num; ++j) {
                    vlds(vreg0, srcPtr, (i * TileData::Cols + j * batch_size), NORM);
                    count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                    preg = CreatePredicate<T>(count);
                    vsqrt(vreg1, vreg0, preg, MODE_ZEROING);
                    vdiv(vreg3, vreg2, vreg1, preg);
                    
                    vsts(vreg3, dstPtr, (i * TileData::Cols + j * batch_size), NORM_B32, preg);
                }
            }
        }
    }

    template <typename TileData>
    __tf__ __aicore__ void TSqrtCustom(typename TileData::TileDType __out__ dst,
                                       typename TileData::TileDType __in__ src,
                                       unsigned validCol, unsigned validRow) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;

            uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileData::DType));
            uint16_t loop_num = CEIL(validCol, batch_size);
            for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                for(uint16_t j = 0; j < loop_num; ++j) {
                    vlds(vreg0, srcPtr, (i * TileData::Cols + j * batch_size), NORM);
                    uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                    MaskReg preg = CreatePredicate<T>(count);
                    vsqrt(vreg1, vreg0, preg, MODE_ZEROING);
                    vsts(vreg1, dstPtr, (i * TileData::Cols + j * batch_size), NORM_B32, preg);
                }
            }
        }
    }

    template <typename TileData>
    __tf__ __aicore__ void TExpCustom(typename TileData::TileDType __out__ dst,
                                      typename TileData::TileDType __in__ src,
                                      unsigned validCol, unsigned validRow) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        __VEC_SCOPE__
        {
            RegTensor<T> vreg0;
            RegTensor<T> vreg1;

            uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileData::DType));
            uint16_t loop_num = CEIL(validCol, batch_size);
            for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                for(uint16_t j = 0; j < loop_num; ++j) {
                    vlds(vreg0, srcPtr, (i * TileData::Cols + j * batch_size), NORM);
                    uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                    MaskReg preg = CreatePredicate<T>(count);
                    vexp(vreg1, vreg0, preg, MODE_ZEROING);
                    vsts(vreg1, dstPtr, (i * TileData::Cols + j * batch_size), NORM_B32, preg);
                }
            }
        }
    }
 
    template <typename TileData>
    __aicore__ void TRSQRT_IMPL(TileData &dst, TileData &src) {
        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();

        TRsqrtCustom<TileData>(dst.data(), src.data(), validCol, validRow);
    }

    template <typename TileData>
    __aicore__ void TSQRT_IMPL(TileData &dst, TileData &src) {
        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();

        TSqrtCustom<TileData>(dst.data(), src.data(), validCol, validRow);
    }

    template <typename TileData>
    __aicore__ void TEXP_IMPL(TileData &dst, TileData &src) {
        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();

        TExpCustom<TileData>(dst.data(), src.data(), validCol, validRow);
    }
}
#endif