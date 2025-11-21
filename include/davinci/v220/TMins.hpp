#ifndef TMINS_HPP
#define TMINS_HPP

#include "common/constants.hpp"

namespace pto {
    template <typename TileData, typename ScalarType, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ __aicore__ void TMinsImpl(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src,
                                ScalarType scalar,
                                unsigned validRow,
                                unsigned numRepeatPerLine,
                                unsigned numRemainPerLine) {
        using T = typename TileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vmins(
                            dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                            srcPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                            scalar,
                            REPEAT_MAX, 1, 1, 8, 8
                        );
                    }
                }
                if (remainAfterLoop) {
                    vmins(
                        dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                        srcPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                        scalar,
                        remainAfterLoop, 1, 1, 8, 8
                    );
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        srcPtr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine) {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop) {
                for (int i = 0; i < numLoop; i++) {
                    if (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            vmins(
                                dstPtr + i * REPEAT_MAX * stride + j * stride,
                                srcPtr + i * REPEAT_MAX * stride + j * stride,
                                scalar,
                                1, 1, 1, 1, 1
                            );
                        }
                    } else {
                        vmins(
                            dstPtr + i * REPEAT_MAX * stride,
                            srcPtr + i * REPEAT_MAX * stride,
                            scalar,
                            REPEAT_MAX, 1, 1, stride / blockSizeElem, stride / blockSizeElem
                        );
                    }
                }
            }
            if (remainAfterLoop) {
                if (strideOverFlag) {
                    for (uint64_t j = 0; j < remainAfterLoop; j++) {
                        vmins(
                            dstPtr + numLoop * REPEAT_MAX * stride + j * stride,
                            srcPtr + numLoop * REPEAT_MAX * stride + j * stride,
                            scalar,
                            1, 1, 1, 1, 1
                        );
                    }
                } else {
                    vmins(
                        dstPtr + numLoop * REPEAT_MAX * stride,
                        srcPtr + numLoop * REPEAT_MAX * stride,
                        scalar,
                        remainAfterLoop, 1, 1, stride / blockSizeElem, stride / blockSizeElem
                    );
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileData>
    __aicore__ PTO_INLINE void TMINS_IMPL(TileData &dst, TileData &src, typename TileData::DType scalar) {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TMINS: Invalid data type");
        using T = typename TileData::DType;
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        constexpr unsigned stride = TileData::RowStride;
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        unsigned validRow = dst.GetValidRow();

        TMinsImpl<TileData, T, elementsPerRepeat, blockSizeElem, stride>(
            dst.data(), src.data(), scalar, validRow, numRepeatPerLine, numRemainPerLine);
    }
}
#endif