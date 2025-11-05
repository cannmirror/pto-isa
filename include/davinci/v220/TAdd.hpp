#ifndef TADD_HPP
#define TADD_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
    __tf__ __aicore__ void TAdd(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src0,
                                typename TileData::TileDType __in__ src1,
                                unsigned validRow,
                                unsigned validCol) {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *src0Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileData::DType *src1Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src1);
        
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;

        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vadd(dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src0Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src1Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                    }
                }
                if (remainAfterLoop) {
                    vadd(dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         src0Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         src1Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         remainAfterLoop, 1, 1, 1, 8, 8, 8);
                }   
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        src0Ptr += numRepeatPerLine * elementsPerRepeat;
        src1Ptr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine) {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop) {
                for (int i = 0; i < numLoop; i++) {
                    if (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            vadd(dstPtr + i * REPEAT_MAX * stride + j * stride,
                                 src0Ptr + i * REPEAT_MAX * stride + j * stride,
                                 src1Ptr + i * REPEAT_MAX * stride + j * stride,
                                 1, 1, 1, 1, 1, 1, 1);
                        }
                    } else {
                        vadd(dstPtr + i * REPEAT_MAX * stride,
                             src0Ptr + i * REPEAT_MAX * stride,
                             src1Ptr + i * REPEAT_MAX * stride,
                             REPEAT_MAX, 1, 1, 1, stride / blockSizeElem, stride / blockSizeElem, stride / blockSizeElem);
                    }
                }
            }
            if (remainAfterLoop) {
                if (strideOverFlag) {
                    for (uint64_t j = 0; j < remainAfterLoop; j++) {
                        vadd(dstPtr + numLoop * REPEAT_MAX * stride + j * stride,
                             src0Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                             src1Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                             1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    vadd(dstPtr + numLoop * REPEAT_MAX * stride,
                         src0Ptr + numLoop * REPEAT_MAX * stride,
                         src1Ptr + numLoop * REPEAT_MAX * stride,
                         remainAfterLoop, 1, 1, 1, stride / blockSizeElem, stride / blockSizeElem, stride / blockSizeElem);
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileData>
    __aicore__ void TADD_IMPL(TileData &dst, TileData &src0, TileData &src1) {
        static_assert(std::is_same<typename TileData::DType, int32_t>::value ||
                      std::is_same<typename TileData::DType, int>::value ||
                      std::is_same<typename TileData::DType, int16_t>::value ||
                      std::is_same<typename TileData::DType, half>::value ||
                      std::is_same<typename TileData::DType, float16_t>::value ||
                      std::is_same<typename TileData::DType, float>::value ||
                      std::is_same<typename TileData::DType, float32_t>::value,
                      "TADD: Invalid data type");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();

        TAdd<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), src0.data(), src1.data(),
                                                                 validRow, validCol);
    }
}
#endif