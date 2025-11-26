#ifndef TGATHERB_HPP
#define TGATHERB_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

using namespace std;

namespace pto {

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstRowStride, unsigned offsetRowStride>
__tf__ __PTO_INSTR__ void TGatherB(typename TileDataDst::TileDType __out__ dst,
    typename TileDataSrc::TileDType __in__ src, typename TileDataOffset::TileDType __in__ offset, unsigned validRow,
    unsigned validCol)
{
    __ubuf__ uint32_t *offsetPtr = (__ubuf__ uint32_t *)__cce_get_tile_ptr(offset);
    uint32_t srcAddr = (uint64_t)(__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src);
    if constexpr (sizeof(typename TileDataDst::DType) == 4 || sizeof(typename TileDataDst::DType) == 2) {
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;
        using T = typename std::conditional<sizeof(typename TileDataDst::DType) == 4, uint32_t, uint16_t>::type;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vgatherb((__ubuf__ T *)(dstPtr + i * dstRowStride + j * elementsPerRepeat * REPEAT_MAX),
                            offsetPtr + i * offsetRowStride + j * 8 * REPEAT_MAX,
                            srcAddr,
                            8,
                            1,
                            REPEAT_MAX);
                    }
                }
                if (remainAfterLoop) {
                    vgatherb((__ubuf__ T *)(dstPtr + i * dstRowStride + numLoop * elementsPerRepeat * REPEAT_MAX),
                        offsetPtr + i * offsetRowStride + numLoop * 8 * REPEAT_MAX,
                        srcAddr,
                        8,
                        1,
                        remainAfterLoop);
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        offsetPtr += numRepeatPerLine * 8;
        if (numRemainPerLine) {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            if (numLoop) {
                for (int i = 0; i < numLoop; i++) {
                    vgatherb((__ubuf__ T *)(dstPtr + i * REPEAT_MAX * dstRowStride),
                        offsetPtr + i * REPEAT_MAX * offsetRowStride,
                        srcAddr,
                        dstRowStride / blockSizeElem,
                        1,
                        REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                vgatherb((__ubuf__ T *)(dstPtr + numLoop * REPEAT_MAX * dstRowStride),
                    offsetPtr + numLoop * REPEAT_MAX * offsetRowStride,
                    srcAddr,
                    dstRowStride / blockSizeElem,
                    1,
                    remainAfterLoop);
            }
        }
    } else if constexpr (sizeof(typename TileDataDst::DType) == 1) {
        __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;
        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vgatherb((__ubuf__ uint16_t *)(dstPtr + i * dstRowStride + j * elementsPerRepeat * REPEAT_MAX),
                            offsetPtr + i * offsetRowStride + j * 8 * REPEAT_MAX,
                            srcAddr,
                            8,
                            1,
                            REPEAT_MAX);
                    }
                }
                if (remainAfterLoop) {
                    vgatherb(
                        (__ubuf__ uint16_t *)(dstPtr + i * dstRowStride + numLoop * elementsPerRepeat * REPEAT_MAX),
                        offsetPtr + i * offsetRowStride + numLoop * 8 * REPEAT_MAX,
                        srcAddr,
                        8,
                        1,
                        remainAfterLoop);
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        offsetPtr += numRepeatPerLine * 8;
        if (numRemainPerLine) {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            if (numLoop) {
                for (int i = 0; i < numLoop; i++) {
                    vgatherb((__ubuf__ uint16_t *)(dstPtr + i * REPEAT_MAX * dstRowStride),
                        offsetPtr + i * REPEAT_MAX * offsetRowStride,
                        srcAddr,
                        dstRowStride / blockSizeElem,
                        1,
                        REPEAT_MAX);
                }
            }
            if (remainAfterLoop) {
                vgatherb((__ubuf__ uint16_t *)(dstPtr + numLoop * REPEAT_MAX * dstRowStride),
                    offsetPtr + numLoop * REPEAT_MAX * offsetRowStride,
                    srcAddr,
                    dstRowStride / blockSizeElem,
                    1,
                    remainAfterLoop);
            }
        }
    } else {
        static_assert(sizeof(typename TileDataDst::DType) == 4 || sizeof(typename TileDataDst::DType) == 2 ||
                          sizeof(typename TileDataDst::DType) == 1,
            "TGATHERB: Invalid data type.");
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset>
__PTO_INSTR__ void TGATHERB_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned offsetRowStride = TileDataOffset::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TGatherB<TileDataDst, TileDataSrc, TileDataOffset, elementsPerRepeat, blockSizeElem, dstRowStride, offsetRowStride>(
        dst.data(), src.data(), offset.data(), validRow, validCol);
}
}  // namespace pto

#endif