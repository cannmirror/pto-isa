#ifndef TCOPY_HPP
#define TCOPY_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto{
    template <typename TileDataDst, typename TileDataSrc, unsigned blockSizeElem, unsigned srcStride,
        unsigned dstStride>
    __tf__ __aicore__ PTO_INLINE void TCopy(typename TileDataDst::TileDType __out__ dst,
        typename TileDataSrc::TileDType __in__ src, uint64_t validRow, uint64_t validCol) {
        if (validRow ==0 || validCol == 0) {
            return;
        }
        using T = typename TileDataSrc::DType;
        using U = typename TileDataDst::DType;
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);

        static_assert(sizeof(T) == sizeof(U), "TCOPY: src and dst data type is different!");
        set_mask_count();  // counter mode
        if constexpr (sizeof(T) == 4) {
            uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
            uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
            unsigned numLoop = validRow / REPEAT_MAX; // REPEAT_MAX = 255
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            set_vector_mask(0, validCol);
            for (int i = 0; i < numLoop; i++) {
                vcopy((__ubuf__ uint32_t *)(dstPtr + i * validCol * REPEAT_MAX),
                    (__ubuf__ uint32_t *)(srcPtr + i * validCol * REPEAT_MAX),
                    REPEAT_MAX,
                    1,
                    1,
                    dstRepeatSize,
                    srcRepeatSize);
            }
            if (remainAfterLoop) {
                vcopy((__ubuf__ uint32_t *)(dstPtr + numLoop * validCol * REPEAT_MAX),
                    (__ubuf__ uint32_t *)(srcPtr + numLoop * validCol * REPEAT_MAX),
                    remainAfterLoop,
                    1,
                    1,
                    dstRepeatSize,
                    srcRepeatSize);
            }
        } else if constexpr (sizeof(T) == 2) {
            uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
            uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
            unsigned numLoop = validRow / REPEAT_MAX; // REPEAT_MAX = 255
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            set_vector_mask(0, validCol);
            for (int i = 0; i < numLoop; i++) {
                vcopy((__ubuf__ uint16_t *)(dstPtr + i * validCol * REPEAT_MAX),
                    (__ubuf__ uint16_t *)(srcPtr + i * validCol * REPEAT_MAX),
                    REPEAT_MAX,
                    1,
                    1,
                    dstRepeatSize,
                    srcRepeatSize);
            }
            if (remainAfterLoop) {
                vcopy((__ubuf__ uint16_t *)(dstPtr + numLoop * validCol * REPEAT_MAX),
                    (__ubuf__ uint16_t *)(srcPtr + numLoop * validCol * REPEAT_MAX),
                    remainAfterLoop,
                    1,
                    1,
                    dstRepeatSize,
                    srcRepeatSize);
            }
        } else if constexpr (sizeof(T) == 1) {
            uint64_t mask = validCol >> 1;
            uint64_t num_tail = validCol % 2;
            uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
            uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
            unsigned numLoop = validRow / REPEAT_MAX; // REPEAT_MAX = 255
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            set_vector_mask(0, mask);
            for (int i = 0; i < numLoop; i++) {
                vcopy((__ubuf__ uint16_t *)(dstPtr + i * validCol * REPEAT_MAX),
                    (__ubuf__ uint16_t *)(srcPtr + i * validCol * REPEAT_MAX),
                    REPEAT_MAX,
                    1,
                    1,
                    dstRepeatSize,
                    srcRepeatSize);
            }
            if (remainAfterLoop) {
                vcopy((__ubuf__ uint16_t *)(dstPtr + numLoop * validCol * REPEAT_MAX),
                    (__ubuf__ uint16_t *)(srcPtr + numLoop * validCol * REPEAT_MAX),
                    remainAfterLoop,
                    1,
                    1,
                    dstRepeatSize,
                    srcRepeatSize);
            }
            if (num_tail) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                for (unsigned i = 0; i < validRow; ++i) {
                    dstPtr[i * dstStride + num_tail - 1] = srcPtr[i * srcStride + num_tail - 1];
                }
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            }
        } else {
            static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TCOPY: Invalid data type.");
        }
        set_mask_norm();  // restore to norm mode
        set_vector_mask(-1, -1);
    }  // end of tf

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TCOPY_IMPL(TileDataDst &dst, TileDataSrc &src) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);
        constexpr unsigned srcStride = TileDataSrc::RowStride;
        constexpr unsigned dstStride = TileDataDst::RowStride;
        uint64_t validSrcRow = src.GetValidRow();
        uint64_t validSrcCol = src.GetValidCol();
        uint64_t validDstRow = dst.GetValidRow();
        uint64_t validDstCol = dst.GetValidCol();

        uint64_t validRow = (validSrcRow < validDstRow) ? validSrcRow : validDstRow;
        uint64_t validCol = (validSrcCol < validDstCol) ? validSrcCol : validDstCol;

        TCopy<TileDataDst, TileDataSrc, blockSizeElem, srcStride, dstStride>(
            dst.data(), src.data(), validRow, validCol);
    }
}
#endif