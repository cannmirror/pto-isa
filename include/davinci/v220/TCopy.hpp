#ifndef TCOPY_HPP
#define TCOPY_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto{
    inline [aicore] void set_mark(uint64_t v) {
        __asm__ __volatile__("");
        asm volatile("MOV COND, %0\n":"+l"(v));

        __asm__ __volatile__("");
    }

    template <typename TileDataDst, typename TileDataSrc, TCopyMode copyMode, unsigned blockSizeElem, unsigned srcStride, unsigned dstStride>
    __tf__ __aicore__ PTO_INLINE void TCopy(typename TileDataDst::TileDType __out__ dst,
                                            typename TileDataSrc::TileDType __in__ src,
                                            uint64_t validRow, uint64_t validCol) {
        if (validRow ==0 || validCol == 0) {
            return;
        }
        using T = typename TileDataSrc::TileDType;
        using U = typename TileDataDst::TileDType;
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);

        if constexpr(copyMode == pto::TCopyMode::SHALLOW_COPY) {  // shadow copy
            (__ubuf__ U*)dstPtr = (__ubuf__ T*)srcPtr;
        } else {  // deep copy
            static_assert(sizeof(T) == sizeof(U), "TCOPY: Unsupported deep copy if data type of src and dst is different!");
            set_mask_count();  // counter mode
            if constexpr (sizeof(T) == 4) {
                uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
                uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
                set_mark(0xdeadbeef);
                set_mark(validCol);
                set_mark(srcRepeatSize);
                set_mark(dstRepeatSize);
                set_vector_mask(0, validCol);
                vcopy((__ubuf__ uint32_t *)(dstPtr), (__ubuf__ uint32_t *)(srcPtr), validRow, 1, 1, dstRepeatSize, srcRepeatSize);
            } else if constexpr (sizeof(T) == 2) {
                uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem);
                uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem);
                set_vector_mask(0, validCol);
                vcopy((__ubuf__ uint16_t *)(dstPtr), (__ubuf__ uint16_t *)(srcPtr), validRow, 1, 1, dstRepeatSize, srcRepeatSize);
            } else if constexpr (sizeof(T) == 1) {
                uint64_t mask = validCol / 2;
                uint64_t num_tail = validCol % 2;
                uint16_t srcRepeatSize = CeilDivision(srcStride, blockSizeElem >> 1);
                uint16_t dstRepeatSize = CeilDivision(dstStride, blockSizeElem >> 1);
                set_vector_mask(0, mask);
                vcopy((__ubuf__ uint16_t *)(dstPtr), (__ubuf__ uint16_t *)(srcPtr), validRow, 1, 1, dstRepeatSize, srcRepeatSize);
                if (num_tail) {
                    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    wait_flag(PEPE_V, PIPE_S, EVENT_ID7);
                    for (unsigned i = 0; i < validRow; ++i) {
                        dstPtr[i * dstStride + num_tail - 1] = srcPtr[i * srcStride + num_tail - 1];
                    }
                    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                }
            } else {
                static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TCOPY: Invalid data type");
            }
            set_mask_norm();  // restore to norm mode
            set_vector_mask(-1, -1);
        }  // end of deep copy
    }  // end of tf

    template <typename TileDataDst, typename TileDataSrc, TCopyMode copyMode>
    __aicore__ PTO_INLINE void TCOPY(TileDataDst &dst, TileDataSrc &src) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);
        constexpr unsigned srcStride = TileDataSrc::RowStride;
        constexpr unsigned dstStride = TileDataDst::RowStride;
        uint64_t validSrcRow = src.GetValidRow();
        uint64_t validSrcCol = src.GetValidCol();
        uint64_t validDstRow = dst.GetValidRow();
        uint64_t validDstCol = dst.GetValidCol();

        TCopy<TileDataDst, TileDataSrc, copyMode, blockSizeElem, srcStride, dstStride>(dst.data(), src.data(), validRow, validCol);
    }
}

#endif