#ifndef TTRANS_HPP
#define TTRANS_HPP

#define __DAV_V220__
#include "constants.hpp"
#include "common/utils.hpp"

namespace pto {
    template<typename T, unsigned dstStride, unsigned srcStride>
    __aicore__ PTO_INLINE void transpose_full_subtiles(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned num_subtile_x, unsigned num_subtile_y)
    {
        if (num_subtile_x == 0 || num_subtile_y == 0) {
            return;
        }

        constexpr unsigned blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);

        if constexpr (sizeof(T) == 4) {  // b32 case
            // num_subtile_y can reach REPEAT_MAX = 255 in b32 case
            // 1 subtile = 16 * 32B = 0.5KB, UB = 192KB, so UB can contain 384 subtiles at most
            uint64_t srcUb[16] = {0};
            uint64_t dstUb[16] = {0};
            const unsigned num_repeat = (num_subtile_y + REPEAT_MAX - 1) / REPEAT_MAX;
            for (unsigned repeat = 0; repeat < num_repeat; ++repeat,
                srcPtr += 16 * REPEAT_MAX * srcStride,
                dstPtr += 16 * REPEAT_MAX,
                num_subtile_y -= REPEAT_MAX) {
                for (unsigned i = 0; i < num_subtile_x; ++i) {
                    for (unsigned j = 0; j < 16; ++j) {
                        srcUb[j] = (uint64_t)(srcPtr + i * blockElemSize + j * srcStride);
                        dstUb[j] = (uint64_t)(dstPtr + ((j >> 1) + i * blockElemSize) * dstStride + (j & 1) * blockElemSize);
                    }
                    set_va_reg_sb(VA2, srcUb);
                    set_va_reg_sb(VA3, &srcUb[8]);
                    set_va_reg_sb(VA0, dstUb);
                    set_va_reg_sb(VA1, &dstUb[8]);
                    if (num_subtile_y == 1) {
                        scatter_vnchwconv_b32(VA0, VA2, 1, 0, 0);
                    } else if (num_subtile_y < REPEAT_MAX) {
                        scatter_vnchwconv_b32(VA0, VA2, num_subtile_y, 2, 16 * srcStride / blockElemSize);
                    } else {
                        scatter_vnchwconv_b32(VA0, VA2, REPEAT_MAX, 2, 16 * srcStride / blockElemSize);
                    }
                }
            }
        } else if constexpr (sizeOf(T) == 2) {
            // num_subtile_y can reach REPEAT_MAX = 255 in b16 case
            // 1 subtile = 16 * 32B = 0.5KB, UB = 192KB, so UB can contain 384 subtiles at most
            uint64_t srcUb[16] = {0};
            uint64_t dstUb[16] = {0};
            const unsigned num_repeat = (num_subtile_y + REPEAT_MAX - 1) / REPEAT_MAX;
            for (unsigned repeat = 0; repeat < num_repeat; ++repeat,
                srcPtr += 16 * REPEAT_MAX * srcStride,
                dstPtr += 16 * REPEAT_MAX,
                num_subtile_y -= REPEAT_MAX) {
                for (unsigned i = 0; i < num_subtile_x; ++i) {
                    for (unsigned j = 0; j < 16; ++j) {
                        srcUb[j] = (uint64_t)(srcPtr + i * blockElemSize + j * srcStride);
                        dstUb[j] = (uint64_t)(dstPtr + (j + i * blockElemSize) * dstStride);
                    }
                    set_va_reg_sb(VA2, srcUb);
                    set_va_reg_sb(VA3, &srcUb[8]);
                    set_va_reg_sb(VA0, dstUb);
                    set_va_reg_sb(VA1, &dstUb[8]);
                    if (num_subtile_y == 1) {
                        scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
                    } else if (num_subtile_y < REPEAT_MAX) {
                        scatter_vnchwconv_b16(VA0, VA2, num_subtile_y, 1, 16 * srcStride / blockElemSize);
                    } else {
                        scatter_vnchwconv_b16(VA0, VA2, REPEAT_MAX, 1, 16 * srcStride / blockElemSize);
                    }
                }
            }
        } else if constexpr (sizeOf(T) == 1) {
            // num_subtile_y couldn't reach REPEAT_MAX = 255 in b8 case
            // 1 subtile = 32 * 32B = 1KB, UB = 192KB, so UB can contain 192 subtiles at most
            uint64_t srcUb[16] = {0};
            uint64_t dstUb[16] = {0};
            uint64_t srcUb1[16] = {0};
            uint64_t dstUb1[16] = {0};
            for (unsigned i = 0; i < num_subtile_x; ++i) {
                for (unsigned j = 0; j < 16; ++j) {
                    srcUb[j] = (uint64_t)(srcPtr + i * blockElemSize + j * srcStride);
                    srcUb1[j] = srcUb[j] + 16 * srcStride;
                    dstUb[j] = (uint64_t)(dstPtr + (j + i * blockElemSize) * dstStride);
                    dstUb1[j] = dstUb[j] + 16 * dstStride;
                }
                set_va_reg_sb(VA2, srcUb);
                set_va_reg_sb(VA3, &srcUb[8]);
                set_va_reg_sb(VA0, dstUb);
                set_va_reg_sb(VA1, &dstUb[8]);
                set_va_reg_sb(VA4, srcUb1);
                set_va_reg_sb(VA5, &srcUb1[8]);
                set_va_reg_sb(VA6, dstUb1);
                set_va_reg_sb(VA7, &dstUb1[8]);
                if (num_subtile_y == 1) {
                    scatter_vnchwconv_b8(VA0, VA2, 1, 0, 0, false, false);
                    scatter_vnchwconv_b8(VA6, VA2, 1, 0, 0, false, true);
                    scatter_vnchwconv_b8(VA0, VA4, 1, 0, 0, true, false);
                    scatter_vnchwconv_b8(VA6, VA4, 1, 0, 0, true, true);
                } else {
                    scatter_vnchwconv_b8(VA0, VA2, num_subtile_y, 1, 32 * srcStride / blockElemSize, false, false);
                    scatter_vnchwconv_b8(VA6, VA2, num_subtile_y, 1, 32 * srcStride / blockElemSize, false, true);
                    scatter_vnchwconv_b8(VA0, VA4, num_subtile_y, 1, 32 * srcStride / blockElemSize, true, false);
                    scatter_vnchwconv_b8(VA6, VA4, num_subtile_y, 1, 32 * srcStride / blockElemSize, true, true);
                }
            }
        }  else {
            static_assert(sizeOf(T) == 4 || sizeOf(T) == 2 || sizeOf(T) == 1, "TTRANS: Invalid data type.");
        }
    }

    template <typename T, unsigned dstStride, unsigned srcStride>
    __aicore__ PTO_INLINE void copy_rows_with_mask(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned num_row, unsigned num_tail)
    {
        if (num_row == 0 || num_tail == 0) {
            return;
        }

        constexpr uint16_t blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr uint16_t srcRepeatSize = srcStride / blockElemSize;
        constexpr uint16_t dstRepeatSize = dstStride / blockElemSize;

        if constexpr (sizeof(T) == 4) {
            SetContinuousMask(num_tail);
            vcopy((__ubuf__ uint32_t *)dstPtr, (__ubuf__ uint32_t *)srcPtr, num_row, 1, 1, dstRepeatSize, srcRepeatSize);
        } else if constexpr (sizeof(T) == 2) {
            SetContinuousMask(num_tail);
            vcopy((__ubuf__ uint16_t *)dstPtr, (__ubuf__ uint16_t *)srcPtr, num_row, 1, 1, dstRepeatSize, srcRepeatSize);
        } else if constexpr (sizeof(T) == 1) {
            if (num_tail > 1) {
                SetContinuousMask(num_tail / 2);
                vcopy((__ubuf__ uint16_t *)dstPtr, (__ubuf__ uint16_t *)srcPtr, num_row, 1, 1, dstRepeatSize, srcRepeatSize);
            }
            // vcopy(...) doesn't support b8 data type pointers. So in rare case of odd num_tail we should additionally use
            // scalar copy from src to dst for the last dst column
            if (num_tail % 2) {
                // The sync is necessary for scalar copy to be after all vector ops
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                for (unsigned i = 0; i < num_row; ++i) {
                    dstPtr[i * dstStride + num_tail - 1] = srcPtr[i * srcStride + num_tail - 1];
                }
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            }
        } else {
            static_assert(sizeOf(T) == 4 || sizeOf(T) == 2 || sizeOf(T) == 1, "TTRANS: Invalid data type.");
        }
    }

    template <typename T, unsigned dstStride, unsigned srcStride>
    __aicore__ PTO_INLINE void transpose_x_tail_subtiles(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned num_tail_x, unsigned num_subtile_y)
    {
        if (num_tail_x == 0 || num_subtile_y == 0) {
            return;
        }

        constexpr uint16_t blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr uint16_t tmpStride = TMP_UB_SIZE / BLOCK_BYTE_SIZE / sizeof(T);  // 8KB/32B/sizeof(T) elements
        constexpr uint16_t fullBurst = tmpStride / blockElemSize;
        constexpr uint16_t dstFullGap = (uint16_t)(dstStride / blockElemSize) - fullBurst;
        constexpr uint16_t tmpSubtilesMax = sizeOf(T) == 1 ? tmpStride / 32 : tmpStride / 16;

        __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB, start from 184KB, UB:192KB=184+8KB
        const unsigned full_iter_num = num_subtile_y / tmpSubtilesMax;
        for (unsigned iter = 0; iter < full_iter_num; ++iter) {
            transpose_full_subtiles<T, tmpStride, srcStride>(tmpPtr, (srcPtr + iter * tmpStride * srcStride), 1, tmpSubtilesMax);
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf((dstPtr + iter * tmpStride), tmpPtr, 0, num_tail_x, fullBurst, 0, dstFullGap);
            pipe_barrier(PIPE_V);
        }

        if (const uint16_t tmpSubtilesTail = num_subtile_y % tmpSubtilesMax) {
            const uint16_t tailBurst = sizeof(T) == 1 ? 32 * tmpSubtilesTail / blockElemSize : 16 * tmpSubtilesTail / blockElemSize;
            const uint16_t tmpTailGap = tmpStride / blockElemSize - tailBurst;
            const uint16_t dstTailGap = (uint16_t)(dstStride / blockElemSize) - tailBurst;
            transpose_full_subtiles<T, tmpStride, srcStride>(tmpPtr, (srcPtr + full_iter_num * tmpStride * srcStride), 1, tmpSubtilesTail);
            pipe_barrier(PIPE_V);
            copy_ubuf_to_ubuf((dstPtr + full_iter_num * tmpStride), tmpPtr, 0, num_tail_x, tailBurst, tmpTailGap, dstTailGap);
            pipe_barrier(PIPE_V);
        }
    }

    template <typename T, unsigned dstStride, unsigned srcStride>
    __aicore__ PTO_INLINE void transpose_y_tail_subtiles(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned num_subtile_x, unsigned num_tail_y)
    {
        if (num_subtile_x == 0 || num_tail_y == 0) {
            return;
        }

        constexpr uint16_t blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr uint16_t tmpStride = sizeof(T) == 1 ? 32 : 16;
        // Here we decrement tmpSubtilesMax to make sure the copying within TMP UB buffer (8KB)
        // and tmpRowsMax within repeatTimes limit (255)
        constexpr uint16_t tmpSubtilesMax = TMP_UB_SIZE / BLOCK_BYTE_SIZE / tmpStride - 1;  // 8KB/32B/tmpStride-1
        constexpr uint16_t tmpRowsMax = tmpSubtilesMax * blockElemSize;
        static_assert(tmpRowsMax <= REPEAT_MAX);

        __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB, start from 184KB, UB:192KB=184+8KB
        const unsigned full_iter_num = num_subtile_x / tmpSubtilesMax;
        for (unsigned iter = 0; iter < full_iter_num; ++iter) {
            transpose_full_subtiles<T, tmpStride, srcStride>(tmpPtr, (srcPtr + iter * tmpRowsMax), tmpSubtilesMax, 1);
            pipe_barrier(PIPE_V);
            copy_rows_with_mask<T, dstStride, tmpStride>((dstPtr + iter * tmpRowsMax * dstStride), tmpPtr, tmpRowMax, num_tail_y);
            pipe_barrier(PIPE_V);
        }

        if (const unsigned tmpSubtilesTail = num_subtile_x % tmpSubtilesMax) {
            transpose_full_subtiles<T, tmpStride, srcStride>(tmpPtr, (srcPtr + full_iter_num * tmpSubtilesMax * blockElemSize), tmpSubtilesTail, 1);
            pipe_barrier(PIPE_V);
            copy_rows_with_mask<T, dstStride, tmpStride>((dstPtr + full_iter_num * tmpRowsMax * dstStride), tmpPtr, tmpSubtilesTail * blockElemSize, num_tail_y);
            pipe_barrier(PIPE_V);
        }
    }

    template <typename T, unsigned dstStride, unsigned srcStride>
    __aicore__ PTO_INLINE void transpose_xy_tail_subtile(
        __ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned num_tail_x, unsigned num_tail_y)
    {
        if (num_tail_x ==0 || num_tail_y == 0) {
            return;
        }

        constexpr unsigned tmpStride = sizeof(T) == 1 ? 32 : 16;

        __ubuf__ T *tmpPtr = (__ubuf__ T *)(TMP_UB_OFFSET);  // 8KB, start from 184KB, UB:192KB=184+8KB
        transpose_full_subtiles<T, tmpStride, srcStride>(tmpPtr, srcPtr, 1, 1);
        pipe_barrier(PIPE_V);
        copy_rows_with_mask<T, dstStride, tmpStride>(dstPtr, tmpPtr, num_tail_x, num_tail_y);
        pipe_barrier(PIPE_V);
    }

    template <typename TileData, unsigned dstStride, unsigned srcStride>
    __tf__ __aicore__ PTO_INLINE void TTrans(typename TileData::TileDType __out__ dst,
                                typename TileData::TileDType __in__ src,
                                unsigned validRow,
                                unsigned validCol) {
        if (validRow == 0 || validCol == 0) {
            return;
        }

        using T = typename TileData::DType;
        constexpr unsigned blockElemSize = BLOCK_BYTE_SIZE / sizeof(T);

        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

        // Corner case of [1, C]
        if (validRow == 1) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            for (unsigned x = 0; x < validCol; ++x) {
                dstPtr[x * dstStride] = srcPtr[x];
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        // Corner case of [R, 1]
        } else if (validCol == 1) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            for (unsigned y = 0; y < validRow; ++y) {
                dstPtr[y] = srcPtr[y * srcStride];
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        // Common case of [R, C]
        } else {
            const unsigned num_full_subtile_x = validCol / blockElemSize;
            const unsigned num_tail_x = validCol % blockElemSize;
            const unsigned num_full_subtile_y = (sizeof(T) == 1) ? validRow / 32 : validRow / 16;
            const unsigned num_tail_y = (sizeof(T) == 1) ? validRow % 32 : validRow % 16;
            
            transpose_full_subtiles<T, dstStride, srcStride>(
                dstPtr,
                srcPtr,
                num_full_subtile_x,
                num_full_subtile_y);
            
            transpose_x_tail_subtiles<T, dstStride, srcStride>(
                dstPtr + (validCol - num_tail_x) * dstStride,
                srcPtr + (validCol - num_tail_x),
                num_tail_x,
                num_full_subtile_y);
            
            transpose_y_tail_subtiles<T, dstStride, srcStride>(
                dstPtr + (validRow - num_tail_y),
                srcPtr + (validRow - num_tail_y) * srcStride,
                num_full_subtile_x,
                num_tail_y);

            transpose_xy_tail_subtile<T, dstStride, srcStride>(
                dstPtr + (validCol - num_tail_x) * dstStride + (validRow - num_tail_y),
                srcPtr + (validRow - num_tail_y) * srcStride + (validCol - num_tail_x),
                num_tail_x,
                num_tail_y);
        }
    }

    template <typename TileDataDst, typename TileDataSrc>
    __aicore__ PTO_INLINE void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src) {
        using T = typename TileDataSrc::DType;
        static_assert(sizeof(T) == sizeof(typename TileDataDst::DType), "TTRANS: Inconsistent source and destination data types.");
        static_assert(TileDataSrc::isRowMajor, "TTRANS: Inconsistent source BLayout type.");
        static_assert(TileDataDst::isRowMajor, "TTRANS: Inconsistent destination BLayout type.");

        constexpr unsigned dstStride = TileDataDst::RowStride;
        constexpr unsigned srcStride = TileDataSrc::RowStride;
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans<TileDataSrc, dstStride, srcStride>(dst.data(), src.data(), validRow, validCol);
    }
}
#endif