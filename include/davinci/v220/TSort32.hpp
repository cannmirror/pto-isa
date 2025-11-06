#ifndef TSORT32_HPP
#define TSORT32_HPP

#include "constants.hpp"

using namespace pto;
using namespace std;

namespace pto {
    constexpr const uint32_t TSORT32_BLOCK_SIZE = 32;

    template <typename DstTileData, typename SrcTileData, typename IdxTileData,
        unsigned dstStride, unsigned srcStride>
    __tf__ __aicore__ void TSort32Impl(typename DstTileData::TileDType __out__ dst,
                                typename SrcTileData::TileDType __in__ src,
                                typename IdxTileData::TileDType __in__ idx,
                                unsigned validRow,
                                unsigned repeatNumPerRow,
                                unsigned idxStride)
    {
        using T = typename DstTileData::DType;
        using IdxT = typename IdxTileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ IdxT *idxPtr = (__ubuf__ IdxT *)__cce_get_tile_ptr(idx);

        if (repeatNumPerRow < REPEAT_MAX) {
            for (int32_t i = 0; i < validRow; i++) {
                vbitsort(dstPtr + i * dstStride, srcPtr + i * srcStride, idxPtr + i * idxStride, repeatNumPerRow);
                pipe_barrier(PIPE_V);
            }
            // Returning early, compiler constant folding might cause errors, and subsequent function segments might
            // still be compiled. This scenario needs to be changed to an if-else structure.
            return;
        }

        auto loopNum = repeatNumPerRow / REPEAT_MAX;
        auto tailRepeatNum = repeatNumPerRow % REPEAT_MAX;
        for (int32_t i = 0; i < vaildRow; i++) {
            for (int32_t j = 0; j < loopNum; j++) {
                vbitsort(dstPtr + i * dstStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE,
                    srcPtr + i * srcStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE,
                    idxPtr + i * idxStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE, REPEAT_MAX);
                pipe_barrier(PIPE_V);
            }
            vbitsort(dstPtr + i * dstStride + repeatNumPerRow * TSORT32_BLOCK_SIZE,
                srcPtr + i * srcStride + repeatNumPerRow * TSORT32_BLOCK_SIZE,
                idxPtr + i * idxStride + repeatNumPerRow * TSORT32_BLOCK_SIZE, tailRepeatNum);
            pipe_barrier(PIPE_V);
        }
    }

    template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData, unsigned dstStride, unsigned srcStride>
    __tf__ __aicore__ void TSort32Impl(typename DstTileData::TileDType __out__ dst,
                                typename SrcTileData::TileDType __in__ src,
                                typename IdxTileData::TileDType __in__ idx,
                                typename TmpTileData::TileDType __in__ tmp,
                                unsigned validRow,
                                unsigned repeatNumPerRow,
                                unsigned idxStride,
                                unsigned srcShapeBytesPerRow,
                                unsigned srcTailPerRow,
                                unsigned srcTailRepeatNum)
    {
        using T = typename DstTileData::DType;
        using IdxT = typename IdxTileData::DType;
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
        __ubuf__ IdxT *idxPtr = (__ubuf__ IdxT *)__cce_get_tile_ptr(idx);
        __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

        T MIN_VAL = -(0.0 / 0.0);
        constexpr uint32_t MAX_UB_TMP = 8 * 1024;
        // 8KB memory tmp reserved need to decide, meet <= 8KB mean repeat Time <= 255;
        if (srcShapeBytesPerRow <= MAX_UB_TMP) {
            for (int32_t i = 0; i < validRow; i++) {
                // copy row src cbuf to tmp cbuf
                uint16_t lenBurst = (srcShapeBytesPerRow + TSORT32_BLOCK_SIZE - 1) / TSORT32_BLOCK_SIZE;
                copy_ubuf_to_ubuf(tmpPtr,srcPtr + i * srcStride, 0, 1, lenBurst, 0, 0);
                pipe_barrier(PIPE_V);

                // dup -NAN of tial value in tmp cbuf
                uint64_t mask = ~(((static_cast<uint64_t>(1)) << (srcTailPerRow)) - 1);
                set_mask_norm();
                set_vector_mask(0, mask);
                vector_dup(tmpPtr + repeatNumPerRow * TSORT32_BLOCK_SIZE, MIN_VAL, 1, 1, 1, 8, (int64_t)0);
                pipe_barrier(PIPE_V);

                // sort for tmp and out to dst
                vbitsort(dstPtr + i * dstStride, tmpPtr, idxPtr + i * idxStride, repeatNumPerRow + 1);
                pipe_barrier(PIPE_V);
                set_vector_mask(-1, -1);
            }
            return;
        }

        auto loopNum = repeatNumPerRow / REPEAT_MAX + 1;
        for (int32_t i = 0; i < validRow; i++) {
            for (int32_t j = 0; j < loopNum; j++) {
                if (j < loopNum - 1) {
                    // sort for (loopNum - 1) * REPEAT_MAX
                    vbitsort(dstPtr + i * dstStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE,
                        srcPtr + i * srcStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE,
                        idxPtr + i * idxStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE, REPEAT_MAX);
                    pipe_barrier(PIPE_V);
                } else {
                    // sort for last block
                    vbitsort(dstPtr + i * dstStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE,
                        srcPtr + i * srcStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE,
                        idxPtr + i * idxStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE, srcTailRepeatNum - 1);
                    pipe_barrier(PIPE_V);

                    // copy row src cbuf to tmp cbuf
                    uint16_t lenBurst = (srcTailPerRow * sizeof(T) + TSORT32_BLOCK_SIZE - 1) / TSORT32_BLOCK_SIZE;
                    copy_ubuf_to_ubuf(tmpPtr, srcPtr + i * srcStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE +
                        (srcTailRepeatNum - 1) * TSORT32_BLOCK_SIZE, 0, 1, lenBurst, 0, 0);
                    pipe_barrier(PIPE_V);

                    // dup -inf of tial value in tmp cbuf
                    uint64_t mask = ~(((static_cast<uint64_t>(1)) << (srcTailPerRow)) - 1);
                    set_mask_norm();
                    set_vector_mask(0, mask);
                    vector_dup(tmpPtr, MIN_VAL, 1, 1, 1, 8, (int64_t)0);
                    pipe_barrier(PIPE_V);

                    // sort for tmp and out to dst
                    vbitsort(dstPtr + i * dstStride + j * REPEAT_MAX * TSORT32_BLOCK_SIZE +
                        (srcTailRepeatNum - 1) * TSORT32_BLOCK_SIZE, tmpPtr, idxPtr + i * idxStride +
                        j * REPEAT_MAX * TSORT32_BLOCK_SIZE + (srcTailRepeatNum - 1) * TSORT32_BLOCK_SIZE, 1);
                    pipe_barrier(PIPE_V);
                    set_vector_mask(-1, -1);
                }
            }
        }
    }

    template <typename DstTileData, typename SrcTileData, typename IdxTileData>
    __aicore__ PTO_INLINE void CheckStatic()
    {
        static_assert((std::is_same<typename DstTileData::DType, half>::value) ||
                      (std::is_same<typename DstTileData::DType, float>::value),
                      "Dst and src must be half or float.");
        static_assert((std::is_same<typename IdxTileData::DType, uint32_t>::value),
                      "Idx must be uint32_t");
        static_assert((std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value),
                      "Dst and src must be same.");
        static_assert((DstTileData::Loc == Location::Vec) && (SrcTileData::Loc == Location::Vec) &&
                      (IdxTileData::Loc == Location::Vec),
                      "Location must be Vec.");
        static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor && IdxTileData::isRowMajor),
                      "Expect row major");
    }

    // 32 Align Interface, No tmpTile
    template <typename DstTileData, typename SrcTileData, typename IdxTileData>
    __aicore__ PTO_INLINE void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx)
    {
        CheckStatic<DstTileData, SrcTileData, IdxTileData>();
        unsigned validRow = dst.GetValidRow();
        unsigned repeatNumPerRow = src.GetValidCol() / TSORT32_BLOCK_SIZE;
        constexpr unsigned dstStride = DstTileData::RowStride;
        constexpr unsigned srcStride = SrcTileData::RowStride;
        unsigned idxStride = idx.GetValidRow() == 1 ? 0 : IdxTileData::RowStride;

        TSort32Impl<DstTileData, SrcTileData, IdxTileData, dstStride, srcStride>
            (dst.data(), src.data(), idx.data(), validRow, repeatNumPerRow, idxStride);
    }

    // 32 Non-Align Interface, Have tmpTile
    template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
    __aicore__ PTO_INLINE void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp)
    {
        CheckStatic<DstTileData, SrcTileData, IdxTileData>();
        unsigned validRow = dst.GetValidRow();
        unsigned repeatNumPerRow = src.GetValidCol() / TSORT32_BLOCK_SIZE;
        constexpr unsigned byteSize = sizeof(typename DstTileData::DTtype);
        constexpr unsigned idxByteSize = sizeof(typename SrcTileData::DTtype);
        constexpr unsigned dstStride =
            ((DstTileData::RowStride * byteSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE) / byteSize;
        constexpr unsigned srcStride = 
            ((SrcTileData::RowStride * byteSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE) / byteSize;
        constexpr unsigned tmpIdxStride = 
            ((IdxTileData::RowStride * idxByteSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE) / idxByteSize;
        unsigned idxStride = idx.GetValidRow() == 1 ? 0 : tmpIdxStride;

        if (src.GetValidCol() % TSORT32_BLOCK_SIZE > 0) {
            unsigned srcShapeBytesPerRow = src.GetValidCol() * byteSize;
            unsigned srcTailPerRow = src.GetValidCol() % TSORT32_BLOCK_SIZE;
            unsigned srcTailRepeatNum = ((src.GetValidCol() + TSORT32_BLOCK_SIZE - 1) / TSORT32_BLOCK_SIZE) % REPEAT_MAX;
            TSort32Impl<DstTileData, SrcTileData, IdxTileData, TmpTileData, dstStride, srcStride>
                (dst.data(), src.data(), idx.data(), validRow, repeatNumPerRow, idxStride,
                srcShapeBytesPerRow, srcTailPerRow, srcTailRepeatNum);
        } else {
            TSort32Impl<DstTileData, SrcTileData, IdxTileData, dstStride, srcStride>
                (dst.data(), src.data(), idx.data(), validRow, repeatNumPerRow, idxStride);
        }
    }
}
#endif