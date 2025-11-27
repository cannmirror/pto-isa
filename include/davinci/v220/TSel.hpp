#ifndef TSEL_HPP
#define TSEL_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"

namespace pto {
enum class SELMODE : uint8_t {
    VSEL_CMPMASK_SPR = 0,
    VSEL_TENSOR_SCALAR_MODE = 1,
    VSEL_TENSOR_TENSOR_MODE = 2,
};

template <typename TileData, typename MaskTile, unsigned rowStride, unsigned maskRowStride>
__tf__
__aicore__
void TSel(typename TileData::TileDType __out__ dst, typename MaskTile::TileDType __in__ selMask,
    typename TileData::TileDType __in__ src0, typename TileData::TileDType __in__ src1, unsigned validRow,
    unsigned validCol)
{
    using T = typename std::conditional<sizeof(typename TileData::DType) == 4, float, half>::type;
    if constexpr (sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 2) {
        __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
        __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
        __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
        uint32_t maskPtr = static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(__cce_get_tile_ptr(selMask))));
        __ubuf__ uint32_t *cmpMaskPtr = reinterpret_cast<__ubuf__ uint32_t *>(get_imm(TMP_UB_OFFSET));  // 8KB tmpbuf address

        set_mask_count();
        for (unsigned i = 0; i < validRow; i++) {
            set_vector_mask(0, BLOCK_BYTE_SIZE);
            vector_dup(cmpMaskPtr, (uint32_t)(maskPtr + i * maskRowStride), 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(cmpMaskPtr);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, validCol);
            vsel((__ubuf__ T *)(dstPtr + i * rowStride),
                (__ubuf__ T *)(src0Ptr + i * rowStride),
                (__ubuf__ T *)(src1Ptr + i * rowStride),
                1, 1, 1, 1, 8, 8, 8, SELMODE::VSEL_TENSOR_TENSOR_MODE);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        static_assert(
            sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 2, "TSEL: Invalid data type.");
    }
}

template <typename TileData, typename MaskTile>
__aicore__
void TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1)
{
    constexpr unsigned rowStride = TileData::RowStride;
    constexpr unsigned maskRowStride = MaskTile::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TSel<TileData, MaskTile, rowStride, maskRowStride>(
        dst.data(), selMask.data(), src0.data(), src1.data(), validRow, validCol);
}
}  // namespace pto
#endif