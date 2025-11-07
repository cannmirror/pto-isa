#ifndef TSTORE_HPP
#define TSTORE_HPP

namespace pto {
    template <typename GlobalData, typename TileData>
    __aicore__ PTO_INLINE void TStoreInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src,
        uint16_t nBurst, uint32_t lenBurst, uint32_t gmGap, uint32_t ubGap)
    {
        if constexpr (sizeof(typename TileData::DType) == 1) {
            copy_ubuf_to_gm_align_b8(dst,
                src,
                0, /* sid */
                nBurst,
                lenBurst,
                0 /* left padding count */,
                0 /* right padding count */,
                ubGap,
                gmGap);
        } else if constexpr (sizeof(typename TileData::DType) == 2) {
            copy_ubuf_to_gm_align_b16(dst,
                src,
                0, /* sid */
                nBurst,
                lenBurst,
                0 /* left padding count */,
                0 /* right padding count */,
                ubGap,
                gmGap);
        } else if constexpr (sizeof(typename TileData::DType) == 4) {
            copy_ubuf_to_gm_align_b32(dst,
                src,
                0, /* sid */
                nBurst,
                lenBurst,
                0 /* left padding count */,
                0 /* right padding count */,
                ubGap,
                gmGap);
        }
    }

    template <typename GlobalData, typename TileData>
    __tf__ __aicore__ void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        constexpr uint32_t blockSize = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        __ubuf__ typename TileData::DType *srcAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
        typename GlobalData::DType *dstAddr = dst;

        // 处理ND格式数据
        if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            // 单条指令处理一个二维数据搬运
            uint16_t nBurst = gShape3;
            uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
            uint32_t gmGap = (gStride3 - gShape4) * sizeof(typename TileData::DType);
            uint32_t ubGap = (TileData::Cols - validCol) / blockSize;

            typename GlobalData::DType *dstGlobalAddr = dstAddr;
            __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;

            int64_t srcStride2 = gShape3 * TileData::Cols;
            int64_t srcStride1 = gShape2 * srcStride2;
            int64_t srcStride0 = gShape1 * srcStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t srcAddr0 = i * srcStride0;
                int64_t dstAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t srcAddr1 = j * srcStride1;
                    int64_t dstAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        dstGlobalAddr = dstAddr + dstAddr0 + dstAddr1 + k * gStride2;
                        srcTileAddr = srcAddr + srcAddr0 + srcAddr1 + k * srcStride2;
                        TStoreInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
                    }
                }
            }
        }
        // 处理DN格式数据
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        uint16_t nBurst = gShape4;
        uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
        uint32_t gmGap = (gStride4 - gShape3) * sizeof(typename TileData::DType);
        uint32_t ubGap = (TileData::Rows - gShape3) / blockSize;

        typename GlobalData::DType *dstGlobalAddr = dstAddr;
        __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;

        int64_t srcStride2 = TileData::Rows * gShape4;
        int64_t srcStride1 = gShape2 * srcStride2;
        int64_t srcStride0 = gShape1 * srcStride1;
        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t srcAddr0 = i * srcStride0;
            int64_t dstAddr0 = i * gStride0;
            for (uint32_t j = 0; j < gShape1; j++) {
                int64_t srcAddr1 = j * srcStride1;
                int64_t dstAddr1 = j * gStride1;
                for (uint32_t k = 0; k < gShape2; k++) {
                    dstGlobalAddr = dstAddr + dstAddr0 + dstAddr1 + k * gStride2;
                    srcTileAddr = srcAddr + srcAddr0 + srcAddr1 + k * srcStride2;
                    TStoreInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
                }
            }
        }
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        // 小分型由gShape3 * gShape4表示
        constexpr uint32_t c0_size = 32;
        uint16_t nBurst = gShape1;
        uint32_t lenBurst = validRow * c0_size;
        uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
        uint32_t ubGap = TileData::Rows - validRow;

        typename GlobalData::DType *dstGlobalAddr = dstAddr;
        __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;

        int64_t tileStride = gShape1 * gShape2 * TileData::Rows * gShape4;
        for (uint32_t i = 0; i < gShape0; i++) {
            dstGlobalAddr = dstAddr +  i * gStride0;
            srcTileAddr = srcAddr + i * tileStride;
            TStoreInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);   
        }
    }

    template <typename GlobalData, typename TileData>
    __aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
    {
        static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                      (sizeof(typename TileData::DType) == 4), "Data type must be b8/16/32");
        static_assert(TileData::Loc == pto::Location::Vec, "Source location only support Vec!");
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype!");
        static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                      (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                      (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
                      "Src and dst layout must be same!");
        
        // gm_to_vecTile处理
        if constexpr (TileData::Loc == pto::Location::Vec) {
            TStore<GlobalData, TileData>(dst.data(),
                src.data(),
                dst.GetShape(0),
                dst.GetShape(1),
                dst.GetShape(2),
                dst.GetShape(3),
                dst.GetShape(4),
                dst.GetStride(0),
                dst.GetStride(1),
                dst.GetStride(2),
                dst.GetStride(3),
                dst.GetStride(4),
                src.GetValidRow(),
                src.GetValidCol());
        }
    }
}
#endif