#ifndef TSTORE_HPP
#define TSTORE_HPP

namespace pto
{
    template <typename TileData, typename GlobalData>
    __aicore__ PTO_INLINE void TStoreInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src, 
    uint16_t nBurst, uint16_t lenBurst, uint64_t burstDstStride, uint32_t burstSrcStride) {
        copy_ubuf_to_gm_align_v2(dst, src, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
    }


    template <typename GlobalData, typename TileData>
    __tf__ __aicore__ void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol) {

            __ubuf__ typename TileData::DType *srcAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
            typename GlobalData::DType *dstAddr = dst;

            typename GlobalData::DType *dstGlobalAddr = dstAddr;
            __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;

            if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {    
                uint32_t loop1SrcStride = gShape3 * TileData::Cols * sizeof(typename TileData::DType);
                uint32_t loop1DstStride = gStride2 * sizeof(typename TileData::DType);
                uint32_t loop2SrcStride = gShape2 * gShape3 * TileData::Cols * sizeof(typename TileData::DType);
                uint32_t loop2DstStride = gStride1 * sizeof(typename TileData::DType);
                    
                uint64_t loop1Config = 0;
                loop1Config |= ((uint64_t)loop1SrcStride) << 40;
                loop1Config |= (uint64_t)loop1DstStride;
                set_loop1_stride_ubtoout(loop1Config);
                uint64_t loop2Config = 0;
                loop2Config |= ((uint64_t)loop2SrcStride) << 40;
                loop2Config |= (uint64_t)loop2DstStride;
                set_loop2_stride_ubtoout(loop2Config);

                uint64_t loopSizeConfig = 0;
                uint64_t loop1Size = gShape2 & 0x1FFFFF;
                loopSizeConfig |= loop1Size;
                uint64_t loop2Size = (static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21;
                loopSizeConfig |= loop2Size;
                set_loop_size_ubtoout(loopSizeConfig);


                uint64_t srcStride0 = gShape1 * gShape2 * gShape3 * TileData::Cols;
                uint16_t nBurst = gShape3;
                uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
                uint64_t burstDstStride = gStride3 * sizeof(typename TileData::DType);
                uint32_t burstSrcStride = TileData::Cols * sizeof(typename TileData::DType);  
                for (uint32_t k = 0; k < gShape0; k++) {
                    dstGlobalAddr = dstAddr + k * gStride0;
                    srcTileAddr = srcAddr +  k * srcStride0;
                    TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
                }	
            } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
                uint32_t loop1SrcStride = TileData::Rows * gShape4 * sizeof(typename TileData::DType);
                uint32_t loop1DstStride = gStride2 * sizeof(typename TileData::DType);
                uint32_t loop2SrcStride = gShape2 * TileData::Rows * gShape4  * sizeof(typename TileData::DType);
                uint32_t loop2DstStride = gStride1 * sizeof(typename TileData::DType);
                    
                uint64_t loop1Config = 0;
                loop1Config |= ((uint64_t)loop1SrcStride) << 40;
                loop1Config |= (uint64_t)loop1DstStride;
                set_loop1_stride_ubtoout(loop1Config);
                uint64_t loop2Config = 0;
                loop2Config |= ((uint64_t)loop2SrcStride) << 40;
                loop2Config |= (uint64_t)loop2DstStride;
                set_loop2_stride_ubtoout(loop2Config);
                    
                uint64_t loopSizeConfig = 0;
                uint64_t loop1Size = gShape2 & 0x1FFFFF;
                loopSizeConfig |= loop1Size;
                uint64_t loop2Size = (static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21;
                loopSizeConfig |= loop2Size;
                set_loop_size_ubtoout(loopSizeConfig);

                uint64_t srcStride0 = gShape1 * gShape2 * gShape4 * TileData::Rows;
                uint16_t nBurst = gShape4;
                uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
                uint64_t burstDstStride = gStride4 * sizeof(typename TileData::DType);
                uint32_t burstSrcStride = TileData::Rows * sizeof(typename TileData::DType);
                for (uint32_t k = 0; k < gShape0; k++) {
                    dstGlobalAddr = dstAddr + k * gStride0;
                    srcTileAddr = srcAddr + k * srcStride0;
                    TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
                }
            }  else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
                constexpr uint32_t c0_size = 32;
                uint16_t nBurst = gShape1;
                uint32_t lenBurst = validRow * c0_size;
                uint64_t burstDstStride = gStride1 * sizeof(typename TileData::DType);
                uint32_t burstSrcStride = TileData::Rows * c0_size;
                int64_t tileStride = gShape1 * gShape2 * TileData::Rows * gShape4;
                for (uint32_t k = 0; k < gShape0; k++) {
                    dstGlobalAddr = dstAddr + k * gStride0;
                    srcTileAddr = srcAddr + k * tileStride;
                    TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
                }
            }
        }

    template <typename TileData, typename GlobalData>
    __aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src) {
        static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                          (sizeof(typename TileData::DType) == 4),
            "Data type must be b8/b16/b32");
        static_assert(TileData::Loc == pto::Location::Vec, "Source location only suport Vec!");
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
            "Source dtype must be same with dst dtype!");
        static_assert(((GlobalData::layout == pto::Layout::ND) && 
                          (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                          ((GlobalData::layout == pto::Layout::DN) &&
                              (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                          ((GlobalData::layout == pto::Layout::NZ) &&
                              (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
            "Src and dst layout must be same!");
        static_assert(((GlobalData::layout == pto::Layout::ND) && (TileData::Cols * sizeof(typename TileData::DType) % 32 == 0)) ||
         ((GlobalData::layout == pto::Layout::DN) && (TileData::Rows * sizeof(typename TileData::DType) % 32 == 0)) || 
         (GlobalData::layout == pto::Layout::NZ));
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

