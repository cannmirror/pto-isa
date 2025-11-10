#ifndef TLOAD_HPP
#define TLOAD_HPP

namespace pto {

template <typename TileData>
__aicore__ constexpr auto getPadValue()
{
    if constexpr ( std::is_same<typename TileData::DType, float>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint32_t (0);
            case PadValue::Min: return uint32_t (0xff800000UL);				
            case PadValue::Max: return uint32_t (0x7f800000UL);
        }
    }
    else if constexpr ( std::is_same<typename TileData::DType, int32_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint32_t (0);
            case PadValue::Min: return uint32_t (0xffffffffUL);				
            case PadValue::Max: return uint32_t (0x7fffffffUL);
        }			
    }		
    else if constexpr ( std::is_same<typename TileData::DType, uint32_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero:
            case PadValue::Min: return uint32_t (0);
            case PadValue::Max: return uint32_t (0xffffffffUL);

        }			
    }		
    else if constexpr ( std::is_same<typename TileData::DType, bfloat16_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint16_t (0);
            case PadValue::Min: return uint16_t (0xff80);
            case PadValue::Max: return uint16_t (0x7f80);
        }			
    }		
    else if constexpr ( std::is_same<typename TileData::DType, half>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint16_t (0);
            case PadValue::Min: return uint16_t (0xfc00);				
            case PadValue::Max: return uint16_t (0x7c00);
        }			
    }
    else if constexpr ( std::is_same<typename TileData::DType, int16_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint16_t (0);
            case PadValue::Min: return uint16_t (0xffff);		
            case PadValue::Max: return uint16_t (0x7fff);
        }			
    }		
    else if constexpr ( std::is_same<typename TileData::DType, uint16_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: 
            case PadValue::Min: return uint16_t (0);
            case PadValue::Max: return uint16_t (0xffff);
        }			
    }	
    else if constexpr ( std::is_same<typename TileData::DType, int8_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint8_t (0);
            case PadValue::Min: return uint8_t (0xff);		
            case PadValue::Max: return uint8_t (0x7f);
        }			
    }		
    else if constexpr ( std::is_same<typename TileData::DType, uint8_t>::value )
    {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: 
            case PadValue::Min: return uint8_t (0);
            case PadValue::Max: return uint8_t (0xff);
        }			
    }
    else {
            static_assert(sizeof(TileData::DType)<0, "TLOAD: Unsupported DType for PadValue!");
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadInstr(__ubuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
    uint16_t nBurst, uint16_t lenBurst, uint64_t gmStride, uint32_t ubStride, bool ubPad)
{
        copy_gm_to_ubuf_align_v2(dst, src, 0 /*sid*/, nBurst, lenBurst, 
            0 /*left padding count*/, 0 /*right padding count*/, ubPad /*data select bit*/, 0 /*l2 cache ctl*/, gmStride, ubStride);
}

template <typename TileData, typename GlobalData>
__tf__ __aicore__ PTO_INLINE void TLoad(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    __ubuf__ typename TileData::DType *dstAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    constexpr bool ubPad = TileData::PadVal != PadValue::Null;
	if constexpr ( ubPad ) 
	{
		set_mov_pad_val(getPadValue<TileData>());
	}
    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) 
	{
        uint16_t nBurst = gShape3;
        uint16_t lenBurst = validCol * sizeof(typename TileData::DType);
        uint64_t gmStride = gStride3 * sizeof(typename TileData::DType);
        uint32_t ubStride = TileData::Cols * sizeof(typename TileData::DType);
        typename GlobalData::DType *srcAddrP = srcAddr;
        __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

        int64_t dstStride2 = gShape3 * TileData::Cols;
        int64_t dstStride1 = gShape2 * dstStride2;
        int64_t dstStride0 = gShape1 * dstStride1;
		constexpr const bool useLoopOpt = true;
		if constexpr ( useLoopOpt )
		{
			uint64_t loop2 = gShape1;
			uint64_t loop1 = gShape2;
			uint64_t loop2_src_stride = gStride1*sizeof(typename TileData::DType);
			uint64_t loop1_src_stride = gStride2*sizeof(typename TileData::DType);
			uint64_t loop2_dst_stride = dstStride1*sizeof(typename TileData::DType);
			uint64_t loop1_dst_stride = dstStride2*sizeof(typename TileData::DType);
			set_loop2_stride_outtoub(loop2_dst_stride<<40 | loop2_src_stride);
			set_loop1_stride_outtoub(loop1_dst_stride<<40 | loop1_src_stride);
			set_loop_size_outtoub(loop2<<21 | loop1);		
			for (uint32_t i = 0; i < gShape0; i++) {
				int64_t dstAddr0 = i * dstStride0;
				int64_t srcAddr0 = i * gStride0;
				dstAddrP = dstAddr + dstAddr0;
				srcAddrP = srcAddr + srcAddr0;
				TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, ubPad);			
			}
			set_loop_size_outtoub(1ULL<<21 | 1ULL);	
		}
		else
		{
			set_loop_size_outtoub(1ULL<<21 | 1ULL);
			for (uint32_t i = 0; i < gShape0; i++) {
				int64_t dstAddr0 = i * dstStride0;
				int64_t srcAddr0 = i * gStride0;
				for (uint32_t j = 0; j < gShape1; j++) {	//loop2
					int64_t dstAddr1 = j * dstStride1;
					int64_t srcAddr1 = j * gStride1;
					for (uint32_t k = 0; k < gShape2; k++) {	//loop1
						dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
						srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
						TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, ubPad);
					}
				}			
			}
		}		
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        uint16_t nBurst = gShape4;
        uint16_t lenBurst = validRow * sizeof(typename TileData::DType);
        uint64_t gmGapValue = (gStride4 - gShape3) * sizeof(typename TileData::DType);
        uint64_t gmStride = gStride4 * sizeof(typename TileData::DType);
        uint32_t ubStride = TileData::Rows * sizeof(typename TileData::DType);		
        typename GlobalData::DType *srcAddrP = srcAddr;
        __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

        int64_t dstStride2 = gShape4 * TileData::Rows;
        int64_t dstStride1 = gShape2 * dstStride2;
        int64_t dstStride0 = gShape1 * dstStride1;
		set_loop_size_outtoub(1ULL<<21 | 1ULL);		
        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t dstAddr0 = i * dstStride0;
            int64_t srcAddr0 = i * gStride0;
            for (uint32_t j = 0; j < gShape1; j++) {
                int64_t dstAddr1 = j * dstStride1;
                int64_t srcAddr1 = j * gStride1;
                for (uint32_t k = 0; k < gShape2; k++) {
                    dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                    srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                    TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, ubPad);
                }
            }
        }
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        constexpr uint32_t c0_size = 32;
        uint16_t nBurst = gShape1;
        uint32_t lenBurst = validRow * c0_size;
        uint64_t gmStride = gStride1 * sizeof(typename TileData::DType);
        uint32_t ubStride = TileData::Rows * c0_size;

        typename GlobalData::DType *srcAddrP = srcAddr;
        __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

        int64_t tileStride = gShape1 * gShape2 * TileData::Rows * gShape4;
		set_loop_size_outtoub(1ULL<<21 | 1ULL);
        for (uint32_t i = 0; i < gShape0; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * tileStride;
            TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, 0);
        }
    }
}

template <typename TileData, typename GlobalData>
__aicore__ void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                      (sizeof(typename TileData::DType) == 4),
        "Data type must be b8/b16/b32");
    static_assert(TileData::Loc == pto::Location::Vec, "Dst location must be Vec!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Source dtype must be same with dst dtype!");
    static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
        "Src and dst layout must be same!");

	//for static shape case, enforce the global tensor (tiled) shape matching with vecTile valid shape for xfer
	if constexpr ( TileData::Loc == pto::Location::Vec )
	{
		if constexpr ( TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox) )
		{
			if constexpr ( TileData::ValidCol>0 && GlobalData::staticShape[4]>0 )
			{
				static_assert( TileData::ValidCol==GlobalData::staticShape[4], "Src GlobalTensor Col and Tile ValidCol must be the same!");
			}
			if constexpr ( TileData::ValidRow>0 && GlobalData::staticShape[0]>0 &&
					GlobalData::staticShape[1]>0 && GlobalData::staticShape[2]>0 && GlobalData::staticShape[3]>0 )
			{
				constexpr const int mergedRows = GlobalData::staticShape[0]*GlobalData::staticShape[1]
												*GlobalData::staticShape[2]*GlobalData::staticShape[3];
				static_assert( TileData::ValidRow==mergedRows, "Src GlobalTensor Row Products and Tile ValidRow must be the same!");
			}			
		}
	}

    if constexpr (TileData::Loc == pto::Location::Vec) {
        TLoad<TileData, GlobalData>(dst.data(),
            src.data(),
            src.GetShape(0),
            src.GetShape(1),
            src.GetShape(2),
            src.GetShape(3),
            src.GetShape(4),
            src.GetStride(0),
            src.GetStride(1),
            src.GetStride(2),
            src.GetStride(3),
            src.GetStride(4),
            dst.GetValidRow(),
            dst.GetValidCol());
    }
}
}  // namespace pto
#endif  // TLOAD_HPP
