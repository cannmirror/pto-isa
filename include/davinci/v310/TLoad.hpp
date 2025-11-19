#ifndef TLOAD_HPP
#define TLOAD_HPP

namespace pto {

template <typename TileData>
__aicore__ constexpr auto getPadValue()
{
    if constexpr (std::is_same<typename TileData::DType, int64_t>::value ||
                  std::is_same<typename TileData::DType, uint64_t>::value) {
        return uint32_t(0);
    } else if constexpr (std::is_same<typename TileData::DType, float>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint32_t (0);
            case PadValue::Min: return uint32_t (0xff800000UL);				
            case PadValue::Max: return uint32_t (0x7f800000UL);
        }
    } else if constexpr (std::is_same<typename TileData::DType, int32_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint32_t (0);
            case PadValue::Min: return uint32_t (0xffffffffUL);				
            case PadValue::Max: return uint32_t (0x7fffffffUL);
        }
    } else if constexpr (std::is_same<typename TileData::DType, uint32_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero:
            case PadValue::Min: return uint32_t (0);
            case PadValue::Max: return uint32_t (0xffffffffUL);

        }
    } else if constexpr (std::is_same<typename TileData::DType, bfloat16_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint16_t (0);
            case PadValue::Min: return uint16_t (0xff80);
            case PadValue::Max: return uint16_t (0x7f80);
        }
    } else if constexpr (std::is_same<typename TileData::DType, half>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint16_t (0);
            case PadValue::Min: return uint16_t (0xfc00);				
            case PadValue::Max: return uint16_t (0x7c00);
        }
    } else if constexpr (std::is_same<typename TileData::DType, int16_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint16_t (0);
            case PadValue::Min: return uint16_t (0xffff);		
            case PadValue::Max: return uint16_t (0x7fff);
        }
    } else if constexpr (std::is_same<typename TileData::DType, uint16_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: 
            case PadValue::Min: return uint16_t (0);
            case PadValue::Max: return uint16_t (0xffff);
        }
    } else if constexpr (std::is_same<typename TileData::DType, int8_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: return uint8_t (0);
            case PadValue::Min: return uint8_t (0xff);		
            case PadValue::Max: return uint8_t (0x7f);
        }
    } else if constexpr (std::is_same<typename TileData::DType, uint8_t>::value) {
        switch (TileData::PadVal)
        {
            case PadValue::Null: 
            case PadValue::Zero: 
            case PadValue::Min: return uint8_t (0);
            case PadValue::Max: return uint8_t (0xff);
        }
    } else {
        static_assert(sizeof(TileData::DType) < 0, "TLOAD: Unsupported DType for PadValue!");
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadInstr(__ubuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
    uint32_t nBurst, uint32_t lenBurst, uint64_t gmStride, uint32_t ubStride, bool ubPad)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src),
                                    0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/,
                                    ubPad /*data select bit*/, 0 /*l2 cache ctl*/, gmStride, ubStride);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ uint16_t*>(dst), reinterpret_cast<__gm__ uint16_t*>(src),
                                    0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/,
                                    ubPad /*data select bit*/, 0 /*l2 cache ctl*/, gmStride, ubStride);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src),
                                    0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/,
                                    ubPad /*data select bit*/, 0 /*l2 cache ctl*/, gmStride, ubStride);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src),
                                    0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/,
                                    ubPad /*data select bit*/, 0 /*l2 cache ctl*/, gmStride, ubStride);
    }
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
        uint32_t nBurst = gShape3;
        uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
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
        uint32_t nBurst = gShape4;
        uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
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
        uint32_t nBurst = gShape1;
        uint32_t lenBurst = validRow * c0_size;
        uint64_t gmStride = gStride1 * sizeof(typename TileData::DType);
        uint32_t ubStride = TileData::Rows * c0_size;

        typename GlobalData::DType *srcAddrP = srcAddr;
        __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

        int64_t tileStride = gShape1 * TileData::Rows * gShape4;
		set_loop_size_outtoub(1ULL<<21 | 1ULL);
        for (uint32_t i = 0; i < gShape0; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * tileStride;
            TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, 0);
        }
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadCubeCheck()
{
    // support ND2NZ DN2NZ ND2ND DN2DN NZ2NZ
    static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
                      (((GlobalData::layout == pto::Layout::ND) &&
                          (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)))) ||
                      (((GlobalData::layout == pto::Layout::DN) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)))),
        "now only support ND2NZ DN2NZ ND2ND DN2DN NZ2NZ in current platform");

    // L1 space check
    static_assert(TileData::Rows <= 16384, "TileData::Rows must less than 16384 in L1");
    static_assert(TileData::Rows * TileData::Cols <= 512 * 1024, "TileData static shape must less than 512KB in L1");

    // ND2NZ or DN2NZ
    if constexpr ((GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN) &&
                  (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor))) {
        static_assert(TileData::SFractalSize == 512, "TileData SFractalSize must be 512 of NZ format in L1");
        static_assert(sizeof(typename TileData::DType) != 8, "DType not support b64 in ND2NZ or DN2NZ");
        // globaltensor only support 2 dim
        static_assert(
            GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
            "GlobalTensor input shape now only support 2 dim");
    }

    // NZ2NZ
    if constexpr ((GlobalData::layout == pto::Layout::NZ) &&
                  (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor))) {
        static_assert(BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType) == GlobalData::staticShape[4] &&
                          BLOCK_LEN == GlobalData::staticShape[3],
            "Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadCubeInstr(typename TileData::TileDType dst, typename GlobalData::DType *src,
    uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue)
{
    if constexpr (GlobalData::layout == Layout::ND) {
        copy_gm_to_cbuf_multi_nd2nz(dst, src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue, 0, false);  // ndNum = 1
    } else {
        copy_gm_to_cbuf_multi_dn2nz(dst, src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue, 0, false);  // dnNum = 1
    }
}
template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadCubeInstr(typename TileData::TileDType dst, typename GlobalData::DType* src,
                                          uint32_t nBurst, uint32_t lenBurst, uint64_t srcStride, uint32_t dstStride,
                                          uint32_t padCount)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount /*right padding count*/, true /*data select bit*/, 0 /*l2 cache ctl*/,
                                 srcStride, dstStride);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint16_t*>(dst), reinterpret_cast<__gm__ uint16_t*>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount /*right padding count*/, true /*data select bit*/, 0 /*l2 cache ctl*/,
                                 srcStride, dstStride);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount /*right padding count*/, true /*data select bit*/, 0 /*l2 cache ctl*/,
                                 srcStride, dstStride);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint32_t*>(dst), reinterpret_cast<__gm__ uint32_t*>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount * 2 /*right padding count*/, true /*data select bit*/, 0 /*l2 cache ctl*/,
                                 srcStride, dstStride);
    }
}

template <typename TileData, typename GlobalData>
__tf__ __aicore__ PTO_INLINE void TLoadCube(typename TileData::TileDType __out__ dst,
    typename GlobalData::DType __in__ *src, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
#if defined(__DAV_CUBE__)
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    using L1Type = typename TileData::TileDType;
    using GMType = typename GlobalData::DType;
    L1Type dstAddr = (L1Type)__cce_get_tile_ptr(dst);
    L1Type dstAddrP = dstAddr;
    GMType *srcAddrP = src;

    // ND2NZ or DN2NZ
    if constexpr ((GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN) &&
                  (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor))) {  // ND2NZ or DN2NZ , 大N小Z
        int64_t dstStride2 = gShape3 * TileData::Cols;
        int64_t dstStride1 = gShape2 * dstStride2;
        int64_t dstStride0 = gShape1 * dstStride1;
        uint16_t nValue = gShape3;
        uint32_t dValue = validCol;

        uint64_t loop1SrcStride = gStride3 * sizeof(GMType);  // whole shape 5
        if constexpr (GlobalData::layout == pto::Layout::DN) {
            loop1SrcStride = gStride4 * sizeof(GMType);
        }
        constexpr uint16_t ndNum = 1;
        uint16_t loop2DstStride = 1;
        uint16_t loop3DstStride = TileData::Rows;                           // unit is 32B
        uint16_t loop4DstStride = 0;                                        // because ndNum = 1
        uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48;  // MTE2_NZ_PARA[63:48]
        mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;          // MTE2_NZ_PARA[47:32]
        mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;          // MTE2_NZ_PARA[31:16]
        mte2NzPara |= static_cast<uint64_t>(ndNum);                         // MTE2_NZ_PARA[15:0]
        set_mte2_nz_para(mte2NzPara);                                       // only set once

        TLoadCubeInstr<TileData, GlobalData>(dstAddr, src, loop1SrcStride, nValue, dValue);

    } else if constexpr ((GlobalData::layout == pto::Layout::NZ) &&
                         (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor))) {  // NZ2NZ no padding
        uint32_t nBurst = gShape1;
        uint32_t lenBurst = validRow * BLOCK_BYTE_SIZE;
        uint64_t gmStride = gStride1 * sizeof(typename TileData::DType);
        uint32_t dstStride = TileData::Rows * BLOCK_BYTE_SIZE;

        int64_t tileStride = gShape1 * TileData::Rows * gShape4;
        set_loop_size_outtol1(1ULL << 21 | 1ULL);
        for (uint32_t i = 0; i < gShape0; i++) {
            srcAddrP = src + i * gStride0;
            dstAddrP = dstAddr + i * tileStride;
            TLoadCubeInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
        }

    } else if constexpr ((GlobalData::layout == pto::Layout::ND &&
                             (TileData::isRowMajor &
                                 (TileData::SFractal == SLayout::NoneBox)))) {  // ND2ND support cols padding
        uint32_t nBurst = gShape3;
        uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
        uint64_t gmStride = gStride3 * sizeof(typename TileData::DType);
        uint32_t dstStride = TileData::Cols * sizeof(typename TileData::DType);
        uint32_t padCount = 0;

        constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        uint32_t gapElement = (TileData::Cols - validCol);
        padCount = gapElement % blockSizeElem;
        set_pad_val_outtol1(getPadValue<TileData>());

        int64_t dstStride2 = gShape3 * TileData::Cols;
        int64_t dstStride1 = gShape2 * dstStride2;
        int64_t dstStride0 = gShape1 * dstStride1;

        uint64_t loop2 = gShape1;
        uint64_t loop1 = gShape2;
        uint64_t loop2_src_stride = gStride1 * sizeof(typename TileData::DType);
        uint64_t loop1_src_stride = gStride2 * sizeof(typename TileData::DType);
        uint64_t loop2_dst_stride = dstStride1 * sizeof(typename TileData::DType);
        uint64_t loop1_dst_stride = dstStride2 * sizeof(typename TileData::DType);
        set_loop2_stride_outtol1(loop2_dst_stride << 40 | loop2_src_stride);
        set_loop1_stride_outtol1(loop1_dst_stride << 40 | loop1_src_stride);
        set_loop_size_outtol1(loop2 << 21 | loop1);
        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t dstAddr0 = i * dstStride0;
            int64_t srcAddr0 = i * gStride0;
            dstAddrP = dstAddr + dstAddr0;
            srcAddrP = src + srcAddr0;
            TLoadCubeInstr<TileData, GlobalData>(
                dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
        }
    } else if constexpr (GlobalData::layout == pto::Layout::DN &&
                         (!TileData::isRowMajor &
                             (TileData::SFractal == SLayout::NoneBox))) {  // dn support rows padding
        uint32_t nBurst = gShape4;
        uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
        uint64_t gmStride = gStride4 * sizeof(typename TileData::DType);
        uint32_t dstStride = TileData::Rows * sizeof(typename TileData::DType);
        uint32_t padCount = 0;

        constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        uint32_t gapElement = (TileData::Rows - validRow);
        padCount = gapElement % blockSizeElem;
        set_pad_val_outtol1(getPadValue<TileData>());

        int64_t dstStride2 = gShape4 * TileData::Rows;
        int64_t dstStride1 = gShape2 * dstStride2;
        int64_t dstStride0 = gShape1 * dstStride1;

        uint64_t loop2 = gShape1;
        uint64_t loop1 = gShape2;
        uint64_t loop2_src_stride = gStride1 * sizeof(typename TileData::DType);
        uint64_t loop1_src_stride = gStride2 * sizeof(typename TileData::DType);
        uint64_t loop2_dst_stride = dstStride1 * sizeof(typename TileData::DType);
        uint64_t loop1_dst_stride = dstStride2 * sizeof(typename TileData::DType);
        set_loop2_stride_outtol1(loop2_dst_stride << 40 | loop2_src_stride);
        set_loop1_stride_outtol1(loop1_dst_stride << 40 | loop1_src_stride);
        set_loop_size_outtol1(loop2 << 21 | loop1);
        for (uint32_t i = 0; i < gShape0; i++) {
            int64_t dstAddr0 = i * dstStride0;
            int64_t srcAddr0 = i * gStride0;
            dstAddrP = dstAddr + dstAddr0;
            srcAddrP = src + srcAddr0;
            TLoadCubeInstr<TileData, GlobalData>(
                dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
        }
    }
#endif
}

template <typename TileData, typename GlobalData>
__aicore__ void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                      (sizeof(typename TileData::DType) == 4) || (sizeof(typename TileData::DType) == 8),
        "Data type must be b8/b16/b32/b64");
    if constexpr (std::is_same<typename TileData::DType, int64_t>::value ||
                  std::is_same<typename TileData::DType, uint64_t>::value) {
        static_assert(TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero,
                      "TileData::PadVal only support Null or Zero in B64 mode");
    }
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Source dtype must be same with dst dtype!");

	//for static shape case, enforce the global tensor (tiled) shape matching with vecTile valid shape for xfer
	if constexpr ( TileData::Loc == pto::Location::Vec )
	{
        static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                          (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
            "Src and dst layout must be same!");
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
    } else if constexpr(TileData::Loc == pto::Location::Mat) {
        TLoadCubeCheck<TileData, GlobalData>();
        TLoadCube<TileData, GlobalData>(dst.data(),
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
