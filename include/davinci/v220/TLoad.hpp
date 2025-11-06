#ifndef TLOAD_HPP
#define TLOAD_HPP

namespace pto {
    template <typename TileData>
    __aicore__ constexpr auto getPadValue()
    {
        if constexpr (std::is_same<typename TileData::DType, float>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint32_t(0);
                case PadValue::Min: return uint32_t(0xff800000UL);
                case PadValue::Max: return uint32_t(0x7f800000UL);
            }
        } else if constexpr (std::is_same<typename TileData::DType, int32_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint32_t(0);
                case PadValue::Min: return uint32_t(0xffffffffUL);
                case PadValue::Max: return uint32_t(0x7fffffffUL);
            }
        } else if constexpr (std::is_same<typename TileData::DType, uint32_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero:
                case PadValue::Min: return uint32_t(0);
                case PadValue::Max: return uint32_t(0xffffffffUL);
            }
        } else if constexpr (std::is_same<typename TileData::DType, bfloat16_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint16_t(0);
                case PadValue::Min: return uint16_t(0xff80);
                case PadValue::Max: return uint16_t(0x7f80);
            }
        } else if constexpr (std::is_same<typename TileData::DType, half>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint16_t(0);
                case PadValue::Min: return uint16_t(0xfc00);
                case PadValue::Max: return uint16_t(0x7c00);
            }
        } else if constexpr (std::is_same<typename TileData::DType, int16_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint16_t(0);
                case PadValue::Min: return uint16_t(0xffff);
                case PadValue::Max: return uint16_t(0x7fff);
            }
        } else if constexpr (std::is_same<typename TileData::DType, uint16_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero:
                case PadValue::Min: return uint16_t(0);
                case PadValue::Max: return uint16_t(0xffff);
            }
        } else if constexpr (std::is_same<typename TileData::DType, int8_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero: return uint8_t(0);
                case PadValue::Min: return uint8_t(0xff);
                case PadValue::Max: return uint8_t(0x7f);
            }
        } else if constexpr (std::is_same<typename TileData::DType, uint8_t>::value) {
            switch (TileData::PadVal)
            {
                case PadValue::Null:
                case PadValue::Zero:
                case PadValue::Min: return uint8_t(0);
                case PadValue::Max: return uint8_t(0xff);
            }
        } else {
            static_assert(sizeof(TileData::DType) < 0, "TLOAD: Unsupported DType for PadValue");
        }
    }
    
    template <typename TileData, typename GlobalData>
    __aicore__ PTO_INLINE void TLoadInstr(__ubuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
        uint16_t nBurst, uint16_t lenBurst, uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
    {
        if constexpr (sizeof(typename TileData::DType) == 1) {
            copy_gm_to_ubuf_align_b8(dst,
                src,
                0 /* sid */,
                nBurst,
                lenBurst,
                0 /* left padding count */,
                ubPad /* right padding count*/,
                gmGap,
                ubGap);
        } else if constexpr (sizeof(typename TileData::DType) == 2) {
            copy_gm_to_ubuf_align_b16(dst,
                src,
                0 /* sid */,
                nBurst,
                lenBurst,
                0 /* left padding count */,
                ubPad /* right padding count*/,
                gmGap,
                ubGap);
        } else if constexpr (sizeof(typename TileData::DType) == 4) {
            copy_gm_to_ubuf_align_b32(dst,
                src,
                0 /* sid */,
                nBurst,
                lenBurst,
                0 /* left padding count */,
                ubPad /* right padding count*/,
                gmGap,
                ubGap);
        }
    }

    template <typename TileData, typename GlobalData>
    __tf__ aicore void TLoad(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
        int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
        int gStride3, int gStride4, int validRow, int validCol)
    {
        constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::TileDType);
        __ubuf__ typename TileData::DType *dstAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        typename GlobalData::DType *srcAddr = src;

        if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            static_assert(TileData::Rows<4096, "TLOAD: Rows>=4095 not supported in A2/A3");
            uint16_t nBurst = gShape3;
            uint16_t lenBurst = validCol * sizeof(typename TileData::DType);
            uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);
            // assert(gmGapValue > 0xffffffffUL, "TLOAD: Gap > 32bit is not supported in A2/A3");  /* FIXME runtime assert function */
            uint32_t gmGap = (uint32_t) gmGapValue;
            uint32_t ubGapElement = (TileData::Cols - validCol);
            uint32_t ubGap = ubGapElement / blockSizeElem;
            uint32_t ubPad = 0;
            if (TileData::PadVal != PadValue::Null) {
                ubPad = ubGapElement % blockSizeElem;
                set_mov_pad_val(getPadValue<TileData>());
            }
            typename Global::DType *srcAddrP = srcAddr;
            __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

            int64_t dstStride2 = gShape3 * TileData::Cols;
            int64_t dstStride1 = gShape2 * dstStride2;
            int64_t dstStride0 = gShape1 * dstStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t dstAddr0 = i * dstStride0;
                int64_t srcAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t dstAddr1 = j * dstStride1;
                    int64_t srcAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                        srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                        TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
                    }
                }
            }
        } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
            uint16_t nBurst = gShape4;
            uint16_t lenBurst = validRow * sizeof(typename TileData::DType);
            uint64_t gmGapValue = (gStride4 - gShape3) * sizeof(typename TileData::DType);
            // assert(gmGapValue > 0xffffffffUL, "TLOAD: Gap > 32bit is not supported in A2/A3");  /* FIXME runtime assert function */
            uint32_t gmGap = (uint32_t) gmGapValue;
            uint32_t ubGapElement = (TileData::Rows - gShape3);
            uint32_t ubGap = ubGapElement / blockSizeElem;
            uint32_t ubPad = 0;
            if (TileData::PadVal != PadValue::Null) {
                ubPad = ubGapElement % blockSizeElem;
                set_mov_pad_val(getPadValue<TileData>());
            }
            typename Global::DType *srcAddrP = srcAddr;
            __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

            int64_t dstStride2 = gShape4 * TileData::Rows;
            int64_t dstStride1 = gShape2 * dstStride2;
            int64_t dstStride0 = gShape1 * dstStride1;
            for (uint32_t i = 0; i < gShape0; i++) {
                int64_t dstAddr0 = i * dstStride0;
                int64_t srcAddr0 = i * gStride0;
                for (uint32_t j = 0; j < gShape1; j++) {
                    int64_t dstAddr1 = j * dstStride1;
                    int64_t srcAddr1 = j * gStride1;
                    for (uint32_t k = 0; k < gShape2; k++) {
                        dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                        srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                        TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
                    }
                }
            }
        } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
            constexpr uint32_t c0_size = 32;
            uint16_t nBurst = gShape1;
            uint32_t lenBurst = validRow * c0_size;
            uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
            uint32_t ubGap = TileData::Rows - validRow;

            typename GlobalData::DType *srcAddrP = srcAddr;
            __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

            int64_t tileStride = gShape1 * gShape2 * TileData::Rows * gShape4;

            for (uint32_t i = 0; i < gShape0; i++) {
                srcAddrP = srcAddr + i * gStride0;
                dstAddrP = dstAddr + i * tileStride;
                TLoadInstr<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, 0);
            }
        }
    }

    template <typename TileData, typename GlobalData>
    __aicore__ void TLOAD_IMPL(TileData &dst, GlobalData &src)
    {
        static_assert((sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
                      (sizeof(typename TileData::DType) == 4), "Data type must be b8/16/32");
        static_assert(TileData::Loc == pto::Loaction::Vec, "Dst location must be Vec!");
        static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                      "Source dtype must be same with dst dtype");
        static_assert(((GlobalData::layout == pto::Layout::ND) &&
                      (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::DN) &&
                      (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                      ((GlobalData::layout == pto::Layout::NZ) &&
                      (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
                      "Src and dst layout must be same");
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
}
#endif