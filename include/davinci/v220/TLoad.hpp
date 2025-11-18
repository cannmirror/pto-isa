#ifndef TLOAD_HPP
#define TLOAD_HPP

namespace pto {
template <typename TileData>
__aicore__ constexpr auto getPadValue()
{
    if constexpr (std::is_same<typename TileData::DType, int64_t>::value ||
        std::is_same<typename TileData::DType, uint64_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint32_t(0);
            default:
                static_assert((TileData::PadVal == PadValue::Null) || (TileData::PadVal == PadValue::Zero),
                    "TLOAD: only PadNull and PadZero is supported for b8!");
        }
    } else if constexpr (std::is_same<typename TileData::DType, float>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint32_t(0);
            case PadValue::Min:
                return uint32_t(0xff800000UL);
            case PadValue::Max:
                return uint32_t(0x7f800000UL);
        }
    } else if constexpr (std::is_same<typename TileData::DType, int32_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint32_t(0);
            case PadValue::Min:
                return uint32_t(0xffffffffUL);
            case PadValue::Max:
                return uint32_t(0x7fffffffUL);
        }
    } else if constexpr (std::is_same<typename TileData::DType, uint32_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
            case PadValue::Min:
                return uint32_t(0);
            case PadValue::Max:
                return uint32_t(0xffffffffUL);
        }
    } else if constexpr (std::is_same<typename TileData::DType, bfloat16_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint16_t(0);
            case PadValue::Min:
                return uint16_t(0xff80);
            case PadValue::Max:
                return uint16_t(0x7f80);
        }
    } else if constexpr (std::is_same<typename TileData::DType, half>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint16_t(0);
            case PadValue::Min:
                return uint16_t(0xfc00);
            case PadValue::Max:
                return uint16_t(0x7c00);
        }
    } else if constexpr (std::is_same<typename TileData::DType, int16_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint16_t(0);
            case PadValue::Min:
                return uint16_t(0xffff);
            case PadValue::Max:
                return uint16_t(0x7fff);
        }
    } else if constexpr (std::is_same<typename TileData::DType, uint16_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
            case PadValue::Min:
                return uint16_t(0);
            case PadValue::Max:
                return uint16_t(0xffff);
        }
    } else if constexpr (std::is_same<typename TileData::DType, int8_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
                return uint8_t(0);
            case PadValue::Min:
                return uint8_t(0xff);
            case PadValue::Max:
                return uint8_t(0x7f);
        }
    } else if constexpr (std::is_same<typename TileData::DType, uint8_t>::value) {
        switch (TileData::PadVal) {
            case PadValue::Null:
            case PadValue::Zero:
            case PadValue::Min:
                return uint8_t(0);
            case PadValue::Max:
                return uint8_t(0xff);
        }
    } else {
        static_assert(sizeof(TileData::DType) < 0, "TLOAD: Unsupported DType for PadValue!");
    }
}

template <typename TileData>
__aicore__ PTO_INLINE constexpr pto::Layout GetTileLayout()
{
    if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)) {
        return pto::Layout::ND;
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        return pto::Layout::DN;
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        return pto::Layout::NZ;
    } else {
        static_assert("Unsupport Layout!");
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadInstrGm2ub(__ubuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
    uint16_t nBurst, uint32_t lenBurst, uint32_t gmGap, uint32_t ubGap, uint32_t ubPad)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);
    } else if constexpr (sizeof(typename TileData::DType) == 8) {
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, 0, ubPad * 2, gmGap, ubGap);
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadNd2nzInstr(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
    uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
    uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_gm_to_cbuf_multi_nd2nz_b8(dst,
            src,
            0,                   // uint8_t
            ndNum,               // uint16_t
            nValue,              // uint16_t
            dValue,              // uint16_t
            srcNdMatrixStride,   // uint16_t
            srcDValue,           // uint16_t
            dstNzC0Stride,       // uint16_t
            dstNzNStride,        // uint16_t
            dstNzMatrixStride);  // uint16_t
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_gm_to_cbuf_multi_nd2nz_b16(dst,
            src,
            0,
            ndNum,
            nValue,
            dValue,
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride);
    } else if constexpr (sizeof(typename TileData::DType) == 4) {
        copy_gm_to_cbuf_multi_nd2nz_b32s(dst,
            src,
            0,
            ndNum,
            nValue,
            dValue,
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride);
    }
}

template <typename TileData, typename GlobalData>
__aicore__ PTO_INLINE void TLoadInstrGm2L1(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
    uint16_t nBurst, uint16_t lenBurst, uint16_t gmGap, uint16_t l1Gap)
{
    copy_gm_to_cbuf(dst, src, (uint8_t)0, nBurst, lenBurst, gmGap, l1Gap, (pad_t)0);
}

template <typename TileData, typename GlobalData>
__tf__ __aicore__ void TLoadGm2ub(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    __ubuf__ typename TileData::DType *dstAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        static_assert(TileData::Rows < 4096, "TLOAD: Rows>=4095 not supported in A2/A3");
        uint16_t nBurst = gShape3;
        uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
        uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);
        uint32_t gmGap = (uint32_t)gmGapValue;
        uint32_t ubGapElement = (TileData::Cols - validCol);
        uint32_t ubGap = ubGapElement / blockSizeElem;
        uint32_t ubPad = 0;
        if (TileData::PadVal != PadValue::Null) {
            ubPad = ubGapElement % blockSizeElem;
            set_mov_pad_val(getPadValue<TileData>());
        }
        typename GlobalData::DType *srcAddrP = srcAddr;
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
                    TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
                }
            }
        }
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        uint16_t nBurst = gShape4;
        uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
        uint64_t gmGapValue = (gStride4 - gShape3) * sizeof(typename TileData::DType);
        uint32_t gmGap = (uint32_t)gmGapValue;
        uint32_t ubGapElement = (TileData::Rows - gShape3);
        uint32_t ubGap = ubGapElement / blockSizeElem;
        uint32_t ubPad = 0;
        if (TileData::PadVal != PadValue::Null) {
            ubPad = ubGapElement % blockSizeElem;
            set_mov_pad_val(getPadValue<TileData>());
        }
        typename GlobalData::DType *srcAddrP = srcAddr;
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
                    TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
                }
            }
        }
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        constexpr uint32_t c0_size = 32;
        static_assert(GlobalData::staticShape[3] == 16 &&
                          GlobalData::staticShape[4] == c0_size / sizeof(typename TileData::DType),
            "When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        uint16_t nBurst = gShape1;
        uint32_t lenBurst = validRow * c0_size;
        uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
        uint32_t ubGap = TileData::Rows - validRow;

        typename GlobalData::DType *srcAddrP = srcAddr;
        __ubuf__ typename TileData::DType *dstAddrP = dstAddr;

        // validRow = gShape2 * gShape3, validCol = gShape1 * gShape4
        int64_t tileStride = TileData::Rows * gShape1 * gShape4;

        for (uint32_t i = 0; i < gShape0; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * tileStride;
            TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, 0);
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ __aicore__ void TLoadGm2L1(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    if constexpr (GetTileLayout<TileData>() == pto::Layout::ND) {
        uint16_t nBurst = gShape3;
        uint16_t lenBurst = validCol / blockSizeElem;
        uint16_t gmGap = (gStride3 - gShape4) / blockSizeElem;
        uint16_t l1Gap = (TileData::Cols - validCol) / blockSizeElem;

        typename GlobalData::DType *srcAddrP = srcAddr;
        __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

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
                    TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
                }
            }
        }
    } else if constexpr (GetTileLayout<TileData>() == pto::Layout::DN) {
        uint16_t nBurst = gShape4;
        uint16_t lenBurst = validRow / blockSizeElem;
        uint16_t gmGap = (gStride4 - gShape3) / blockSizeElem;
        uint16_t l1Gap = (TileData::Rows - gShape3) / blockSizeElem;

        typename GlobalData::DType *srcAddrP = srcAddr;
        __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

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
                    TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, l1Gap);
                }
            }
        }
    } else if constexpr (GetTileLayout<TileData>() == pto::Layout::NZ) {
        static_assert(GlobalData::staticShape[3] == 16 &&
                          GlobalData::staticShape[4] == 32 / sizeof(typename TileData::DType),
            "When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        uint16_t nBurst = gShape1;
        uint32_t lenBurst = validRow;
        uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType) / BLOCK_BYTE_SIZE;
        uint32_t ubGap = TileData::Rows - validRow;

        typename GlobalData::DType *srcAddrP = srcAddr;
        __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

        int64_t tileStride = TileData::Rows * gShape1 * gShape4;

        for (uint32_t i = 0; i < gShape0; i++) {
            srcAddrP = srcAddr + i * gStride0;
            dstAddrP = dstAddr + i * tileStride;
            TLoadInstrGm2L1<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap);
        }
    }
}

template <typename TileData, typename GlobalData>
__tf__ __aicore__ void TLoadGm2L1Nd2nz(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, int validRow, int validCol)
{
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;
    uint16_t ndNum = 1;
    uint16_t nValue = gShape3;
    uint16_t dValue = gShape4;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = gStride3;
    uint16_t dstNzC0Stride = TileData::Rows;
    uint16_t dstNzNStride = 1;
    uint16_t dstNzMatrixStride = 1;

    TLoadNd2nzInstr<TileData, GlobalData>(dstAddr,
        srcAddr,
        ndNum,
        nValue,
        dValue,
        srcNdMatrixStride,
        srcDValue,
        dstNzC0Stride,
        dstNzNStride,
        dstNzMatrixStride);
}

template <typename TileData, typename GlobalData>
__aicore__ void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float/int64_t/uint64_t/!");
    static_assert(
        TileData::Loc == pto::Location::Vec || TileData::Loc == pto::Location::Mat, "Dst location must be Vec or Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Source dtype must be same with dst dtype!");

    if constexpr (std::is_same_v<typename TileData::DType, int64_t> ||
        std::is_same_v<typename TileData::DType, uint64_t>) {
        static_assert(GlobalData::layout == GetTileLayout<TileData>(),
            "TLOAD only support ND2ND/DN2DN for b8!");
        static_assert((GlobalData::layout == pto::Layout::ND) ||
            (GlobalData::layout == pto::Layout::DN),
            "TLOAD only support ND2ND/DN2DN for b8!");
    }

    if constexpr (TileData::Loc == pto::Location::Vec) {
        static_assert(GlobalData::layout == GetTileLayout<TileData>(),
            "TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ!");
        TLoadGm2ub<TileData, GlobalData>(dst.data(),
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
    } else if constexpr (TileData::Loc == pto::Location::Mat) {
        static_assert(GlobalData::layout == GetTileLayout<TileData>() ||
                          (GlobalData::layout == pto::Layout::ND && GetTileLayout<TileData>() == pto::Layout::NZ),
            "TLOAD(MatTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ/ND2NZ!");
        if constexpr (GlobalData::layout == GetTileLayout<TileData>()) {
            TLoadGm2L1<TileData, GlobalData>(dst.data(),
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
        } else if constexpr (GlobalData::layout == pto::Layout::ND && GetTileLayout<TileData>() == pto::Layout::NZ) {
            static_assert(
                GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
                "GlobalTensor ony support 2 dim when ND2NZ!");
            static_assert(TileData::SFractalSize == 512, "TileData ony support SFractalSize = 512Bytes!");
            TLoadGm2L1Nd2nz<TileData, GlobalData>(dst.data(),
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
}  // namespace pto
#endif  // TLOAD_HPP