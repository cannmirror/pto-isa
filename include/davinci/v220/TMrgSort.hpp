#ifndef TMRGSORT_HPP
#define TMRGSORT_HPP

#include "common/constants.hpp"
#define TRUE 1
#define FALSE 0
#define STRUCTSIZE 8
#define UBSIZE 262144  // 256 * 1024 B
#define ELEMSIZE 4

namespace pto
{
    struct MrgSortExecutedNumList {
        uint16_t mrgSortList0;
        uint16_t mrgSortList1;
        uint16_t mrgSortList2;
        uint16_t mrgSortList3;
    };
    
    __aicore__ PTO_INLINE void GetExhaustedDat(uint16_t mrgSortList0, uint16_t mrgSortList1,
                                               uint16_t mrgSortList2, uint16_t mrgSortList3)
    {
        int64_t mrgSortResult = get_vms4_sr();
        constexpr uint64_t resMask = 0xFFFF;
        // VMS4_SR[15:0], number of finished region proposals in list0
        mrgSortList0 = static_cast<uint64_t>(mrgSortResult) & resMask;
        constexpr uint64_t sortList1Bit = 16;
        // VMS4_SR[31:16], number of finished region proposals in list1
        mrgSortList1 = static_cast<uint64_t>(mrgSortResult >> sortList1Bit) & resMask;
        constexpr uint64_t sortList2Bit = 33;
        // VMS4_SR[47:32], number of finished region proposals in list2
        mrgSortList2 = static_cast<uint64_t>(mrgSortResult >> sortList2Bit) & resMask;
        constexpr uint64_t sortList3Bit = 48;
        // VMS4_SR[63:48], number of finished region proposals in list3
        mrgSortList3 = static_cast<uint64_t>(mrgSortResult >> sortList3Bit) & resMask;
    }

    template <typename DstTileData, typename TmpTileData,
              typename Src0TileData, typename Src1TileData, typename Src2TileData,
              typename Src3TileData, bool exhausted, unsigned listNum>
    __tf__ __aicore__ void TMrgsort(typename DstTileData::TileDType __out__ dst,
                                    typename TmpTileData::TileDType __out__ tmp,
                                    typename Src0TileData::TileDType __in__ src0,
                                    typename Src0TileData::TileDType __in__ src1,
                                    typename Src0TileData::TileDType __in__ src2,
                                    typename Src0TileData::TileDType __in__ src3,
                                    unsigned dstCol,
                                    uint16_t &mrgSortList0, uint16_t &mrgSortList1,
                                    uint16_t &mrgSortList2, uint16_t &mrgSortList3,
                                    unsigned src0Col, unsigned src1Col, unsigned src2Col, unsigned src3Col) {
        __ubuf__ typename DstTileData::DType *dstPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename DstTileData::DType *tmpPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(tmp);
        __ubuf__ typename DstTileData::DType *src0Ptr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename DstTileData::DType *src1Ptr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src1);
        __ubuf__ typename DstTileData::DType *src2Ptr = nullptr;
        __ubuf__ typename DstTileData::DType *src3Ptr = nullptr;
        if constexpr (listNum >= 3) {
            src2Ptr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src2);
        }
        if constexpr (listNum == 4) {
            src3Ptr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src3);
        }
        if constexpr (listNum == 2) {
            uint64_t config = 0;
            config |= uint64_t(1);  // Xt[7:0]: repeat time
            config |= (uint64_t(0x0011) << 8);  // Xt[11:8]: 4-bit mask signal
            if constexpr (exhausted == TRUE) {
                config |= (uint64_t(0b1) << 12);  // Xt[12]: 1-enable input list exhausted suspension
            }
            if constexpr (exhausted == FALSE) {
                config |= (uint64_t(0b0) << 12);  // Xt[12]: 0-enable input list exhausted suspension
            }
            
            // 每次计算的数据
            uint64_t count = 0;
            count |= (uint64_t(src0Col));
            count |= (uint64_t(src1Col) << 16);

            __ubuf__ typename DstTileData::DType *addr_array[4] = {(__ubuf__ typename DstTileData::DType *)(src0Ptr),
                (__ubuf__ typename DstTileData::DType *)(src1Ptr)};
            vmrgsort4(tmpPtr, addr_array, count, config);
        }
        if constexpr (listNum == 3) {
            uint64_t config = 0;
            config |= uint64_t(1);  // Xt[7:0]: repeat time
            config |= (uint64_t(0x0111) << 8);  // Xt[11:8]: 4-bit mask signal
            if constexpr (exhausted == TRUE) {
                config |= (uint64_t(0b1) << 12);  // Xt[12]: 1-enable input list exhausted suspension
            }
            if constexpr (exhausted == FALSE) {
                config |= (uint64_t(0b0) << 12);  // Xt[12]: 0-enable input list exhausted suspension
            }
            
            // 每次计算的数据
            uint64_t count = 0;
            count |= (uint64_t(src0Col));
            count |= (uint64_t(src1Col) << 16);
            count |= (uint64_t(src2Col) << 32);

            __ubuf__ typename DstTileData::DType *addr_array[4] = {(__ubuf__ typename DstTileData::DType *)(src0Ptr),
                (__ubuf__ typename DstTileData::DType *)(src1Ptr), (__ubuf__ typename DstTileData::DType *)(src2Ptr)};
            vmrgsort4(tmpPtr, addr_array, count, config);
        }
        if constexpr (listNum == 4) {
            uint64_t config = 0;
            config |= uint64_t(1);  // Xt[7:0]: repeat time
            config |= (uint64_t(0x1111) << 8);  // Xt[11:8]: 4-bit mask signal
            if constexpr (exhausted == TRUE) {
                config |= (uint64_t(0b1) << 12);  // Xt[12]: 1-enable input list exhausted suspension
            }
            if constexpr (exhausted == FALSE) {
                config |= (uint64_t(0b0) << 12);  // Xt[12]: 0-enable input list exhausted suspension
            }
            
            // 每次计算的数据
            uint64_t count = 0;
            count |= (uint64_t(src0Col));
            count |= (uint64_t(src1Col) << 16);
            count |= (uint64_t(src2Col) << 32);
            count |= (uint64_t(src3Col) << 48);

            __ubuf__ typename DstTileData::DType *addr_array[4] = {(__ubuf__ typename DstTileData::DType *)(src0Ptr),
                (__ubuf__ typename DstTileData::DType *)(src1Ptr), (__ubuf__ typename DstTileData::DType *)(src2Ptr),
                (__ubuf__ typename DstTileData::DType *)(src3Ptr)};
            vmrgsort4(tmpPtr, addr_array, count, config);
        }
        if constexpr (exhausted == TRUE) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            GetExhaustedData(mrgSortList0, mrgSortList1, mrgSortList2, mrgSortList3);
        }
        pipe_barrier(PIPE_V);
        // (dst, src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
        copy_ubuf_to_ubuf((__ubuf__ void *)dstPtr, (__ubuf__ void *)tmpPtr, 0, 1, dstCol, 0, 0);
    }

    // 新增输入单个大块Tile输入
    template <typename DstTileData, typename SrcTileData>
    __tf__ __aicore__ void TMrgsort(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src,
                                    uint32_t numStrcutures, uint8_t repeatTimes) {
        __ubuf__ typename DstTileData::DType *dstPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename SrcTileData::DType *srcPtr = (__ubuf__ typename SrcTileData::DType *)__cce_get_tile_ptr(src);

        uint64_t config = 0;
        config |= uint64_t(repeatTime);  // Xt[7:0]: repeat time
        config |= (uint64_t(0b1111) << 8);  // Xt[11:8]: 4-bit mask signal
        config |= (uint64_t(0b0) << 12);  // Xt[12]: 1-enable input list exhausted suspension

        // 每次计算的数据
        uint64_t count = 0;
        count |= (uint64_t(numStrcutures));
        count |= (uint64_t(numStrcutures) << 16);
        count |= (uint64_t(numStrcutures) << 32);
        count |= (uint64_t(numStrcutures) << 48);

        unsigned offset = numStrcutures * STRUCTSIZE / sizeof(typename DstTileData::DType);

        __ubuf__ typename DstTileData::DType *addr_array[4] = {(__ubuf__ typename SrcTileData::DType *)(srcPtr),
            (__ubuf__ typename SrcTileData::DType *)(srcPtr + offset), (__ubuf__ typename SrcTileData::DType *)(srcPtr + 2 * offset),
            (__ubuf__ typename SrcTileData::DType *)(srcPtr + offset * 3)};
        vmrgsort4(dstPtr, addr_array, count, config);
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              typename Src2TileData, typename Src3TileData, unsigned listNum>
    __aicore__ PTO_INLINE void CheckOverMemory() {
        constexpr unsigned totalSrcCols = Src0TileData::Cols +
                                          (listNum >= 2 ? Src1TileData::Cols : 0) +
                                          (listNum >= 3 ? Src2TileData::Cols : 0) +
                                          (listNum >= 4 ? Src3TileData::Cols : 0);

        // tmpCols在 listNum == 1 时为0，否则等于totalSrcCols
        constexpr unsigned tmpCols = (listNum == 1) ? 0 : totalSrcCols;
        constexpr unsigned dstCols = DstTileData::Cols;
        constexpr size_t srcSize = totalSrcCols * ELEMSIZE;
        constexpr size_t tmpSize = tmpCols * ELEMSIZE;
        constexpr size_t dstSize = dstCols * ELEMSIZE;

        static_assert(srcSize + tmpSize + dstSize < UBSIZE,
                      "ERROR: Total memory usage exceeds UB limit!");
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              typename Src2TileData, typename Src3TileData>
    __aicore__ PTO_INLINE void CheckStatic() 
    {
        static_assert((std::is_same<typename DstTileData::DType, half>::value) ||
                      (std::is_same<typename DstTileData::DType, float>::value),
                      "expect half/float");
        static_assert((std::is_same<typename DstTileData::DType, typename TmpTileData::DType>::value) &&
                      (std::is_same<typename DstTileData::DType, typename Src0TileData::DType>::value) &&
                      (std::is_same<typename DstTileData::DType, typename Src1TileData::DType>::value) &&
                      (std::is_same<typename DstTileData::DType, typename Src2TileData::DType>::value) &&
                      (std::is_same<typename DstTileData::DType, typename Src3TileData::DType>::value),
                      "expect same size");
        static_assert((DstTileData::Loc == Location::Vec) && (TmpTileData::Loc == Location::Vec) &&
                      (Src0TileData::Loc == Location::Vec) && (Src1TileData::Loc == Location::Vec) &&
                      (Src2TileData::Loc == Location::Vec) && (Src3TileData::Loc == Location::Vec),
                      "location must be Vec!");
        static_assert((DstTileData::Rows == 1) && (TmpTileData::Rows == 1) && (Src0TileData::Rows == 1) &&
                      (Src1TileData::Rows == 1) && (Src2TileData::Rows == 1),
                      "expect single row");
        static_assert((DstTileData::isRowMajor && TmpTileData::isRowMajor && Src0TileData::isRowMajor &&
                       Src1TileData::isRowMajor && Src2TileData::isRowMajor && Src3TileData::isRowMajor),
                      "expect row major");
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              typename Src2TileData, typename Src3TileData, bool exhausted>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecuteNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1,
                                        Src2TileData &src2, Src3TileData &src3) {
        CheckStatic<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData>();
        CheckOverMemory<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, 4>();
        unsigned dstCol = dst.GetValidCol();
        unsigned eleNum = STRUCTSIZE / sizeof(typename DstTileData::DType);
        unsigned src0Col = src0.GetValidCol() / eleNum;
        unsigned src1Col = src1.GetValidCol() / eleNum;
        unsigned src2Col = src2.GetValidCol() / eleNum;
        unsigned src3Col = src3.GetValidCol() / eleNum;
        TMrgsort<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, exhausted, 4>
            (dst.data(), tmp.data(), src0.data(), src1.data(), src2.data(), src3.data(), dstCol,
             executedNumList.mrgSortList0, executedNumList.mrgSortList1,
             executedNumList.mrgSortList2, executedNumList.mrgSortList3,
             src0Col, src1Col, src2Col, src3Col);
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              typename Src2TileData, bool exhausted>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecuteNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1,
                                        Src2TileData &src2) {
        CheckStatic<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src0TileData>();
        CheckOverMemory<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src0TileData, 3>();
        unsigned dstCol = dst.GetValidCol();
        unsigned eleNum = STRUCTSIZE / sizeof(typename DstTileData::DType);
        unsigned src0Col = src0.GetValidCol() / eleNum;
        unsigned src1Col = src1.GetValidCol() / eleNum;
        unsigned src2Col = src2.GetValidCol() / eleNum;
        TMrgsort<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src2TileData, exhausted, 3>
            (dst.data(), tmp.data(), src0.data(), src1.data(), src2.data(), nullptr, dstCol,
             executedNumList.mrgSortList0, executedNumList.mrgSortList1,
             executedNumList.mrgSortList2, executedNumList.mrgSortList3,
             src0Col, src1Col, src2Col, 0);
    }

    template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
              bool exhausted>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecuteNumList &executedNumList, TmpTileData &tmp,
                                        Src0TileData &src0, Src1TileData &src1) {
        CheckStatic<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src0TileData, Src0TileData>();
        CheckOverMemory<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src0TileData, Src0TileData, 2>();
        unsigned dstCol = dst.GetValidCol();
        unsigned eleNum = STRUCTSIZE / sizeof(typename DstTileData::DType);
        unsigned src0Col = src0.GetValidCol() / eleNum;
        unsigned src1Col = src1.GetValidCol() / eleNum;
        TMrgsort<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src1TileData, Src1TileData, exhausted, 2>
            (dst.data(), tmp.data(), src0.data(), src1.data(), nullptr, nullptr, dstCol,
             executedNumList.mrgSortList0, executedNumList.mrgSortList1,
             executedNumList.mrgSortList2, executedNumList.mrgSortList3,
             src0Col, src1Col, 0, 0);
    }

    // blockLen大小包含值+索引，比如32个值+索引：blockLen=64
    template <typename DstTileData, typename SrcTileData>
    __aicore__ PTO_INLINE void TMRGSORT_IMPL(DstTileData &dst, SrcTileData &src0, uint32_t blockLen) {
        CheckStatic<DstTileData, DstTileData, SrcTileData, SrcTileData, SrcTileData, SrcTileData>();
        CheckOverMemory<DstTileData, DstTileData, SrcTileData, SrcTileData, SrcTileData, SrcTileData, 1>();
        unsigned dstCol = dst.GetValidCol();
        unsigned srcCol = src.GetValidCol();
        unsigned validRow = dst.GetValidRow();
        // 一个struct是8字节
        uint32_t numStructures = blockLen * sizeof(typename SrcTileData::DType) / STRUCTSIZE;
        uint8_t repeatTimes = srcCol / (blockLen * 4);
        TMrgsort<DstTileData, SrcTileData>(dst.data(), src.data(), numStructures, repeatTimes);
    }

    template <typename Src0TileData, typename Src1TileData, typename Src2TileData, typename Src3TileData>
    __aicore__ PTO_INLINE constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols + Src3TileData::Cols;
    }

    template <typename Src0TileData, typename Src1TileData, typename Src2TileData>
    __aicore__ PTO_INLINE constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols + Src2TileData::Cols;
    }

    template <typename Src0TileData, typename Src1TileData>
    __aicore__ PTO_INLINE constexpr uint32_t GETMRGSORTTMPSIZE() {
        return Src0TileData::Cols + Src1TileData::Cols;
    }
}
#endif