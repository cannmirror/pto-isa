#ifndef TGATHER_HPP
#define TGATHER_HPP

#include "common/constants.hpp"


namespace pto {
    __aicore__ PTO_INLINE int CEIL(int a, int b) {
        return (a + (b - 1)) / (b);
    }

    template <typename DstTileData, typename Src0TileData, typename Src1TileData>
    __aicore__ PTO_INLINE void CheckValid() {
        static_assert((sizeof(typename DstTileData::DType) == 2) || (sizeof(typename DstTileData::DType) == 4),
                      "expect b16/b32");
        static_assert((sizeof(typename Src1TileData::DType) == 4),
                      "expect b32");
        static_assert((std::is_same<typename DstTileData::DType, typename Src0TileData::DType>::value),
                      "expect same size for indice and dst");
    }

    template <typename TileDataD, typename TileDataS0, typename TileDataS1>
    __tf__ __aicore__ void TGather(typename TileDataD::TileDType __out__ dst,
                                typename TileDataS0::TileDType __in__ src0,
                                typename TileDataS1::TileDType __in__ src1,
                                unsigned validCol, unsigned validRow) {

        __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);

        unsigned TShape0 = TileDataD::Rows;
        unsigned TShape1 = TileDataD::Cols;
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
        constexpr unsigned stride = TileDataD::RowStride;
        unsigned numRepeatPerLine = validCol / elementsPerRepeat;
        unsigned numRemainPerLine = validCol % elementsPerRepeat;
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX; 

        if constexpr (std::is_same_v<typename TileDataS0::DType, float> || std::is_same_v<typename TileDataS0::DType, int32_t> || std::is_same_v<typename TileDataS0::DType, uint32_t>) {
            //64 element per VL
            //counter mode
            set_mask_count();
            for (int i =  0; i < validRow; i++) {
                set_vector_mask(0, validCol);
                vmuls((__ubuf__ int32_t*) (dstPtr + i*TShape1), (__ubuf__ int32_t*) (src1Ptr + i*TShape1), sizeof(typename TileDataD::DType), 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vgather((__ubuf__ uint32_t*) (dstPtr + i*TShape1), (__ubuf__ uint32_t*) (dstPtr + i*TShape1),  (uintptr_t)src0Ptr, 8, 1);
            }

            set_mask_norm();
            set_vector_mask(-1, -1);
        }
        else{
            //128 element per VL
            //counter mode
            set_mask_count();
            for (int i =  0; i < validRow; i++) {
                set_vector_mask(0, validCol);
                vmuls((__ubuf__ int32_t*) (dstPtr + i*TShape1), (__ubuf__ int32_t*) (src1Ptr + i*TShape1), sizeof(typename TileDataD::DType), 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vgather((__ubuf__ uint16_t*) (dstPtr + i*TShape1), (__ubuf__ uint32_t*) (dstPtr + i*TShape1),  (uintptr_t)src0Ptr, 8, 1);
            }
            set_mask_norm();
            set_vector_mask(-1, -1);
        }
        
    }

    template <typename TileDataD, typename TileDataS0, typename TileDataS1, unsigned TShape0, unsigned TShape1>
    __tf__ __aicore__ void TGather2D(typename TileDataS0::TileDType __out__ dst,
                                typename TileDataS0::TileDType __in__ src0,
                                typename TileDataS1::TileDType __in__ src1,
                                typename TileDataS1::TileDType __in__ tmp,
                                unsigned src0Shape1, 
                                unsigned dstShape1, 
                                unsigned validCol, unsigned validRow, unsigned axis) {

        __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
        __ubuf__ typename TileDataS1::DType *tmpPtr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(tmp);

        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
        unsigned numRepeatPerLine = TShape1 / elementsPerRepeat;
        unsigned numRemainPerLine = TShape1 % elementsPerRepeat;

        if (axis == 0) {

            for( int i=0; i<TShape1; i++ )
            {
                tmpPtr[i] = i; 
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

	        set_mask_count();
            set_vector_mask(0, TShape0*validCol);
            vmuls((__ubuf__ int32_t*) (dstPtr), (__ubuf__ int32_t*) (src1Ptr), TShape1, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0, validCol);
            for (int i = 0; i < TShape0; ++i) {
                vadd((__ubuf__ int32_t*) (dstPtr + i * TShape1),
                      (__ubuf__ int32_t*) (dstPtr + i * TShape1),
                      tmpPtr,
                      1, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
            set_vector_mask(0, TShape0*validCol);
            vmuls((__ubuf__ int32_t*) (dstPtr), (__ubuf__ int32_t*) (dstPtr), sizeof(typename TileDataD::DType), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vmins((__ubuf__ int32_t*) (dstPtr),
                  (__ubuf__ int32_t*) (dstPtr),
                  (int32_t)0x0002ffe0,
                  1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vgather((__ubuf__ uint32_t*) (dstPtr), (__ubuf__ uint32_t*) (dstPtr), (uintptr_t)src0Ptr, 8, 1);
            set_mask_norm();
            set_vector_mask(-1, -1);
        }
        else {
            set_mask_count();
            set_vector_mask(0, validCol);
            for (int i = 0; i < TShape0; ++i) {
                vmuls((__ubuf__ int32_t*) (dstPtr + i * TShape1),
                      (__ubuf__ int32_t*) (src1Ptr + i * TShape1),
                      sizeof(typename TileDataD::DType),
                      1, 1, 1, 8, 8);
            }
            pipe_barrier(PIPE_V);
            for (int i = 0; i < TShape0; ++i) {
                vmins((__ubuf__ int32_t*) (dstPtr + i * TShape1),
                      (__ubuf__ int32_t*) (dstPtr + i * TShape1),
                      (int32_t)0x0002ffe0,
                      1, 1, 1, 8, 8);
            }
            pipe_barrier(PIPE_V);
            for (int i = 0; i < TShape0; ++i) {
                vgather((__ubuf__ uint32_t*) (dstPtr + i * TShape1),
                        (__ubuf__ uint32_t*) (dstPtr + i * TShape1),
                        (uintptr_t)src0Ptr + i * src0Shape1 * sizeof(typename TileDataD::DType),
                        8, 1);
	        }
            set_mask_norm();
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileDataD, typename TileDataS0, typename TileDataS1>
    __aicore__ void TGATHER_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1) {
        CheckValid<TileDataD, TileDataS0, TileDataS1>();

        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();

        TGather<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), validCol, validRow);
    }

    template <typename TileDataD, typename TileDataS0, typename TileDataS1>
    __aicore__ void TGATHER2D(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataS1 &tmp, unsigned axis) {

        
        constexpr unsigned TShape0 = TileDataD::Rows;
        constexpr unsigned TShape1 = TileDataD::Cols;
        unsigned src0Shape1 = TileDataD::Cols;
        if(axis == 1)
        {
            src0Shape1 = TileDataS0::Cols;
        }
        unsigned dstShape1 = TileDataD::Cols;
        unsigned validCol = dst.GetValidCol();
        unsigned validRow = dst.GetValidRow();

        TGather2D<TileDataD, TileDataS0, TileDataS1, TShape0, TShape1>(dst.data(), src0.data(), src1.data(), tmp.data(), src0Shape1, dstShape1, validCol, validRow, axis);
    }

    template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
    __tf__ __aicore__ void TGather(typename DstTileData::TileDType __out__ dst,
                                   typename SrcTileData::TileDType __in__ src,
                                   unsigned validRow, unsigned validCol)
    {
        using T = typename SrcTileData::DType;
        using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
 
        constexpr unsigned srcRepeatStride = SrcTileData::Cols * sizeof(T) / BLOCK_BYTE_SIZE;
 
        __ubuf__ typename DstTileData::DType *dstPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename SrcTileData::DType *srcPtr = (__ubuf__ typename SrcTileData::DType *)__cce_get_tile_ptr(src);
 
        set_mask_count();
        set_vector_mask(0, validCol);
        vreducev2(reinterpret_cast<__ubuf__ U *>(dstPtr), reinterpret_cast<__ubuf__ U *>(srcPtr),
                  reinterpret_cast<__ubuf__ U *>(srcPtr), validRow, 1, maskPattern, srcRepeatStride, 0);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
 
    template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
    __aicore__ PTO_INLINE void TGATHER_IMPL(DstTileData &dst, SrcTileData &src)
    {
        // Todo: add more static_asserts
        using T = typename SrcTileData::DType;
        static_assert(sizeof(T) == 2 || sizeof(T) == 4,
            "TGATHER: src element type must be 16 or 32-bit wide");
        static_assert((DstTileData::Loc == Location::Vec) && (SrcTileData::Loc == Location::Vec), 
            "TGATHER: expect vec location");
        static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor),
            "TGATHER: expect row major");
        static_assert((sizeof(typename DstTileData::DType) == sizeof(T)),
            "TGATHER: expect same type size for dst and src");
        TGather<DstTileData, SrcTileData, maskPattern>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol());
    }
}
#endif