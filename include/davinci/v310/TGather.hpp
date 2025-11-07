#ifndef TGATHER_HPP
#define TGATHER_HPP

#include "common/constants.hpp"
#include "common.hpp"

namespace pto {
    __aicore__ PTO_INLINE int CEIL(int a, int b) {
        return (a + (b - 1)) / (b);
    }

    template <typename DstTileData, typename Src0TileData, typename Src1TileData>
    __aicore__ PTO_INLINE void CheckValid() {
        static_assert((sizeof(typename DstTileData::DType) == 1) || (sizeof(typename DstTileData::DType) == 2) || (sizeof(typename DstTileData::DType) == 4),
                      "expect b8/b16/b32");
        static_assert((sizeof(typename Src1TileData::DType) == 2) || (sizeof(typename Src1TileData::DType) == 4),
                      "expect b16/b32");
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

        if constexpr (sizeof(typename TileDataS0::DType) == 4) {
            __VEC_SCOPE__ 
            {
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
                uint16_t loop_num = CEIL(validCol, batch_size);
                for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        RegTensor<typename TileDataS1::DType> index;
                        vlds(index, src1Ptr, (i * TShape1 + j*batch_size), NORM);

                        uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                        vector_bool preg = plt_b32(count, POST_UPDATE);
                        
                        RegTensor<typename TileDataD::DType> v_output;
                        vgather2(v_output, src0Ptr,(vector_u32 &) index, preg);
                        vsts(v_output, dstPtr, (i * TShape1 +  j*batch_size), NORM_B32, preg);
                    }
                }
            }
        }
        else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 2) {
            __VEC_SCOPE__ 
            {
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
                uint16_t loop_num = CEIL(validCol,batch_size);
                for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        RegTensor<typename TileDataS1::DType> index;
                        vlds(index, src1Ptr, (i * TShape1 + j*batch_size), NORM);
                
                        uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                        vector_bool preg = plt_b16(count, POST_UPDATE);
                
                        RegTensor<typename TileDataD::DType> v_output;
                        vgather2(v_output, src0Ptr,(vector_u16 &) index, preg);
                        vsts(v_output, dstPtr, (i * TShape1 +  j*batch_size), NORM_B16, preg);
                    }
                }
            }
        }
        else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 4) {
            __VEC_SCOPE__ 
            {
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
                uint16_t loop_num = CEIL(validCol,batch_size);
                for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        RegTensor<typename TileDataS1::DType> index;
                        vlds(index, src1Ptr, (i * TShape1 + j*batch_size), NORM);
                
                        uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                        vector_bool preg = plt_b32(count, POST_UPDATE);
                
                        RegTensor<typename TileDataD::DType> v_output;
                        vgather2_bc(v_output, src0Ptr,(vector_u32 &) index, preg);
                        vsts(v_output, dstPtr, (i * TShape1 +  j*batch_size), PK_B32, preg);
                    }
                }
            }
        }
        /*
        else if constexpr (std::is_same<typename TileDataS0::DType, float8_e4m3_t>::value) {
            //fp8, suppose VL/1=256 output at a time, but just VL/2=128 indice
            //8-bit gather data is zero-extended to 16bit, so every time just 128 output
            __VEC_SCOPE__ 
            {
                // auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<uint8_t, DistVST::UNPK_B16>())>();
                constexpr auto partValue1 = std::integral_constant<::HiloPart, static_cast<::HiloPart>(HiloPart::Lower)>();
                constexpr auto partValue2 = std::integral_constant<::HiloPart, static_cast<::HiloPart>(HiloPart::Higher)>();
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType)) * 2;
                uint16_t loop_num = CEIL(TShape1,batch_size);
                for (uint16_t i = 0; i < (uint16_t) TShape0; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        vector_u16 index1;
                        vlds(index1, src1Ptr, (i * TShape1 + j*batch_size), NORM);
                        vector_u16 index2;
                        vlds(index2, src1Ptr, (i * TShape1 + j*batch_size+batch_size/2), NORM);
                
                        uint32_t count1 = ((j + 1) * batch_size >= TShape1 ? TShape1 - j * batch_size : batch_size);
                        vector_bool preg1 = plt_b8(count1, POST_UPDATE);
                        uint32_t count2 = ((j + 1) * batch_size >= TShape1 ? TShape1 - j * batch_size : batch_size);
                        vector_bool preg2 = plt_b8(count2, POST_UPDATE);
                
                        // vector_f8e4m3 v_output1;
                        vector_s16 v_output1;
                        vgather2(v_output1, (__ubuf__ int8_t *)src0Ptr,(vector_u16 &) index1, preg1);    //128 16-bit data
                        vector_s16 v_output2;
                        vgather2(v_output2, (__ubuf__ int8_t *)src0Ptr,(vector_u16 &) index2, preg2);

                        vector_u8 v_output3;    //256 
                        vpack(v_output3, v_output1, partValue1);
                        vpack(v_output3, v_output2, partValue2);
                        vsts(v_output3, (__ubuf__ uint8_t *)dstPtr, (i * TShape1 +  j*batch_size), NORM_B8, preg1);
                    }
                }
            }
        }
        */
        else if constexpr (std::is_same<typename TileDataS0::DType, float8_e4m3_t>::value) {
            __VEC_SCOPE__ 
            {
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
                uint16_t loop_num = CEIL(validCol,batch_size);
                for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        RegTensor<typename TileDataS1::DType> index;
                        vlds(index, src1Ptr, (i * TShape1 + j*batch_size), NORM);
                
                        uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                        // vector_bool preg = plt_b8(count, POST_UPDATE);    //cannot use b8 for vgather is still output b16
                        vector_bool preg = plt_b16(count, POST_UPDATE);
                        // vector_bool preg_b8_ALL = pset_b8(PAT_ALL);  //no mask work
                
                        vector_f8e4m3 v_output;
                        vgather2(v_output, src0Ptr,(vector_u16 &) index, preg);
                        vsts((vector_u8)v_output, (__ubuf__ uint8_t*)dstPtr, (i * TShape1 +  j*batch_size), PK_B16, preg);
                    }
                }
            }
        }
        else {
            __VEC_SCOPE__ 
            {
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
                uint16_t loop_num = CEIL(validCol,batch_size);
                for (uint16_t i = 0; i < (uint16_t) validRow; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        RegTensor<typename TileDataS1::DType> index;
                        vlds(index, src1Ptr, (i * TShape1 + j*batch_size), NORM);
                
                        uint32_t count = ((j + 1) * batch_size >= validCol ? validCol - j * batch_size : batch_size);
                        vector_bool preg = plt_b16(count, POST_UPDATE);
                
                        vector_f8e5m2 v_output;
                        vgather2(v_output, src0Ptr,(vector_u16 &) index, preg);
                        vsts((vector_u8)v_output, (__ubuf__ uint8_t*)dstPtr, (i * TShape1 +  j*batch_size), PK_B16, preg);
                    }
                }
            }
        }
    }

    template <typename TileDataD, typename TileDataS0, typename TileDataS1, unsigned TShape0, unsigned TShape1>
    __tf__ __aicore__ void TGather2D(typename TileDataS0::TileDType __out__ dst,
                                typename TileDataS0::TileDType __in__ src0,
                                typename TileDataS1::TileDType __in__ src1,
                                unsigned src0Shape1, 
                                unsigned dstShape1, 
                                unsigned axis) {

        __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
        __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);

        if constexpr (std::is_same<typename TileDataS0::DType, float>::value)
        {
            __VEC_SCOPE__ 
            {
                uint16_t batch_size = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
                uint16_t loop_num = CEIL(TShape1, batch_size);
                for (uint16_t i = 0; i < (uint16_t) TShape0; ++i) {
                    for (uint16_t j = 0; j < loop_num; ++j) {
                        vector_s32 index;
                        vlds(index, src1Ptr, (i * dstShape1 + j*batch_size), NORM);

                        uint32_t count = ((j + 1) * batch_size >= TShape1 ? TShape1 - j * batch_size : batch_size);
                        vector_bool preg = plt_b32(count, POST_UPDATE);

                
                        if (axis == 0) {
                            // For axis=0: output[i,j] = src0[index[i,j], j]
                            // Calculate offset = index * src0Shape1 + j
 
                            vmuls(index, index, src0Shape1, preg);
                            uint16_t base_offset = j*batch_size;
                            vadds(index, index, (uint16_t)base_offset, preg);
                            vector_s32 row_offset;
                            vci(row_offset,0);
                            vadd(index,index,row_offset,preg);
                    
                        } else {
                            // For axis=1: output[i,j] = src0[i, index[i,j]]
                            // Calculate offset = i * src0Shape1 + index
                            uint16_t base_offset = i * src0Shape1;
                            vadds(index, index, base_offset, preg);
                        }
                
                        vector_f32 v_output;
                        vgather2(v_output, src0Ptr,(vector_u32 &) index, preg);
                        vsts(v_output, dstPtr, (i * dstShape1 +  j*batch_size), NORM_B32, preg);
                    }
                }
            }
        }
        else if constexpr (std::is_same<typename TileDataS0::DType, half>::value)
        {
            __VEC_SCOPE__ 
            {
            uint16_t batch_size = ELE_CNT_B16;
            uint16_t loop_num = CEIL(TShape1,batch_size);
            for (uint16_t i = 0; i < (uint16_t) TShape0; ++i) {
                for (uint16_t j = 0; j < loop_num; ++j) {
                    vector_s16 index;
                    vlds(index, src1Ptr, (i * dstShape1 + j*batch_size), NORM);
                
                    uint32_t count = ((j + 1) * batch_size >= TShape1 ? TShape1 - j * batch_size : batch_size);
                    vector_bool preg = plt_b16(count, POST_UPDATE);
                
                
                    if (axis == 0) {
                        // For axis=0: output[i,j] = src0[index[i,j], j]
                        // Calculate offset = index * src0Shape1 + j
 
                        vmuls(index, index, src0Shape1, preg);
                        uint16_t base_offset = j*batch_size;
                        vadds(index, index, (uint16_t)base_offset, preg);
                        vector_s16 row_offset;
                        vci(row_offset,0);
                        vadd(index,index,row_offset,preg);
                    
                    } else {
                        // For axis=1: output[i,j] = src0[i, index[i,j]]
                        // Calculate offset = i * src0Shape1 + index
                        uint16_t base_offset = i * src0Shape1;
                        vadds(index, index, base_offset, preg);
                    }
                
                    vector_f16 v_output;
                    vgather2(v_output, src0Ptr,(vector_u16 &) index, preg);
                    vsts(v_output, dstPtr, (i * dstShape1 +  j*batch_size), NORM_B16, preg);
                }
            }
        }
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
    __aicore__ void TGATHER2D(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, unsigned src0Shape1, unsigned dstShape1, unsigned axis) {

        
        constexpr unsigned TShape0 = TileDataD::Rows;
        constexpr unsigned TShape1 = TileDataD::Cols;

        TGather2D<TileDataD, TileDataS0, TileDataS1, TShape0, TShape1>(dst.data(), src0.data(), src1.data(), src0Shape1, dstShape1, axis);
    }
}
#endif
