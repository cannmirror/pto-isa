#ifndef TDIVS_HPP
#define TDIVS_HPP

#include <functional>
#include "common/constants.hpp"

namespace pto
{
    template <typename DataType>
    __aicore__ void SDIV(__ubuf__ DataType *dst, __ubuf__ DataType *src0, DataType src1, uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {

        if constexpr (std::is_same<DataType, int32_t>::value)
        {
            vector_dup(dst, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_s322f32(reinterpret_cast<__ubuf__ float *>(dst), dst, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_s322f32(reinterpret_cast<__ubuf__ float *>(src0), src0, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vdiv(reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(src0), repeat, dstBlockStride, srcBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float *>(dst), repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
        }
        else if constexpr (std::is_same<DataType, int16_t>::value)
        {
            vector_dup(dst, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_s162f16(reinterpret_cast<__ubuf__ half *>(dst), dst, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_s162f16(reinterpret_cast<__ubuf__ half *>(src0), src0, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vdiv(reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(src0), repeat, dstBlockStride, srcBlockStride, dstBlockStride, dstRepeatStride, srcRepeatStride, dstRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_f162s16z(dst, reinterpret_cast<__ubuf__ half *>(dst), repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
        }
        else if constexpr (std::is_same<DataType, float>::value)
        {
            vector_dup(dst, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vdiv(dst, dst, src0, repeat, dstBlockStride, srcBlockStride, dstBlockStride, dstRepeatStride, srcRepeatStride, dstRepeatStride);
        }
        else if constexpr (std::is_same<DataType, half>::value)
        {
            vector_dup(dst, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            vdiv(dst, dst, src0, repeat, dstBlockStride, srcBlockStride, dstBlockStride, dstRepeatStride, srcRepeatStride, dstRepeatStride);
        }
    }
    template <typename DataType>
    __aicore__ void DIVS(__ubuf__ DataType *dst, __ubuf__ DataType *src0, DataType src1, uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        float divider = static_cast<float>(src1);
        if (divider != 0)
        {
            divider = 1.0f / divider;
        }
        else
        {
            divider = 1.0 / 0.0;
        }
        if constexpr (std::is_same<DataType, int32_t>::value)
        {
            vconv_s322f32(reinterpret_cast<__ubuf__ float *>(dst), src0, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vmuls(reinterpret_cast<__ubuf__ float *>(dst), reinterpret_cast<__ubuf__ float *>(dst), divider, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_f322s32z(dst, reinterpret_cast<__ubuf__ float *>(dst), repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
        }
        else if constexpr (std::is_same<DataType, int16_t>::value)
        {
            vconv_s162f16(reinterpret_cast<__ubuf__ half *>(dst), src0, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vmuls(reinterpret_cast<__ubuf__ half *>(dst), reinterpret_cast<__ubuf__ half *>(dst), static_cast<half>(divider), repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_f162s16z(dst, reinterpret_cast<__ubuf__ half *>(dst), repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
        }
        else if constexpr (std::is_same<DataType, half>::value)
        {
            vector_dup(dst, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vdiv(dst, src0, dst, repeat, dstBlockStride, srcBlockStride, dstBlockStride, dstRepeatStride, srcRepeatStride, dstRepeatStride);
        }
        else
        {
            vmuls(dst, src0, divider, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
        }
    }
    template <typename DataType>
    inline __aicore__ void MULS(__ubuf__ DataType *dst, __ubuf__ DataType *src0, DataType src1, uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        vmuls(dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    template <typename DataType>
    inline __aicore__ void ADDS(__ubuf__ DataType *dst, __ubuf__ DataType *src0, DataType src1, uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        vadds(dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
    template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride, typename Func>
    __tf__ __aicore__ void TBinScalar(typename TileData::TileDType __out__ dst,
                                      typename TileData::TileDType __in__ src0,
                                      typename TileData::DType __in__ src1,
                                      Func currentFunc,
                                      unsigned numRepeatPerLine,
                                      unsigned numRemainPerLine,
                                      unsigned validRow)
    {
        __ubuf__ typename TileData::DType *dstPtr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
        __ubuf__ typename TileData::DType *src0Ptr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src0);
        if (numRepeatPerLine > 0)
        {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < validRow; i++)
            {
                if (numLoop)
                {
                    for (int j = 0; j < numLoop; j++)
                    {
                        currentFunc(dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                                    src0Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                                    src1,
                                    REPEAT_MAX, 1, 1, 8, 8);
                    }
                }
                if (remainAfterLoop)
                {
                    currentFunc(dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                                src0Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                                src1,
                                remainAfterLoop, 1, 1, 8, 8);
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        src0Ptr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine)
        {
            unsigned numLoop = validRow / REPEAT_MAX;
            unsigned remainAfterLoop = validRow % REPEAT_MAX;
            bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop)
            {
                for (int i = 0; i < numLoop; i++)
                {
                    if (strideOverFlag)
                    {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++)
                        {
                            currentFunc(dstPtr + i * REPEAT_MAX * stride + j * stride,
                                        src0Ptr + i * REPEAT_MAX * stride + j * stride,
                                        src1,
                                        1, 1, 1, 1, 1);
                        }
                    }
                    else
                    {
                        currentFunc(dstPtr + i * REPEAT_MAX * stride,
                                    src0Ptr + i * REPEAT_MAX * stride,
                                    src1,
                                    REPEAT_MAX, 1, 1, static_cast<uint8_t>(stride / blockSizeElem), static_cast<uint8_t>(stride / blockSizeElem));
                    }
                }
            }
            if (remainAfterLoop)
            {
                if (strideOverFlag)
                {
                    for (unsigned j = 0; j < remainAfterLoop; j++)
                    {
                        currentFunc(dstPtr + numLoop * REPEAT_MAX * stride + j * stride,
                                    src0Ptr + numLoop * REPEAT_MAX * stride + j * stride,
                                    src1,
                                    1, 1, 1, 1, 1);
                    }
                }
                else
                {
                    currentFunc(dstPtr + numLoop * REPEAT_MAX * stride,
                                src0Ptr + numLoop * REPEAT_MAX * stride,
                                src1,
                                remainAfterLoop, 1, 1, static_cast<uint8_t>(stride / blockSizeElem), static_cast<uint8_t>(stride / blockSizeElem));
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <typename TileData>
    __aicore__ void TDIVS(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        auto funcPtr = DIVS<typename TileData::DType>;
        TBinScalar<TileData, elementsPerRepeat, blockSizeElem, stride, decltype(funcPtr)>(dst.data(), src0.data(), scalar, DIVS,
                                                                                          numRepeatPerLine, numRemainPerLine, validRow);
    }
    template <typename TileData>
    __aicore__ void TDIVS(TileData &dst, typename TileData::DType scalar, TileData &src0)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        auto funcPtr = SDIV<typename TileData::DType>;
        TBinScalar<TileData, elementsPerRepeat, blockSizeElem, stride, decltype(funcPtr)>(dst.data(), src0.data(), scalar, SDIV,
                                                                                          numRepeatPerLine, numRemainPerLine, validRow);
    }
    template <typename TileData>
    __aicore__ void TADDS(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        auto funcPtr = ADDS<typename TileData::DType>;
        TBinScalar<TileData, elementsPerRepeat, blockSizeElem, stride, decltype(funcPtr)>(dst.data(), src0.data(), scalar, ADDS,
                                                                                          numRepeatPerLine, numRemainPerLine, validRow);
    }
    template <typename TileData>
    __aicore__ void TMULS(TileData &dst, TileData &src0, typename TileData::DType scalar)
    {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        auto funcPtr = MULS<typename TileData::DType>;
        TBinScalar<TileData, elementsPerRepeat, blockSizeElem, stride, decltype(funcPtr)>(dst.data(), src0.data(), scalar, MULS,
                                                                                          numRepeatPerLine, numRemainPerLine, validRow);
    }
}

#endif