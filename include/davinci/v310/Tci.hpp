#ifndef TCI_HPP
#define TCI_HPP

#include "common/constants.hpp"
#include "common/utils.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace pto {
template <typename TileData, typename T>
__aicore__
PTO_INLINE
void CheckValid() {
    static_assert((std::is_same<typename TileData::DType, T>::value),
    "expect src and dst same datatype");
    static_assert((sizeof(typename TileData::DType) == 4 || (sizeof(typename TileData::DType) == 2)),
    "expect b32 or b16");
    static_assert((TileData::Cols != 1),
    "expect row is 1");
}

template <typename TileData, typename T, int descending = 0>
__tf__
__aicore__
void Tci(typename TileData::TileDType __out__ dst, T S, unsigned validCol)
{
    // 1.获取dst中的信息;
    using Tdst = typename TileData::DType;
    __ubuf__ Tdst *dstPtr = (__ubuf__ Tdst *)__cce_get_tile_ptr(dst);
    uint16_t block_size = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType); // 每个block涉及多少个元素；
    uint16_t batch_size = REPEAT_BYTE / static_cast<uint16_t>(sizeof(typename TileData::DType)); // 每次repeat涉及多少个元素；
    uint16_t colNum = (uint32_t)TileData::Cols;
    uint16_t loops = (validCol + batch_size -1) / batch_size;  // 计算循环次数，注意最后一个循环可能不是满拍；
    int32_t t = S;
    MaskReg preg;
    if constexpr(std::is_same_v<typename TileData::DType, int32_t>)
    {
        if(descending == 0 )
        { //递增
            __VEC_SCOPE__
            {
                for(uint16_t i = 0; i < loops; ++i)
                {
                vector_s32 index;
                vci(index, t);  // 没有配置reverse的值，默认0，升序;
                uint32_t count = (i+1)*batch_size > validCol ? (validCol - i * batch_size):batch_size;
                preg = CreatePredicate<Tdst>(count);
                vsts(index, dstPtr, (i * batch_size), NORM_B32, preg);
                t = t + 64;
                }
            }
        }
        else if (descending == 1)
        {
            __VEC_SCOPE__
            {
                for(uint16_t i = 0; i < loops; ++i)
                {
                    vector_s32 index;
                    vci(index, 0);
                    uint32_t count = (i+1) * batch_size > validCol ? (validCol - i * batch_size):batch_size;
                    preg = CreatePredicate<Tdst>(count);
                    vmuls(index, index, -1, preg);
                    vadds(index, index, t, preg);
                    vsts(index, dstPtr, (i * batch_size), NORM_B32, preg);
                    t = t - 64;
                }
            }
        }
    }
    else if constexpr(std::is_same_v<typename TileData::DType, int16_t>)
    {
        if(descending ==0 )
        { // 递增
            __VEC_SCOPE__
            {
                for(uint16_t i = 0; i < loops; ++i)
                {
                vector_s16 index;
                vci(index, t);  // 没有配置reverse的值，默认0，升序;
                uint32_t count = (i+1)*batch_size > validCol ? (validCol - i * batch_size):batch_size;
                preg = CreatePredicate<Tdst>(count);
                vsts(index, dstPtr, (i * batch_size), NORM_B16, preg);
                t = t + 128;
                }
            }
        }
        else if (descending == 1)
        { // 递减
            __VEC_SCOPE__
            {
                for(uint16_t i = 0; i < loops; ++i)
                {
                    vector_s16 index;
                    vci(index, 0);
                    uint32_t count = (i+1) * batch_size > validCol ? (validCol - i * batch_size):batch_size;
                    preg = CreatePredicate<Tdst>(count);
                    vmuls(index, index, -1, preg);
                    vadds(index, index, t, preg);
                    vsts(index, dstPtr, (i * batch_size), NORM_B16, preg);
                    t = t - 128;
                }
            }
        }
    }
}

template <typename TileData, typename T, int descending>
__aicore__
void TCI_IMPL (TileData &dst, T S)
{
    CheckValid<TileData, T>();
    unsigned validCol = dst.GetValidCol();
    Tci<TileData, T, descending>(dst.data(), S, validCol);
}
}
#endif