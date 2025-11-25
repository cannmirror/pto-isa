#ifndef _INCLUDE_DAVINCI_TYPE_H_
#define _INCLUDE_DAVINCI_TYPE_H_
#if defined(__DAV_V220) || defined(__DAV_V310)
#define __aicore__ [aicore]
#else
#define __aicore__
#endif
#define PTO_INLINE inline __attribute__((always_inline))

#define __PTO_INSTR__ __aicore__ PTO_INLINE

namespace pto {
    // 01-bits patterns are read from right to left.
    // Right bits are low bits, corresponding to low index positions of data.
    enum class MaskPattern : uint8_t
    {
        // 以下1~7与指令VREDUCEv2的pattern mode保持一致
        P0101 = 1,  // 1: 01010101...0101 # 每个repeat内每两个元素取第一个元素
        P1010 = 2,  // 2: 10101010...1010 # 每个repeat内每两个元素取第二个元素
        P0001 = 3,  // 3: 00010001...0001 # 每个repeat内每四个元素取第一个元素
        P0010 = 4,  // 4: 00100010...0010 # 每个repeat内每四个元素取第二个元素
        P0100 = 5,  // 5: 01000100...0100 # 每个repeat内每四个元素取第三个元素
        P1000 = 6,  // 6: 10001000...1000 # 每个repeat内每四个元素取第四个元素
        P1111 = 7,  // 7: 11111111...1111 # 每个repeat内取全部元素
    };
}

#if defined(__CPU_SIM)
    #include <stdfloat>
    typedef std::float16_t half;
    typedef std::float16_t bfloat16_t;
    typedef std::float16_t aclFloat16;
#endif

#endif