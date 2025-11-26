#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace pto{
constexpr int REPEAT_BYTE = 256;

constexpr int REPEAT_MAX = 255;

constexpr const int BLOCK_BYTE_SIZE = 32;

constexpr const int REPEAT_STRIDE_MAX = 255;

constexpr const uint64_t BLOCK_MAX_PER_REPEAT = 8;

constexpr const uint32_t TMP_UB_SIZE = 8 * 1024;

constexpr const uint32_t TMP_UB_OFFSET = 184 * 1024;

constexpr const uint64_t MASK_LEN = 64;

constexpr const int BLOCK_LEN = 16;

constexpr const int CUBE_BLOCK_SIZE = 512;

constexpr const int C0_SIZE_BYTE = 32;

enum class RoundMode : uint8_t {
    CAST_NONE = 0,
    CAST_RINT = 1,  // round to nearest, tie to even
    CAST_ROUND = 2, // round to nearest, tie away from zero
    CAST_FLOOR = 3, // round to minus infinity
    CAST_CEIL = 4,  // round to positive infinity
    CAST_TRUNC = 5, // round to zero
    CAST_ODD = 6,   // round to odd (Von Neumann rounding)
};

enum class TCopyMode : uint8_t{
    SHALLOW_COPY = 0,
    DEEP_COPY = 1,
};

enum class L0cToUBMode : uint8_t{
    SingleModeUB0 = 0,
    SingleModeUB1 = 1,
    DualModeSplitM = 2,
    DualModeSplitN = 3,
};

enum class AtomicType : uint8_t {
    AtomicNone = 0,
    AtomicAdd = 1,
};

enum class CmpMode : uint8_t {
    EQ = 0,
    NE = 1,
    LT = 2,
    GT = 3,
    GE = 4,
    LE = 5,
};
}
#endif