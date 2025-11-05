#ifndef TTILE_ASSIGN
#define TTILE_ASSIGN
#include <cstdint>

namespace pto{
    template <typename TileData>
    __aicore__ void TASSIGN_IMPL(TileData &tile, uint32_t addr) {
        tile.assignData(reinterpret_cast<typename TileData::TileDType>(static_cast<std::uintptr_t>(addr)));
    }
}
#endif