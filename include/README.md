# include/

Public C/C++ headers for PTO Tile Lib (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See `include/pto/README.md` for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu)

## Related Docs

- ISA guide: `docs/README.md`
- Getting started: `docs/getting-started.md`
