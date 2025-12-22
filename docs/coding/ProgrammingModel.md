# PTO Tile Intrinsics Programming Model

PTO Tile Lib provides **tile-granularity** intrinsics that map to the PTO ISA. The model is designed for:

- **Portability across device generations**: hardware may change (instruction details, storage layout, scheduling), but the programming model remains stable.
- **Near-hardware performance**: the Tile and GlobalTensor abstractions are low-level enough to express efficient data movement and compute.
- **Two user profiles**: a productive “compiler does the hard work” style, and an expert “I control placement and sync” style.

For the abstract execution model (core/device/host), see `docs/machine/abstract-machine.md`.

## Core concepts

- **Tile**: a fixed-capacity 2-D on-chip buffer (conceptually a tile register / SRAM block) and the unit of computation for most PTO instructions. See `docs/coding/Tile.md`.
- **GlobalTensor**: a lightweight view of global memory (GM) as a 5-D tensor with shape/stride/layout metadata, used by memory instructions such as `TLOAD` and `TSTORE`. See `docs/coding/GlobalTensor.md`.
- **Scalar**: immediate values and enumerations that parameterize instructions (rounding modes, comparison modes, atomic modes, etc.). See `docs/coding/Scalar.md`.
- **Event**: explicit dependency tokens between pipeline classes, used to order operations without introducing a full barrier everywhere. See `docs/coding/Event.md`.

## Two development styles

### PTO-Auto

PTO-Auto targets developers who prefer a simple, portable programming experience:

- The compiler/runtime chooses memory placement and address binding.
- The compiler inserts required synchronization.
- The compiler schedules operations and applies fusions when possible.

This mode is a good starting point for correctness and portability.

### PTO-Manual

PTO-Manual targets developers who need full control for performance tuning:

- The developer controls memory placement and binding (for example via `TASSIGN`).
- The developer explicitly expresses ordering (events and/or `TSYNC`).
- The developer controls the operation schedule.

This mode enables expert tuning on critical kernels while still using the shared Tile/GlobalTensor abstractions.

