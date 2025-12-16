# kernels/

Kernel implementations and custom operator code that complements PTO Tile Lib.

## Layout

- `custom/`: Example custom kernels/operators (extension points)

## Notes

If you add custom operators, it’s recommended to keep:

- Public interfaces in `include/`
- Implementations and build glue in `kernels/` and `tests/`

This keeps integration and validation straightforward for upper-layer frameworks.
