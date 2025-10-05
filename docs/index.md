# CuSAXS API Documentation

This documentation complements the top-level README by cataloging the public API exposed by both the CUDA backend and the Python tooling. It is organized into two major sections:

- [Backend APIs](backend.md): C++ / CUDA classes, free functions, and configuration structs that power the SAXS computation.
- [Python APIs](python.md): Python classes and functions that orchestrate trajectory loading, configuration, and user interfaces.

Additional guides can be added under this directory as the project evolves. Each reference entry links back to the source file so you can quickly inspect the implementation.

## Quick Navigation

- [Backend Classes](backend.md#classes)
- [Backend Free Functions](backend.md#free-functions)
- [Backend Support Types](backend.md#support-types)
- [Python Classes](python.md#classes)
- [Python Functions](python.md#functions)
- [Python CLI / GUI Utilities](python.md#cli--gui-utilities)

## Conventions

- File paths follow the repository layout (e.g. `cpp-src/Exec/RunSaxs.cu`).
- Method signatures reflect the current code base; notable side-effects and threading/GIL expectations are captured in the descriptions.
- Configuration defaults come from `Options` and `CudaSaxsConfig`; see the README for higher-level guidance.

For build instructions, runtime requirements, and usage walkthroughs, see the root `README.md`.
