# Pre-seed SVE probe result to avoid CMake hanging on Apple Silicon.
# SVE is an ARM server feature; M-series chips do not support it.
# dotprod/i8mm are left uncached so cmake runs them normally (they complete instantly).
set(GGML_MACHINE_SUPPORTS_sve   "" CACHE UNINITIALIZED "" FORCE)
set(GGML_MACHINE_SUPPORTS_nosve "1" CACHE INTERNAL "" FORCE)

# Backends: phones are the only GPU workers
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_RPC   ON  CACHE BOOL "" FORCE)
