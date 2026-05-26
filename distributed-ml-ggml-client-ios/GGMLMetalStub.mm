// iphone6 branch: keep ggml-metal out of the link while satisfying ggml's
// backend registry hook so older devices stay on the CPU path.

#include <ggml-backend.h>

extern "C" ggml_backend_reg_t ggml_backend_metal_reg(void) {
    return NULL;
}
