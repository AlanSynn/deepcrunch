#include <torch/extension.h>
#include "deepcrunch.h"
//==============================================================================
namespace deepcrunch {
//==============================================================================
void register_extension() {
    // This function is intentionally left blank.
}
} // namespace deepcrunch
//==============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_extension", &deepcrunch::register_extension, "register extension");
}
//==============================================================================