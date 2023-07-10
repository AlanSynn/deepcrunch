#include <torch/extensions.h>

namespace deepcrunch {
namespace csrc {

void register_extension() {
    // This function is intentionally left blank.
}

} // namespace csrc
} // namespace deepcrunch


pybind11::module torch_extension_module(const pybind11::module& m) {
    pybind11::module m_ext = m.def_submodule("ext", "DeepCrunch extension module");
}