#ifndef FLASHRT_CPP_FAMILIES_VLA_RUNTIME_H
#define FLASHRT_CPP_FAMILIES_VLA_RUNTIME_H

#include "flashrt/cpp/families/vla/manifest.h"
#include "flashrt/cpp/runtime.h"

namespace flashrt {
namespace families {
namespace vla {

/* Common VLA runtime shape. Concrete models such as Pi0.5 bind this family
 * contract to their own buffers, tokenizer/prompt path, and action schema. */
class Runtime : public flashrt::runtime::ModelRuntime {
public:
    ~Runtime() override = default;
    virtual const Manifest& manifest() const = 0;
};

}  // namespace vla
}  // namespace families
}  // namespace flashrt

#endif  // FLASHRT_CPP_FAMILIES_VLA_RUNTIME_H
