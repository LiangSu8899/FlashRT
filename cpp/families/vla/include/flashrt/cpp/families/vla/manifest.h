#ifndef FLASHRT_CPP_FAMILIES_VLA_MANIFEST_H
#define FLASHRT_CPP_FAMILIES_VLA_MANIFEST_H

#include "flashrt/cpp/modalities/action.h"
#include "flashrt/cpp/modalities/vision.h"

#include <string>
#include <vector>

namespace flashrt {
namespace families {
namespace vla {

struct GraphNames {
    std::string infer = "infer";
    std::string decode_only = "decode_only";
};

struct StateRegion {
    std::string name;
    std::string buffer;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct Manifest {
    modalities::VisionPreprocessSpec vision;
    modalities::ActionPostprocessSpec action;
    GraphNames graphs;
    std::vector<StateRegion> state_regions;
};

}  // namespace vla
}  // namespace families
}  // namespace flashrt

#endif  // FLASHRT_CPP_FAMILIES_VLA_MANIFEST_H
