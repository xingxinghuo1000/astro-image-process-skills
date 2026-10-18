#pragma once
#include <cstdint>
#include <vector>
#include "util.hpp"

namespace astro {

// Adaptive MTF stretch. Returns float image in [0,1]; also reports mtf param
// and measured background level.
ImageF autoStretch(const ImageF& lin, double targetBg, double& mtfM, double& bgOut,
                   double blackPct = 0.3, double whitePct = 99.95);

void toU16(const ImageF& f, std::vector<uint16_t>& out);
ImageF downsampleRGB(const ImageF& f, int nw, int nh);
void toU8(const ImageF& f, std::vector<uint8_t>& out);

}  // namespace astro
