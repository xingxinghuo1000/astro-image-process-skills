#pragma once
#include <string>
#include <vector>
#include "util.hpp"

namespace astro {

// Half-size 8-bit grayscale (default gamma), used for registration.
bool decodeHalfGray(const std::string& path, std::vector<float>& gray, int& w, int& h);

// Full-resolution linear 16-bit RGB (gamma 1.0, camera WB), float in [0,1].
bool decodeFullLinear(const std::string& path, ImageF& rgb);

}  // namespace astro
