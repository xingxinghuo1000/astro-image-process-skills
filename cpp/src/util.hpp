#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace astro {

struct ImageF {
    int w = 0, h = 0, c = 1;
    std::vector<float> d;
    ImageF() = default;
    ImageF(int w_, int h_, int c_) : w(w_), h(h_), c(c_), d((size_t)w_ * h_ * c_, 0.f) {}
    float* row(int y) { return d.data() + (size_t)y * w * c; }
    const float* row(int y) const { return d.data() + (size_t)y * w * c; }
};

struct Point { float x = 0.f, y = 0.f; };

void gaussianBlur1(const ImageF& in, ImageF& out, float sigma);
ImageF resizeArea1(const ImageF& in, int nw, int nh);
void highpass(const ImageF& gray, ImageF& out, float preBlur, float hpSigma);
std::vector<Point> detectStars(const ImageF& hp, float nsigma = 5.f,
                               int minArea = 3, int maxArea = 400);

bool writeTiff16(const std::string& path, const std::vector<uint16_t>& rgb, int w, int h);
bool writePng8(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h);

}  // namespace astro
