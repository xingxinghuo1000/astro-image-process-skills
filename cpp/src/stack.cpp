#include "stack.hpp"

#include <algorithm>
#include <cmath>

namespace astro {

static double percentileFrom(std::vector<float>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idxf = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)std::floor(idxf);
    size_t hi = std::min(lo + 1, v.size() - 1);
    double f = idxf - lo;
    return v[lo] * (1 - f) + v[hi] * f;
}

static float mtf(double x, double m) {
    if (x <= 0) return 0.f;
    if (x >= 1) return 1.f;
    return (float)(((m - 1) * x) / ((2 * m - 1) * x - m));
}

ImageF autoStretch(const ImageF& lin, double targetBg, double& mtfM, double& bgOut,
                   double blackPct, double whitePct) {
    const int nc = lin.c;
    std::vector<double> black(nc, 0.0);
    for (int c = 0; c < nc; ++c) {
        std::vector<float> vals;
        vals.reserve(lin.d.size() / nc / 4 + 1);
        for (size_t i = c; i < lin.d.size(); i += nc * 4) vals.push_back(lin.d[i]);
        black[c] = percentileFrom(vals, blackPct);
    }

    std::vector<float> allv;
    allv.reserve(lin.d.size() / 4 + 1);
    for (size_t i = 0; i < lin.d.size(); i += 4) allv.push_back(lin.d[i]);
    double white = percentileFrom(allv, whitePct);
    if (white <= 0) white = 1.0;

    ImageF xn(lin.w, lin.h, nc);
    for (size_t i = 0; i < lin.d.size(); ++i) {
        double v = lin.d[i] - black[i % nc];
        v = std::max(0.0, std::min(v / white, 1.0));
        xn.d[i] = (float)v;
    }

    // background = median of the per-pixel channel mean (matches the Python impl)
    std::vector<float> lum;
    lum.reserve((size_t)lin.w * lin.h / 4 + 1);
    for (size_t i = 0; i < (size_t)lin.w * lin.h; i += 4) {
        double a = 0;
        for (int c = 0; c < nc; ++c) a += xn.d[i * nc + c];
        lum.push_back((float)(a / nc));
    }
    std::nth_element(lum.begin(), lum.begin() + lum.size() / 2, lum.end());
    double bg = lum[lum.size() / 2];

    double m = 0.5;
    if (bg <= 1e-6) {
        m = 1.0;
    } else {
        double lo = 1e-4, hi = 0.999;
        for (int i = 0; i < 60; ++i) {
            m = 0.5 * (lo + hi);
            if (mtf(bg, m) > targetBg) lo = m; else hi = m;
        }
        m = 0.5 * (lo + hi);
    }
    mtfM = m;
    bgOut = bg;

    ImageF out(lin.w, lin.h, nc);
    for (size_t i = 0; i < xn.d.size(); ++i) out.d[i] = mtf(xn.d[i], m);
    return out;
}

void toU16(const ImageF& f, std::vector<uint16_t>& out) {
    out.resize(f.d.size());
    for (size_t i = 0; i < f.d.size(); ++i) {
        float v = std::max(0.f, std::min(f.d[i], 1.f));
        out[i] = (uint16_t)std::lround(v * 65535.0);
    }
}

ImageF downsampleRGB(const ImageF& f, int nw, int nh) {
    ImageF out(nw, nh, 3);
    double sx = (double)f.w / nw, sy = (double)f.h / nh;
    for (int y = 0; y < nh; ++y) {
        int y0 = (int)(y * sy), y1 = std::max(y0 + 1, (int)((y + 1) * sy));
        y1 = std::min(y1, f.h);
        for (int x = 0; x < nw; ++x) {
            int x0 = (int)(x * sx), x1 = std::max(x0 + 1, (int)((x + 1) * sx));
            x1 = std::min(x1, f.w);
            double acc[3] = {0, 0, 0};
            int n = 0;
            for (int yy = y0; yy < y1; ++yy) {
                const float* p = f.row(yy);
                for (int xx = x0; xx < x1; ++xx) {
                    acc[0] += p[xx * 3 + 0]; acc[1] += p[xx * 3 + 1]; acc[2] += p[xx * 3 + 2];
                    ++n;
                }
            }
            float* o = out.row(y) + x * 3;
            for (int c = 0; c < 3; ++c) o[c] = n ? (float)(acc[c] / n) : 0.f;
        }
    }
    return out;
}

void toU8(const ImageF& f, std::vector<uint8_t>& out) {
    out.resize(f.d.size());
    for (size_t i = 0; i < f.d.size(); ++i) {
        float v = std::max(0.f, std::min(f.d[i], 1.f));
        out[i] = (uint8_t)std::lround(v * 255.0);
    }
}

}  // namespace astro
