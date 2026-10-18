#pragma once
#include <cmath>
#include <vector>
#include "util.hpp"

namespace astro {

struct Similarity {
    double a = 1.0, b = 0.0, tx = 0.0, ty = 0.0;  // [[a,-b,tx],[b,a,ty]]
    double rotDeg() const { return std::atan2(b, a) * 180.0 / 3.14159265358979323846; }
};

inline void applySim(const Similarity& M, double x, double y, double& ox, double& oy) {
    ox = M.a * x - M.b * y + M.tx;
    oy = M.b * x + M.a * y + M.ty;
}

// Inverse of a similarity transform (maps reference coords -> moving coords).
inline Similarity invertSim(const Similarity& M) {
    double s = M.a * M.a + M.b * M.b;
    if (s < 1e-12) return Similarity();
    Similarity I;
    I.a = M.a / s;
    I.b = -M.b / s;
    I.tx = -(I.a * M.tx - I.b * M.ty);
    I.ty = -(I.b * M.tx + I.a * M.ty);
    return I;
}

Similarity fitLeastSquares(const std::vector<Point>& src, const std::vector<Point>& dst);
Similarity ransacSimilarity(const std::vector<Point>& src, const std::vector<Point>& dst,
                            double thr, int iters);
// Nearest-neighbour correspondences (mov->ref) within maxd, then RANSAC fit.
Similarity matchAndFit(const std::vector<Point>& mov, const std::vector<Point>& ref,
                       double maxd, double thr);
Similarity refineTransform(Similarity M, const std::vector<Point>& mov,
                           const std::vector<Point>& ref);
double starScore(const ImageF& refHp, const ImageF& movHp, const Similarity& M, double frac);
void starResidual(const std::vector<Point>& ref, const std::vector<Point>& mov,
                  const Similarity& M, double tol, double& med, double& fracWithin);

ImageF warpGrayF(const ImageF& img, const Similarity& M);
ImageF warpRGB(const ImageF& img, const Similarity& M, std::vector<float>* weight = nullptr);

}  // namespace astro
