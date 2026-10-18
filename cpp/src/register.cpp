#include "register.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace astro {

Similarity fitLeastSquares(const std::vector<Point>& src, const std::vector<Point>& dst) {
    double AtA[4][4] = {{0}}, Atb[4] = {0};
    for (size_t i = 0; i < src.size(); ++i) {
        double x = src[i].x, y = src[i].y, u = dst[i].x, v = dst[i].y;
        double r0[4] = {x, -y, 1, 0};
        double r1[4] = {y, x, 0, 1};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) AtA[r][c] += r0[r] * r0[c] + r1[r] * r1[c];
            Atb[r] += r0[r] * u + r1[r] * v;
        }
    }
    // Gaussian elimination with partial pivoting
    double A[4][5];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) A[r][c] = AtA[r][c];
        A[r][4] = Atb[r];
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        std::swap(A[col], A[piv]);
        if (std::fabs(A[col][col]) < 1e-12) continue;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            double f = A[r][col] / A[col][col];
            for (int c = col; c < 5; ++c) A[r][c] -= f * A[col][c];
        }
    }
    Similarity M;
    if (std::fabs(A[0][0]) > 1e-12) M.a = A[0][4] / A[0][0];
    if (std::fabs(A[1][1]) > 1e-12) M.b = A[1][4] / A[1][1];
    if (std::fabs(A[2][2]) > 1e-12) M.tx = A[2][4] / A[2][2];
    if (std::fabs(A[3][3]) > 1e-12) M.ty = A[3][4] / A[3][3];
    return M;
}

static Similarity solveTwo(const Point& p1, const Point& p2, const Point& q1, const Point& q2) {
    double dxs = p1.x - p2.x, dys = p1.y - p2.y;
    double du = q1.x - q2.x, dv = q1.y - q2.y;
    double den = dxs * dxs + dys * dys;
    Similarity M;
    if (den < 1e-9) return M;
    M.a = (du * dxs + dv * dys) / den;
    M.b = (dv * dxs - du * dys) / den;
    M.tx = q1.x - (M.a * p1.x - M.b * p1.y);
    M.ty = q1.y - (M.b * p1.x + M.a * p1.y);
    return M;
}

Similarity ransacSimilarity(const std::vector<Point>& src, const std::vector<Point>& dst,
                            double thr, int iters) {
    size_t n = src.size();
    if (n < 2) return Similarity();
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    std::vector<char> bestMask(n, 0);
    int bestCount = 0;
    double t2 = thr * thr;
    for (int it = 0; it < iters; ++it) {
        size_t i = pick(rng), j = pick(rng);
        if (i == j) continue;
        Similarity M = solveTwo(src[i], src[j], dst[i], dst[j]);
        int cnt = 0;
        for (size_t k = 0; k < n; ++k) {
            double ox, oy;
            applySim(M, src[k].x, src[k].y, ox, oy);
            double dx = ox - dst[k].x, dy = oy - dst[k].y;
            if (dx * dx + dy * dy < t2) ++cnt;
        }
        if (cnt > bestCount) {
            bestCount = cnt;
            for (size_t k = 0; k < n; ++k) {
                double ox, oy;
                applySim(M, src[k].x, src[k].y, ox, oy);
                double dx = ox - dst[k].x, dy = oy - dst[k].y;
                bestMask[k] = (dx * dx + dy * dy < t2) ? 1 : 0;
            }
        }
    }
    if (bestCount < 2) return Similarity();
    std::vector<Point> s, d;
    s.reserve(bestCount); d.reserve(bestCount);
    for (size_t k = 0; k < n; ++k)
        if (bestMask[k]) { s.push_back(src[k]); d.push_back(dst[k]); }
    return fitLeastSquares(s, d);
}

Similarity matchAndFit(const std::vector<Point>& mov, const std::vector<Point>& ref,
                       double maxd, double thr) {
    std::vector<Point> s, d;
    double md2 = maxd * maxd;
    for (const auto& p : mov) {
        const Point* best = nullptr;
        double bd2 = md2;
        for (const auto& q : ref) {
            double dx = p.x - q.x, dy = p.y - q.y, dd = dx * dx + dy * dy;
            if (dd < bd2) { bd2 = dd; best = &q; }
        }
        if (best) { s.push_back(p); d.push_back(*best); }
    }
    if (s.size() < 8) return Similarity();
    return ransacSimilarity(s, d, thr, 4000);
}

Similarity refineTransform(Similarity M, const std::vector<Point>& mov,
                           const std::vector<Point>& ref) {
    const double tols[] = {8, 4, 2, 1.2, 0.8};
    for (double tol : tols) {
        std::vector<Point> s, d;
        for (const auto& p : mov) {
            double wx, wy;
            applySim(M, p.x, p.y, wx, wy);
            const Point* best = nullptr;
            double bd2 = tol * tol;
            for (const auto& q : ref) {
                double dx = wx - q.x, dy = wy - q.y, dd = dx * dx + dy * dy;
                if (dd < bd2) { bd2 = dd; best = &q; }
            }
            if (best) { s.push_back(p); d.push_back(*best); }
        }
        if (s.size() < 10) break;
        M = ransacSimilarity(s, d, tol, 2000);
    }
    return M;
}

double starScore(const ImageF& refHp, const ImageF& movHp, const Similarity& M, double frac) {
    ImageF w = warpGrayF(movHp, M);
    int rows = (int)(refHp.h * frac);
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    long n = 0;
    for (int y = 0; y < rows; ++y) {
        const float* a = refHp.row(y);
        const float* b = w.row(y);
        for (int x = 0; x < refHp.w; ++x) { sa += a[x]; sb += b[x]; saa += a[x] * a[x]; sbb += b[x] * b[x]; sab += a[x] * b[x]; ++n; }
    }
    double ma = sa / n, mb = sb / n;
    double va = saa - n * ma * ma, vb = sbb - n * mb * mb;
    double cov = sab - n * ma * mb;
    double denom = std::sqrt(va) * std::sqrt(vb);
    return denom > 1e-12 ? cov / denom : 0.0;
}

void starResidual(const std::vector<Point>& ref, const std::vector<Point>& mov,
                  const Similarity& M, double tol, double& med, double& fracWithin) {
    med = 1e9; fracWithin = 0.0;
    if (ref.empty() || mov.empty()) return;
    std::vector<double> dists;
    dists.reserve(mov.size());
    int within = 0;
    for (const auto& p : mov) {
        double wx, wy;
        applySim(M, p.x, p.y, wx, wy);
        double bd2 = 1e18;
        for (const auto& q : ref) {
            double dx = wx - q.x, dy = wy - q.y, dd = dx * dx + dy * dy;
            if (dd < bd2) bd2 = dd;
        }
        double dm = std::sqrt(bd2);
        dists.push_back(dm);
        if (dm < tol) ++within;
    }
    std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
    med = dists[dists.size() / 2];
    fracWithin = (double)within / mov.size();
}

ImageF warpGrayF(const ImageF& img, const Similarity& M) {
    Similarity Mi = invertSim(M);
    ImageF out(img.w, img.h, 1);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < img.h; ++y) {
        float* o = out.row(y);
        for (int x = 0; x < img.w; ++x) {
            double sx, sy;
            applySim(Mi, x, y, sx, sy);
            if (sx < 0 || sy < 0 || sx > img.w - 1 || sy > img.h - 1) { o[x] = 0.f; continue; }
            int x0 = (int)sx, y0 = (int)sy;
            int x1 = std::min(x0 + 1, img.w - 1), y1 = std::min(y0 + 1, img.h - 1);
            double fx = sx - x0, fy = sy - y0;
            const float* r0 = img.row(y0);
            const float* r1 = img.row(y1);
            double v = r0[x0] * (1 - fx) * (1 - fy) + r0[x1] * fx * (1 - fy) +
                       r1[x0] * (1 - fx) * fy + r1[x1] * fx * fy;
            o[x] = (float)v;
        }
    }
    return out;
}

ImageF warpRGB(const ImageF& img, const Similarity& M, std::vector<float>* weight) {
    Similarity Mi = invertSim(M);
    ImageF out(img.w, img.h, 3);
    if (weight) weight->assign((size_t)img.w * img.h, 0.f);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < img.h; ++y) {
        float* o = out.row(y);
        float* wp = weight ? (weight->data() + (size_t)y * img.w) : nullptr;
        for (int x = 0; x < img.w; ++x) {
            double sx, sy;
            applySim(Mi, x, y, sx, sy);
            if (sx < 0 || sy < 0 || sx > img.w - 1 || sy > img.h - 1) continue;
            int x0 = (int)sx, y0 = (int)sy;
            int x1 = std::min(x0 + 1, img.w - 1), y1 = std::min(y0 + 1, img.h - 1);
            double fx = sx - x0, fy = sy - y0;
            const float* r0 = img.row(y0);
            const float* r1 = img.row(y1);
            for (int c = 0; c < 3; ++c) {
                double v = r0[x0 * 3 + c] * (1 - fx) * (1 - fy) + r0[x1 * 3 + c] * fx * (1 - fy) +
                           r1[x0 * 3 + c] * (1 - fx) * fy + r1[x1 * 3 + c] * fx * fy;
                o[x * 3 + c] = (float)v;
            }
            if (wp) wp[x] = 1.f;
        }
    }
    return out;
}

}  // namespace astro
