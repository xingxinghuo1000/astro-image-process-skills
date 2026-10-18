#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "rawdecode.hpp"
#include "register.hpp"
#include "stack.hpp"
#include "util.hpp"

using namespace astro;
namespace fs = std::filesystem;

static void log(const std::string& s) { std::printf("%s\n", s.c_str()); std::fflush(stdout); }

static double percentileOfVec(std::vector<float>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idxf = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)std::floor(idxf);
    size_t hi = std::min(lo + 1, v.size() - 1);
    double f = idxf - lo;
    return v[lo] * (1 - f) + v[hi] * f;
}

static int countInliers(const std::vector<Point>& mov, const std::vector<Point>& ref,
                        const Similarity& M, double tol) {
    double t2 = tol * tol;
    int n = 0;
    for (const auto& p : mov) {
        double wx, wy;
        applySim(M, p.x, p.y, wx, wy);
        double bd2 = t2;
        for (const auto& q : ref) {
            double dx = wx - q.x, dy = wy - q.y, dd = dx * dx + dy * dy;
            if (dd < bd2) { bd2 = dd; }
        }
        if (bd2 < t2) ++n;
    }
    return n;
}

static Similarity scaleTranslation(const Similarity& M, double s) {
    Similarity R = M;
    R.tx = M.tx * s;
    R.ty = M.ty * s;
    return R;
}

int main(int argc, char** argv) {
    std::string dir = "D:\\pic\\DCIM-2026\\08-shangduhu-2";
    int start = 5994, end = 6012, refArg = 6003;
    std::string out = "out", method = "mean", exclude = "";
    bool saveFrames = false;
    double targetBg = 0.20;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--dir") dir = next();
        else if (a == "--start") start = std::atoi(next().c_str());
        else if (a == "--end") end = std::atoi(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--ref") refArg = std::atoi(next().c_str());
        else if (a == "--method") method = next();
        else if (a == "--exclude") exclude = next();
        else if (a == "--stretch-bg") targetBg = std::atof(next().c_str());
        else if (a == "--save-frames") saveFrames = true;
        else if (a == "--help" || a == "-h") {
            log("usage: astro_align [--dir D] [--start N] [--end N] [--out DIR] [--ref N] "
                "[--method mean|median] [--exclude a,b] [--stretch-bg 0.20] [--save-frames]");
            return 0;
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> excl;
    if (!exclude.empty()) {
        size_t pos = 0;
        while (pos < exclude.size()) {
            size_t comma = exclude.find(',', pos);
            std::string tok = exclude.substr(pos, comma - pos);
            if (!tok.empty()) excl.push_back(std::atoi(tok.c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    std::vector<int> ids;
    std::vector<std::string> paths;
    for (int i = start; i <= end; ++i) {
        bool skip = false;
        for (int e : excl) if (e == i) skip = true;
        if (skip) continue;
        std::string n = "IMG_" + std::to_string(i) + ".CR2";
        std::string p = (fs::path(dir) / n).string();
        if (fs::exists(p)) { ids.push_back(i); paths.push_back(p); }
        else log("  (!) " + n + " not found, skipped");
    }
    if (ids.empty()) { log("no input files"); return 1; }
    log(std::to_string(ids.size()) + " frames: " + std::to_string(ids[0]) + ".." +
        std::to_string(ids.back()));

    int refId = refArg;
    bool haveRef = false;
    for (int i : ids) if (i == refArg) haveRef = true;
    if (!haveRef) refId = ids[ids.size() / 2];
    log("reference frame: IMG_" + std::to_string(refId));

    fs::create_directories(out);
    fs::path framesDir = fs::path(out) / "aligned";
    if (saveFrames) fs::create_directories(framesDir);

    // ---- pass 1: half-res grayscale + static-pattern map ----
    log("pass 1: decoding grayscale previews to remove static pattern ...");
    std::vector<std::vector<float>> grays;
    int gw = 0, gh = 0;
    for (size_t k = 0; k < paths.size(); ++k) {
        std::vector<float> g;
        int w, h;
        if (!decodeHalfGray(paths[k], g, w, h)) { log("  (!) decode failed: " + paths[k]); }
        if (k == 0 || grays.empty()) { gw = w; gh = h; }
        grays.push_back(std::move(g));
    }
    size_t nf = grays.size();

    std::vector<float> staticMap((size_t)gw * gh, 0.f);
    if (nf >= 5) {
        std::vector<float> col(nf);
        for (size_t p = 0; p < (size_t)gw * gh; ++p) {
            for (size_t k = 0; k < nf; ++k) col[k] = grays[k][p];
            staticMap[p] = (float)percentileOfVec(col, 75.0);
        }
    }
    std::vector<std::vector<float>> cleaned(nf);
    for (size_t k = 0; k < nf; ++k) {
        cleaned[k].resize((size_t)gw * gh);
        for (size_t p = 0; p < (size_t)gw * gh; ++p)
            cleaned[k][p] = std::max(grays[k][p] - staticMap[p], 0.f);
    }
    grays.clear();
    log("  static-pattern map built (half-res " + std::to_string(gh) + "x" +
        std::to_string(gw) + ", " + std::to_string(nf) + " frames)");

    double regScale = 0.5;
    int H = gh * 2, W = gw * 2;
    int kRef = 0;
    for (size_t k = 0; k < nf; ++k) if (ids[k] == refId) kRef = (int)k;

    auto toImage = [&](const std::vector<float>& g) {
        ImageF im(gw, gh, 1);
        im.d = g;
        return im;
    };
    ImageF refGray = toImage(cleaned[kRef]);
    ImageF refHp;
    highpass(refGray, refHp, 1.0f, 20.0f);
    std::vector<Point> refStars = detectStars(refHp);
    log("  reference stars: " + std::to_string(refStars.size()));

    std::vector<double> accSum, accW;
    bool haveAcc = false;
    std::vector<std::vector<uint16_t>> medianStore;
    std::vector<int> recordIds;
    std::vector<Similarity> recordM;
    std::vector<double> recordRot, recordTx, recordTy, recordScore;
    std::vector<int> recordInl;

    bool useMean = (method == "mean");
    if (useMean) {
        accSum.assign((size_t)W * H * 3, 0.0);
        accW.assign((size_t)W * H, 0.0);
        haveAcc = true;
    }

    log("pass 2: registering + stacking ...");

    for (size_t k = 0; k < nf; ++k) {
        auto ts = std::chrono::steady_clock::now();
        int idx = ids[k];
        ImageF frameHp;
        std::vector<Point> movStars;
        Similarity Mhalf, Mfull;
        int inliers = 0;
        double sc = 1.0;

        if (idx == refId) {
            Mhalf = Similarity();
            Mfull = Similarity();
        } else {
            highpass(toImage(cleaned[k]), frameHp, 1.0f, 20.0f);
            movStars = detectStars(frameHp);
            Mhalf = matchAndFit(movStars, refStars, 30.0, 3.0);
            Mhalf = refineTransform(Mhalf, movStars, refStars);
            Mfull = scaleTranslation(Mhalf, 1.0 / regScale);
        }

        sc = starScore(refHp, (idx == refId ? refHp : frameHp), Mhalf, 0.55);
        double residMed = 0.0, residFrac = 1.0;
        if (idx != refId) {
            starResidual(refStars, movStars, Mhalf, 1.5, residMed, residFrac);
            inliers = countInliers(movStars, refStars, Mhalf, 3.0);
        }

        bool plausible = true;
        if (idx != refId) {
            double rot = Mhalf.rotDeg();
            plausible = (std::fabs(rot) < 15.0 && std::fabs(Mhalf.tx) < 0.15 * gw &&
                         std::fabs(Mhalf.ty) < 0.15 * gh && inliers >= 25 && sc >= 0.55 &&
                         residFrac >= 0.20 && residMed < 3.0);
        }
        if (!plausible) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "  !! IMG_%d: registration rejected (score=%.2f inl=%d resid=%.2fpx frac=%.2f)",
                          idx, sc, inliers, residMed, residFrac);
            log(buf);
            continue;
        }

        ImageF lin;
        if (!decodeFullLinear(paths[k], lin)) { log("  (!) full decode failed"); continue; }

        std::vector<uint16_t> u16;
        if (useMean) {
            std::vector<float> w;
            ImageF warped = warpRGB(lin, Mfull, &w);
#pragma omp parallel for schedule(static)
            for (long long p = 0; p < (long long)W * H; ++p) {
                double ww = w[p];
                if (ww > 0.f) {
                    accSum[p * 3 + 0] += warped.d[p * 3 + 0] * ww;
                    accSum[p * 3 + 1] += warped.d[p * 3 + 1] * ww;
                    accSum[p * 3 + 2] += warped.d[p * 3 + 2] * ww;
                }
                accW[p] += ww;
            }
        } else {
            ImageF warped = (idx == refId) ? lin : warpRGB(lin, Mfull, nullptr);
            u16.resize((size_t)W * H * 3);
            for (size_t i = 0; i < u16.size(); ++i) {
                float v = std::max(0.f, std::min(warped.d[i], 1.f));
                u16[i] = (uint16_t)std::lround(v * 65535.0);
            }
            medianStore.push_back(std::move(u16));
        }

        if (saveFrames) {
            ImageF warped = (idx == refId) ? lin : warpRGB(lin, Mfull, nullptr);
            std::vector<uint16_t> su;
            toU16(warped, su);
            writeTiff16((framesDir / ("IMG_" + std::to_string(idx) + ".tif")).string(),
                        su, W, H);
        }

        double rot = Mfull.rotDeg();
        char buf[256];
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        std::snprintf(buf, sizeof(buf),
                      "  IMG_%d: inliers=%d score=%.3f rot=%+.3fdeg shift=(%+.2f,%+.2f)px  [%.1fs]",
                      idx, inliers, sc, rot, Mfull.tx, Mfull.ty, dt);
        log(buf);
        recordIds.push_back(idx);
        recordM.push_back(Mfull);
        recordRot.push_back(rot);
        recordTx.push_back(Mfull.tx);
        recordTy.push_back(Mfull.ty);
        recordScore.push_back(sc);
        recordInl.push_back(inliers);
    }

    // ---- stack ----
    log("stacking ...");
    ImageF stackF(W, H, 3);
    if (useMean) {
#pragma omp parallel for schedule(static)
        for (long long p = 0; p < (long long)W * H; ++p) {
            double ww = std::max(accW[p], 1.0);
            stackF.d[p * 3 + 0] = (float)(accSum[p * 3 + 0] / ww);
            stackF.d[p * 3 + 1] = (float)(accSum[p * 3 + 1] / ww);
            stackF.d[p * 3 + 2] = (float)(accSum[p * 3 + 2] / ww);
        }
    } else {
        size_t m = medianStore.size();
        if (m == 0) { log("  (!) no frames to stack"); return 1; }
#pragma omp parallel
        {
            std::vector<uint16_t> colbuf(m);
#pragma omp for schedule(static)
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    size_t p = (size_t)y * W + x;
                    for (int c = 0; c < 3; ++c) {
                        for (size_t i = 0; i < m; ++i) colbuf[i] = medianStore[i][p * 3 + c];
                        std::nth_element(colbuf.begin(), colbuf.begin() + m / 2, colbuf.end());
                        stackF.d[p * 3 + c] = colbuf[m / 2] / 65535.f;
                    }
                }
            }
        }
    }

    std::vector<uint16_t> stack16;
    toU16(stackF, stack16);
    std::string stackPath = (fs::path(out) / ("stack_" + method + ".tif")).string();
    writeTiff16(stackPath, stack16, W, H);
    log("  wrote " + stackPath);

    // preview (downsample + simple arcsinh for the linear stack)
    {
        ImageF pv = downsampleRGB(stackF, W / 2, H / 2);
        double gain = 50.0;
        for (auto& v : pv.d) v = (float)(std::asinh(v * gain) / std::asinh(gain));
        std::vector<uint8_t> u8;
        toU8(pv, u8);
        writePng8((fs::path(out) / "stack_preview.png").string(), u8, pv.w, pv.h);
    }

    double mtfM = 0.0, bg = 0.0;
    ImageF stretched = autoStretch(stackF, targetBg, mtfM, bg);
    std::vector<uint16_t> st16;
    toU16(stretched, st16);
    std::string stPath = (fs::path(out) / ("stack_" + method + "_stretched.tif")).string();
    writeTiff16(stPath, st16, W, H);
    {
        ImageF pv = downsampleRGB(stretched, W / 2, H / 2);
        std::vector<uint8_t> u8;
        toU8(pv, u8);
        writePng8((fs::path(out) / "stretched_preview.png").string(), u8, pv.w, pv.h);
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  wrote %s  (auto-stretch: mtf_m=%.4f, bg=%.4f, target=%.2f)",
                  stPath.c_str(), mtfM, bg, targetBg);
    log(buf);

    // transforms.json
    {
        FILE* f = std::fopen((fs::path(out) / "transforms.json").string().c_str(), "w");
        if (f) {
            std::fprintf(f, "{\n  \"reference\": %d,\n  \"reg_scale\": %.1f,\n  \"frames\": [\n",
                         refId, regScale);
            for (size_t i = 0; i < recordIds.size(); ++i) {
                Similarity& M = recordM[i];
                std::fprintf(f,
                    "    {\"frame\": %d, \"inliers\": %d, \"score\": %.6f, \"rot_deg\": %.6f, "
                    "\"tx\": %.6f, \"ty\": %.6f, \"M\": [[%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f]]}%s\n",
                    recordIds[i], recordInl[i], recordScore[i], recordRot[i], recordTx[i],
                    recordTy[i], M.a, -M.b, M.tx, M.b, M.a, M.ty,
                    (i + 1 < recordIds.size()) ? "," : "");
            }
            std::fprintf(f, "  ]\n}\n");
            std::fclose(f);
        }
    }

    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::snprintf(buf, sizeof(buf), "done in %.1fs -> %s", total, fs::absolute(out).string().c_str());
    log(buf);
    std::snprintf(buf, sizeof(buf), "  %zu frames stacked (%s)", recordIds.size(), method.c_str());
    log(buf);
    return 0;
}
