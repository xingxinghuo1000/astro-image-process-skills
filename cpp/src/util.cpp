#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace astro {

void gaussianBlur1(const ImageF& in, ImageF& out, float sigma) {
    out = ImageF(in.w, in.h, 1);
    if (sigma <= 0.f) { out.d = in.d; return; }
    int r = (int)std::ceil(3.f * sigma);
    std::vector<float> k(2 * r + 1);
    float s = 0.f;
    for (int i = -r; i <= r; ++i) { k[i + r] = std::exp(-(float)(i * i) / (2.f * sigma * sigma)); s += k[i + r]; }
    for (auto& v : k) v /= s;

    std::vector<float> tmp((size_t)in.w * in.h);
    for (int y = 0; y < in.h; ++y) {
        const float* src = in.row(y);
        float* dst = tmp.data() + (size_t)y * in.w;
        for (int x = 0; x < in.w; ++x) {
            float acc = 0.f;
            for (int i = -r; i <= r; ++i) {
                int xx = std::min(std::max(x + i, 0), in.w - 1);
                acc += src[xx] * k[i + r];
            }
            dst[x] = acc;
        }
    }
    for (int y = 0; y < in.h; ++y) {
        float* dst = out.row(y);
        for (int x = 0; x < in.w; ++x) {
            float acc = 0.f;
            for (int i = -r; i <= r; ++i) {
                int yy = std::min(std::max(y + i, 0), in.h - 1);
                acc += tmp[(size_t)yy * in.w + x] * k[i + r];
            }
            dst[x] = acc;
        }
    }
}

ImageF resizeArea1(const ImageF& in, int nw, int nh) {
    ImageF out(nw, nh, 1);
    double sx = (double)in.w / nw, sy = (double)in.h / nh;
    for (int y = 0; y < nh; ++y) {
        int y0 = (int)(y * sy), y1 = std::max(y0 + 1, (int)((y + 1) * sy));
        y1 = std::min(y1, in.h);
        for (int x = 0; x < nw; ++x) {
            int x0 = (int)(x * sx), x1 = std::max(x0 + 1, (int)((x + 1) * sx));
            x1 = std::min(x1, in.w);
            double acc = 0.0; int n = 0;
            for (int yy = y0; yy < y1; ++yy) {
                const float* p = in.row(yy);
                for (int xx = x0; xx < x1; ++xx) { acc += p[xx]; ++n; }
            }
            out.row(y)[x] = n ? (float)(acc / n) : 0.f;
        }
    }
    return out;
}

void highpass(const ImageF& gray, ImageF& out, float preBlur, float hpSigma) {
    ImageF g, bg;
    gaussianBlur1(gray, g, preBlur);
    gaussianBlur1(g, bg, hpSigma);
    out = ImageF(gray.w, gray.h, 1);
    for (size_t i = 0; i < out.d.size(); ++i)
        out.d[i] = std::max(g.d[i] - bg.d[i], 0.f);
}

std::vector<Point> detectStars(const ImageF& hp, float nsigma, int minArea, int maxArea) {
    const int W = hp.w, H = hp.h;
    double sum = 0.0, sum2 = 0.0;
    for (float v : hp.d) { sum += v; sum2 += (double)v * v; }
    double n = (double)hp.d.size();
    double mean = sum / n;
    double var = std::max(0.0, sum2 / n - mean * mean);
    float thr = (float)(mean + nsigma * std::sqrt(var));

    std::vector<uint8_t> mask((size_t)W * H);
    for (size_t i = 0; i < mask.size(); ++i) mask[i] = hp.d[i] > thr ? 1 : 0;

    std::vector<Point> pts;
    std::vector<int> stack;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = (size_t)y * W + x;
            if (!mask[idx]) continue;
            mask[idx] = 0;
            stack.clear();
            stack.push_back((int)idx);
            double sx = 0, sy = 0; int area = 0;
            while (!stack.empty()) {
                int cur = stack.back(); stack.pop_back();
                int cx = cur % W, cy = cur / W;
                sx += cx; sy += cy; ++area;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                        size_t ni = (size_t)ny * W + nx;
                        if (mask[ni]) { mask[ni] = 0; stack.push_back((int)ni); }
                    }
            }
            if (area >= minArea && area <= maxArea)
                pts.push_back({(float)(sx / area), (float)(sy / area)});
        }
    }
    return pts;
}

// ---- 16-bit RGB TIFF (uncompressed, single strip, little-endian) ----
static void put16(std::vector<uint8_t>& b, size_t off, uint16_t v) {
    b[off] = (uint8_t)(v & 0xff); b[off + 1] = (uint8_t)(v >> 8);
}
static void put32(std::vector<uint8_t>& b, size_t off, uint32_t v) {
    b[off] = (uint8_t)(v & 0xff); b[off + 1] = (uint8_t)((v >> 8) & 0xff);
    b[off + 2] = (uint8_t)((v >> 16) & 0xff); b[off + 3] = (uint8_t)((v >> 24) & 0xff);
}

bool writeTiff16(const std::string& path, const std::vector<uint16_t>& rgb, int w, int h) {
    size_t dataBytes = (size_t)w * h * 3 * 2;
    const uint16_t N = 12;
    size_t ifdOff = 8 + dataBytes;
    size_t ifdSize = 2 + (size_t)N * 12 + 4;
    size_t extraOff = ifdOff + ifdSize;
    size_t bitsOff = extraOff;          // 3 SHORT
    size_t xresOff = bitsOff + 6;       // RATIONAL (8)
    size_t yresOff = xresOff + 8;       // RATIONAL (8)
    const char sw[] = "astro_align";
    size_t swOff = yresOff + 8;
    size_t total = swOff + (sizeof(sw) - 1);

    std::vector<uint8_t> b(total, 0);
    // header
    b[0] = 'I'; b[1] = 'I'; put16(b, 2, 42); put32(b, 4, (uint32_t)ifdOff);
    // strip data
    for (size_t i = 0; i < (size_t)w * h * 3; ++i) {
        b[8 + i * 2] = (uint8_t)(rgb[i] & 0xff);
        b[8 + i * 2 + 1] = (uint8_t)(rgb[i] >> 8);
    }
    // extra
    put16(b, bitsOff, 16); put16(b, bitsOff + 2, 16); put16(b, bitsOff + 4, 16);
    put32(b, xresOff, 72); put32(b, xresOff + 4, 1);
    put32(b, yresOff, 72); put32(b, yresOff + 4, 1);
    std::memcpy(&b[swOff], sw, sizeof(sw) - 1);

    put16(b, ifdOff, N);
    auto entry = [&](int i, uint16_t tag, uint16_t type, uint32_t cnt, uint32_t val) {
        size_t o = ifdOff + 2 + (size_t)i * 12;
        put16(b, o, tag); put16(b, o + 2, type); put32(b, o + 4, cnt); put32(b, o + 8, val);
    };
    int i = 0;
    entry(i++, 256, 4, 1, (uint32_t)w);
    entry(i++, 257, 4, 1, (uint32_t)h);
    entry(i++, 258, 3, 3, (uint32_t)bitsOff);
    entry(i++, 259, 3, 1, 1);
    entry(i++, 262, 3, 1, 2);
    entry(i++, 273, 4, 1, 8);
    entry(i++, 277, 3, 1, 3);
    entry(i++, 278, 4, 1, (uint32_t)h);
    entry(i++, 279, 4, 1, (uint32_t)dataBytes);
    entry(i++, 282, 5, 1, (uint32_t)xresOff);
    entry(i++, 283, 5, 1, (uint32_t)yresOff);
    entry(i++, 296, 3, 1, 2);
    put32(b, ifdOff + 2 + (size_t)N * 12, 0);  // next IFD

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t wr = std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
    return wr == b.size();
}

// ---- 8-bit RGB PNG (uncompressed deflate) ----
static uint32_t crc_table[256];
static bool crc_init = [] {
    for (uint32_t n = 0; n < 256; ++n) {
        uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
        crc_table[n] = c;
    }
    return true;
}();
static uint32_t crc32_buf(const uint8_t* d, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = crc_table[(c ^ d[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static uint32_t adler32_buf(const uint8_t* d, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) { a = (a + d[i]) % 65521; b = (b + a) % 65521; }
    return (b << 16) | a;
}
static void push_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24)); v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8)); v.push_back((uint8_t)x);
}
static void png_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    push_be32(out, (uint32_t)data.size());
    size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    uint32_t c = crc32_buf(out.data() + start, 4 + data.size());
    push_be32(out, c);
}

bool writePng8(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    // filtered raw: each row prefixed with filter byte 0
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (1 + (size_t)w * 3));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        const uint8_t* p = rgb.data() + (size_t)y * w * 3;
        raw.insert(raw.end(), p, p + (size_t)w * 3);
    }
    std::vector<uint8_t> z;
    z.push_back(0x78); z.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        bool final = (pos + n >= raw.size());
        z.push_back(final ? 1 : 0);
        z.push_back((uint8_t)(n & 0xff)); z.push_back((uint8_t)(n >> 8));
        uint16_t nl = (uint16_t)(~n & 0xffff);
        z.push_back((uint8_t)(nl & 0xff)); z.push_back((uint8_t)(nl >> 8));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    push_be32(z, adler32_buf(raw.data(), raw.size()));

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    std::vector<uint8_t> ihdr;
    push_be32(ihdr, (uint32_t)w); push_be32(ihdr, (uint32_t)h);
    ihdr.push_back(8); ihdr.push_back(2); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    png_chunk(out, "IHDR", ihdr);
    png_chunk(out, "IDAT", z);
    png_chunk(out, "IEND", {});

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t wr = std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return wr == out.size();
}

}  // namespace astro
