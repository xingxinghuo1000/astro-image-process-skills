#include "rawdecode.hpp"

#include "libraw/libraw.h"

namespace astro {

bool decodeHalfGray(const std::string& path, std::vector<float>& gray, int& w, int& h) {
    LibRaw raw;
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) return false;
    raw.imgdata.params.half_size = 1;
    raw.imgdata.params.use_camera_wb = 1;
    raw.imgdata.params.no_auto_bright = 1;
    raw.imgdata.params.output_bps = 8;
    if (raw.unpack() != LIBRAW_SUCCESS) return false;
    if (raw.dcraw_process() != LIBRAW_SUCCESS) return false;
    libraw_processed_image_t* img = raw.dcraw_make_mem_image();
    if (!img) return false;
    w = img->width; h = img->height;
    int nc = img->colors;
    gray.assign((size_t)w * h, 0.f);
    const uint8_t* p = img->data;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        float R = p[i * nc + 0], G = p[i * nc + 1], B = p[i * nc + 2];
        gray[i] = 0.299f * R + 0.587f * G + 0.114f * B;
    }
    LibRaw::dcraw_clear_mem(img);
    return true;
}

bool decodeFullLinear(const std::string& path, ImageF& rgb) {
    LibRaw raw;
    if (raw.open_file(path.c_str()) != LIBRAW_SUCCESS) return false;
    raw.imgdata.params.half_size = 0;
    raw.imgdata.params.use_camera_wb = 1;
    raw.imgdata.params.no_auto_bright = 1;
    raw.imgdata.params.output_bps = 16;
    raw.imgdata.params.gamm[0] = 1.0;
    raw.imgdata.params.gamm[1] = 1.0;
    if (raw.unpack() != LIBRAW_SUCCESS) return false;
    if (raw.dcraw_process() != LIBRAW_SUCCESS) return false;
    libraw_processed_image_t* img = raw.dcraw_make_mem_image();
    if (!img) return false;
    int w = img->width, h = img->height, nc = img->colors;
    rgb = ImageF(w, h, 3);
    const uint16_t* p = reinterpret_cast<const uint16_t*>(img->data);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        rgb.d[i * 3 + 0] = p[i * nc + 0] / 65535.f;
        rgb.d[i * 3 + 1] = p[i * nc + 1] / 65535.f;
        rgb.d[i * 3 + 2] = p[i * nc + 2] / 65535.f;
    }
    LibRaw::dcraw_clear_mem(img);
    return true;
}

}  // namespace astro
