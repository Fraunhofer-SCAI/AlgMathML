/*
 * Copyright (c) 2011, Leo
 *               2018, Jannik Schürg (modifications)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the distribution
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T> inline auto square(const T &n) -> decltype(n * n) {
  return n * n;
}

/**
 * @brief      Compute HOG
 *
 *
 * @param[in]  pixels         Pixel values, column major.
 * @param[in]  nb_bins        The number of bins
 * @param[in]  cwidth         The cell width
 * @param[in]  block_size     The block size
 * @param[in]  unsigned_dirs  Whether to use unsigned directions
 * @param[in]  clip_val       Clip value for block
 * @param[in]  img_size       The image size
 * @param[in]  stride         The stride of the image (distance from one row to
 *                            the next, must be at least image height)
 * @param[out]  out           Output destination, must be at least of size
 *                            computed by `getNumFeatures` below.
 * @param[in]  grayscale      If the image is grayscale
 * @param[in]  channel_stride The distance between one color channel and the next.
 */
static void HoG(const double *pixels, const size_t nb_bins,
                const double cwidth, const size_t block_size,
                const bool unsigned_dirs, const double clip_val,
                const size_t *img_size, const size_t stride,
                double *out, const bool grayscale,
                const size_t channel_stride) {

  constexpr double pi = 3.141592653589793238463;

  const int orient = unsigned_dirs ? 1 : 2;

  const size_t img_width = img_size[1];
  const size_t img_height = img_size[0];

  const int hist1 = 2 + ceil(-0.5 + img_height / cwidth);
  const int hist2 = 2 + ceil(-0.5 + img_width / cwidth);

  const double bin_size = orient * pi / nb_bins;

  double dx[3], dy[3];

  std::vector<std::vector<std::vector<double>>> h(
      hist1, std::vector<std::vector<double>>(
                 hist2, std::vector<double>(nb_bins, 0.0)));
  std::vector<std::vector<std::vector<double>>> block(
      block_size, std::vector<std::vector<double>>(
                      block_size, std::vector<double>(nb_bins, 0.0)));

  // Calculate gradients (zero padding)

  for (size_t y = 0; y < img_height; y++) {
    for (size_t x = 0; x < img_width; x++) {
      if (grayscale) {
        if (x == 0)
          dx[0] = pixels[y + (x + 1) * stride];
        else {
          if (x == img_width - 1)
            dx[0] = -pixels[y + (x - 1) * stride];
          else
            dx[0] = pixels[y + (x + 1) * stride] - pixels[y + (x - 1) * stride];
        }
        if (y == 0)
          dy[0] = -pixels[y + 1 + x * stride];
        else {
          if (y == img_height - 1)
            dy[0] = pixels[y - 1 + x * stride];
          else
            dy[0] = -pixels[y + 1 + x * stride] + pixels[y - 1 + x * stride];
        }
      } else {
        if (x == 0) {
          dx[0] = pixels[y + (x + 1) * stride];
          dx[1] = pixels[y + (x + 1) * stride + channel_stride];
          dx[2] = pixels[y + (x + 1) * stride + 2 * channel_stride];
        } else {
          if (x == img_width - 1) {
            dx[0] = -pixels[y + (x - 1) * stride];
            dx[1] = -pixels[y + (x - 1) * stride + channel_stride];
            dx[2] = -pixels[y + (x - 1) * stride + 2 * channel_stride];
          } else {
            dx[0] = pixels[y + (x + 1) * stride] - pixels[y + (x - 1) * stride];
            dx[1] = pixels[y + (x + 1) * stride + channel_stride] -
                    pixels[y + (x - 1) * stride + channel_stride];
            dx[2] = pixels[y + (x + 1) * stride + 2 * channel_stride] -
                    pixels[y + (x - 1) * stride + 2 * channel_stride];
          }
        }
        if (y == 0) {
          dy[0] = -pixels[y + 1 + x * stride];
          dy[1] = -pixels[y + 1 + x * stride + channel_stride];
          dy[2] = -pixels[y + 1 + x * stride + 2 * channel_stride];
        } else {
          if (y == img_height - 1) {
            dy[0] = pixels[y - 1 + x * stride];
            dy[1] = pixels[y - 1 + x * stride + channel_stride];
            dy[2] = pixels[y - 1 + x * stride + 2 * channel_stride];
          } else {
            dy[0] = -pixels[y + 1 + x * stride] + pixels[y - 1 + x * stride];
            dy[1] = -pixels[y + 1 + x * stride + channel_stride] +
                    pixels[y - 1 + x * stride + channel_stride];
            dy[2] = -pixels[y + 1 + x * stride + 2 * channel_stride] +
                    pixels[y - 1 + x * stride + 2 * channel_stride];
          }
        }
      }

      double grad_mag = sqrt(square(dx[0]) + square(dy[0]));
      double grad_or = atan2(dy[0], dx[0]);

      if (!grayscale) {
        double temp_mag = grad_mag;
        for (size_t cli = 1; cli < 3; ++cli) {
          temp_mag = sqrt(square(dx[cli]) + square(dy[cli]));
          if (temp_mag > grad_mag) {
            grad_mag = temp_mag;
            grad_or = atan2(dy[cli], dx[cli]);
          }
        }
      }

      if (grad_or < 0)
        grad_or += orient * pi;

      // trilinear interpolation

      int bin1 = static_cast<int>(floor(0.5 + grad_or / bin_size) - 1);
      int bin2 = bin1 + 1;
      const int x1 = static_cast<int>(floor(0.5 + x / cwidth));
      const int x2 = x1 + 1;
      const int y1 = static_cast<int>(floor(0.5 + y / cwidth));
      const int y2 = y1 + 1;

      const double Xc = (x1 + 1 - 1.5) * cwidth + 0.5;
      const double Yc = (y1 + 1 - 1.5) * cwidth + 0.5;
      const double Oc = (bin1 + 1 + 1 - 1.5) * bin_size;

      if (bin2 == static_cast<int>(nb_bins)) {
        bin2 = 0;
      }
      if (bin1 < 0) {
        bin1 = nb_bins - 1;
      }

      h[y1][x1][bin1] += grad_mag * (1 - ((x + 1 - Xc) / cwidth)) *
                         (1 - ((y + 1 - Yc) / cwidth)) *
                         (1 - ((grad_or - Oc) / bin_size));
      h[y1][x1][bin2] += grad_mag * (1 - ((x + 1 - Xc) / cwidth)) *
                         (1 - ((y + 1 - Yc) / cwidth)) *
                         (((grad_or - Oc) / bin_size));
      h[y2][x1][bin1] += grad_mag * (1 - ((x + 1 - Xc) / cwidth)) *
                         (((y + 1 - Yc) / cwidth)) *
                         (1 - ((grad_or - Oc) / bin_size));
      h[y2][x1][bin2] += grad_mag * (1 - ((x + 1 - Xc) / cwidth)) *
                         (((y + 1 - Yc) / cwidth)) *
                         (((grad_or - Oc) / bin_size));
      h[y1][x2][bin1] += grad_mag * (((x + 1 - Xc) / cwidth)) *
                         (1 - ((y + 1 - Yc) / cwidth)) *
                         (1 - ((grad_or - Oc) / bin_size));
      h[y1][x2][bin2] += grad_mag * (((x + 1 - Xc) / cwidth)) *
                         (1 - ((y + 1 - Yc) / cwidth)) *
                         (((grad_or - Oc) / bin_size));
      h[y2][x2][bin1] += grad_mag * (((x + 1 - Xc) / cwidth)) *
                         (((y + 1 - Yc) / cwidth)) *
                         (1 - ((grad_or - Oc) / bin_size));
      h[y2][x2][bin2] += grad_mag * (((x + 1 - Xc) / cwidth)) *
                         (((y + 1 - Yc) / cwidth)) *
                         (((grad_or - Oc) / bin_size));
    }
  }

  // Block normalization
  int out_idx = 0;
  for (size_t x = 1; x < hist2 - block_size; x++) {
    for (size_t y = 1; y < hist1 - block_size; y++) {

      double block_norm = 0;
      for (size_t i = 0; i < block_size; i++) {
        for (size_t j = 0; j < block_size; j++) {
          for (size_t k = 0; k < nb_bins; k++) {
            block_norm += square(h[y + i][x + j][k]);
          }
        }
      }
      block_norm = sqrt(block_norm);

      if (block_norm > 0) {
        for (size_t i = 0; i < block_size; i++) {
          for (size_t j = 0; j < block_size; j++) {
            for (size_t k = 0; k < nb_bins; k++) {
              block[i][j][k] = fmin(h[y + i][x + j][k] / block_norm, clip_val);
            }
          }
        }

        block_norm = 0;
        for (size_t i = 0; i < block_size; i++) {
          for (size_t j = 0; j < block_size; j++) {
            for (size_t k = 0; k < nb_bins; k++) {
              block_norm += square(block[i][j][k]);
            }
          }
        }
        block_norm = sqrt(block_norm);

        for (size_t i = 0; i < block_size; i++) {
          for (size_t j = 0; j < block_size; j++) {
            for (size_t k = 0; k < nb_bins; k++) {
              out[out_idx++] = block[i][j][k] / block_norm;
            }
          }
        }
      } else {
        const size_t len = block_size * block_size * nb_bins;
        std::fill_n(out + out_idx, len, 0.0);
        out_idx += len;
      }
    }
  }
}

static size_t getNumFeatures(const size_t *img_size, const size_t nb_bins,
                             const double cwidth, const size_t block_size) {
  const size_t img_width = img_size[1];
  const size_t img_height = img_size[0];

  const size_t hist1 = 2 + ceil(-0.5 + img_height / cwidth);
  const size_t hist2 = 2 + ceil(-0.5 + img_width / cwidth);

  return (hist1 - 2 - (block_size - 1)) * (hist2 - 2 - (block_size - 1)) *
         nb_bins * block_size * block_size;
}
