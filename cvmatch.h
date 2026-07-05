/* cvmatch — minimal, dependency-free TM_CCOEFF_NORMED template matching.
 *
 * The algorithm mirrors OpenCV's matchTemplate(TM_CCOEFF_NORMED) pipeline
 * (modules/imgproc/src/templmatch.cpp, identical in 4.8.1 and 5.x):
 *   1. crossCorr(): block-based FFT cross-correlation (overlap-save tiling,
 *      template spectrum computed once per channel).
 *   2. common_matchTemplate(): per-window normalization
 *        num = corr - sum_k(wndSum_k * templMean_k)
 *        den = sqrt(max(wndSum2 - wndMean2, 0)) * templNorm
 *      with OpenCV's exact rounding guards (min(0.5, 10*FLT_EPSILON*wndSum2)
 *      cutoff and the 1.125 saturation band).
 *
 * Unlike OpenCV, window sums are produced by O(width) sliding column sums
 * instead of full double-precision integral images, and the global min/max
 * scan is fused into the normalization pass.
 */
#ifndef CVMATCH_H
#define CVMATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CvmExtrema {
  double min_val, max_val;
  int32_t min_x, min_y, max_x, max_y;
} CvmExtrema;

#define CVM_OK 0
#define CVM_ERR_BADARG (-1)
#define CVM_ERR_NOMEM (-2)

/* Matches tpl inside img with TM_CCOEFF_NORMED.
 *
 * img: interleaved uint8 pixels, row stride in bytes, step bytes per pixel.
 * tpl: same layout; must satisfy 0 < tw <= iw, 0 < th <= ih.
 * cn:  number of leading channels of each pixel to process (1..4, <= step).
 *      cn=3/step=4 processes RGBA pixels while ignoring alpha, which is
 *      bit-equivalent to the 4-channel result whenever each image's alpha
 *      plane is constant (the constant channel contributes exactly zero to
 *      numerator, denominator and templNorm).
 * result: optional caller buffer of (iw-tw+1)*(ih-th+1) floats that receives
 *         the full normalized response map; pass NULL to let the function use
 *         an internal scratch buffer (freed before returning).
 * out: receives min/max values and their first-occurrence locations, matching
 *      OpenCV minMaxLoc row-major scan order.
 */
int cvm_match_ccoeff_normed_u8(const uint8_t *img, int img_stride, int iw,
                               int ih, const uint8_t *tpl, int tpl_stride,
                               int tw, int th, int cn, int step, float *result,
                               CvmExtrema *out);

#ifdef __cplusplus
}
#endif

#endif /* CVMATCH_H */
