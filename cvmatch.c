/* See cvmatch.h for the algorithm outline. C99, no dependencies. */
#include "cvmatch.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Function multi-versioning: compile the hot loops for both baseline x86-64
 * and AVX2/FMA (arch=haswell), picked at load time via ifunc. Requires glibc;
 * define CVM_NO_TARGET_CLONES to opt out. */
#if defined(__x86_64__) && defined(__GLIBC__) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(CVM_NO_TARGET_CLONES)
#define CVM_HOT __attribute__((target_clones("arch=haswell", "default")))
#else
#define CVM_HOT
#endif

typedef struct {
  float re, im;
} cf;

static int next_pow2(int v) {
  int n = 1;
  while (n < v) n <<= 1;
  return n;
}

/* ---------------------------------------------------------------- FFT ---- */
/* Iterative radix-2 complex FFT, natural order in and out.
 * tw holds forward twiddles: tw[half + j] = exp(-2*pi*i*j/(2*half)). */

static cf *make_twiddles(int n) {
  cf *tw = (cf *)malloc((size_t)n * sizeof(cf));
  if (!tw) return NULL;
  for (int half = 1; half < n; half <<= 1) {
    double step = -3.14159265358979323846 / (double)half;
    for (int j = 0; j < half; j++) {
      tw[half + j].re = (float)cos(step * j);
      tw[half + j].im = (float)sin(step * j);
    }
  }
  tw[0].re = 1.0f;
  tw[0].im = 0.0f;
  return tw;
}

static int *make_bitrev(int n) {
  int *br = (int *)malloc((size_t)n * sizeof(int));
  if (!br) return NULL;
  br[0] = 0;
  for (int i = 1; i < n; i++) br[i] = (br[i >> 1] >> 1) | ((i & 1) ? n >> 1 : 0);
  return br;
}

CVM_HOT
static void fft(cf *a, int n, const cf *tw, const int *br, int inverse) {
  for (int i = 0; i < n; i++) {
    int j = br[i];
    if (j > i) {
      cf t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
  float s = inverse ? -1.0f : 1.0f;
  for (int half = 1; half < n; half <<= 1) {
    const cf *w = tw + half;
    for (int i = 0; i < n; i += half << 1) {
      cf *p = a + i, *q = p + half;
      for (int j = 0; j < half; j++) {
        float wr = w[j].re, wi = s * w[j].im;
        float vr = q[j].re * wr - q[j].im * wi;
        float vi = q[j].re * wi + q[j].im * wr;
        float ur = p[j].re, ui = p[j].im;
        p[j].re = ur + vr;
        p[j].im = ui + vi;
        q[j].re = ur - vr;
        q[j].im = ui - vi;
      }
    }
  }
}

/* FFT along columns of a row-major [n x width] complex array. The butterfly
 * inner loop runs across contiguous row elements, so it vectorizes and stays
 * cache friendly. rowtmp must hold width elements. */
CVM_HOT
static void fft_cols(cf *d, int n, int width, const cf *tw, const int *br,
                     int inverse, cf *rowtmp) {
  size_t rb = (size_t)width * sizeof(cf);
  for (int i = 0; i < n; i++) {
    int j = br[i];
    if (j > i) {
      memcpy(rowtmp, d + (size_t)i * width, rb);
      memcpy(d + (size_t)i * width, d + (size_t)j * width, rb);
      memcpy(d + (size_t)j * width, rowtmp, rb);
    }
  }
  float s = inverse ? -1.0f : 1.0f;
  for (int half = 1; half < n; half <<= 1) {
    const cf *w = tw + half;
    for (int i = 0; i < n; i += half << 1) {
      for (int j = 0; j < half; j++) {
        float wr = w[j].re, wi = s * w[j].im;
        cf *p = d + (size_t)(i + j) * width;
        cf *q = d + (size_t)(i + j + half) * width;
        for (int c = 0; c < width; c++) {
          float vr = q[c].re * wr - q[c].im * wi;
          float vi = q[c].re * wi + q[c].im * wr;
          float ur = p[c].re, ui = p[c].im;
          p[c].re = ur + vr;
          p[c].im = ui + vi;
          q[c].re = ur - vr;
          q[c].im = ui - vi;
        }
      }
    }
  }
}

/* ------------------------------------------------------ FFT plan/blocks -- */

typedef struct {
  int dftW, dftH, hw; /* hw = dftW/2 + 1 spectrum width */
  int blockW, blockH; /* result tile covered per block */
  cf *twW, *twH;
  int *brW, *brH;
  cf *spec, *tspec; /* dftH x hw spectra (image block / template) */
  cf *ztmp;         /* dftW scratch row */
} Plan;

static void plan_free(Plan *p) {
  free(p->twW);
  if (p->twH != p->twW) free(p->twH);
  free(p->brW);
  if (p->brH != p->brW) free(p->brH);
  free(p->spec);
  free(p->tspec);
  free(p->ztmp);
}

/* OpenCV crossCorr block sizing (blockScale=4.5, minBlockSize=256), with DFT
 * dims rounded up to powers of two and a total-area cap that bounds scratch
 * memory for very large templates. */
static int plan_init(Plan *p, int tw, int th, int rw, int rh) {
  memset(p, 0, sizeof(*p));
  int bw = (int)(tw * 4.5 + 0.5), bh = (int)(th * 4.5 + 0.5);
  if (bw < 256 - tw + 1) bw = 256 - tw + 1;
  if (bh < 256 - th + 1) bh = 256 - th + 1;
  if (bw > rw) bw = rw;
  if (bh > rh) bh = rh;
  int dftW = next_pow2(bw + tw - 1), dftH = next_pow2(bh + th - 1);
  if (dftW < 2) dftW = 2;
  if (dftH < 2) dftH = 2;
  while ((long long)dftW * dftH > (1 << 21)) { /* cap scratch at ~16MB/buffer */
    if (dftW >= dftH && (dftW >> 1) >= tw + 1)
      dftW >>= 1;
    else if ((dftH >> 1) >= th + 1)
      dftH >>= 1;
    else
      break;
  }
  p->dftW = dftW;
  p->dftH = dftH;
  p->hw = dftW / 2 + 1;
  p->blockW = dftW - tw + 1;
  if (p->blockW > rw) p->blockW = rw;
  p->blockH = dftH - th + 1;
  if (p->blockH > rh) p->blockH = rh;

  p->twW = make_twiddles(dftW);
  p->brW = make_bitrev(dftW);
  p->twH = dftH == dftW ? p->twW : make_twiddles(dftH);
  p->brH = dftH == dftW ? p->brW : make_bitrev(dftH);
  p->spec = (cf *)malloc((size_t)dftH * p->hw * sizeof(cf));
  p->tspec = (cf *)malloc((size_t)dftH * p->hw * sizeof(cf));
  p->ztmp = (cf *)malloc((size_t)dftW * sizeof(cf));
  if (!p->twW || !p->brW || !p->twH || !p->brH || !p->spec || !p->tspec ||
      !p->ztmp) {
    plan_free(p);
    return CVM_ERR_NOMEM;
  }
  return CVM_OK;
}

/* Real 2D forward DFT of one channel of a uint8 block into spec[dftH][hw].
 * Two real rows are packed into one complex FFT (re/im) and untangled, so
 * only ceil(loadH/2) row FFTs run; fully-zero padding rows are memset. */
CVM_HOT
static void block_forward(const uint8_t *chan, size_t stride, int step, int x0,
                          int y0, int loadW, int loadH, const Plan *p,
                          cf *spec) {
  int n = p->dftW, hw = p->hw, mask = n - 1;
  cf *z = p->ztmp;
  for (int r = 0; r < p->dftH; r += 2) {
    cf *sa = spec + (size_t)r * hw;
    cf *sb = sa + hw;
    if (r >= loadH) {
      memset(sa, 0, (size_t)(p->dftH - r) * hw * sizeof(cf));
      break;
    }
    const uint8_t *ra = chan + (size_t)(y0 + r) * stride + (size_t)x0 * step;
    if (r + 1 < loadH) {
      const uint8_t *rb = ra + stride;
      for (int x = 0; x < loadW; x++) {
        z[x].re = (float)ra[(size_t)x * step];
        z[x].im = (float)rb[(size_t)x * step];
      }
    } else {
      for (int x = 0; x < loadW; x++) {
        z[x].re = (float)ra[(size_t)x * step];
        z[x].im = 0.0f;
      }
    }
    memset(z + loadW, 0, (size_t)(n - loadW) * sizeof(cf));
    fft(z, n, p->twW, p->brW, 0);
    for (int k = 0; k < hw; k++) {
      cf zk = z[k], zn = z[(n - k) & mask];
      sa[k].re = 0.5f * (zk.re + zn.re);
      sa[k].im = 0.5f * (zk.im - zn.im);
      sb[k].re = 0.5f * (zk.im + zn.im);
      sb[k].im = 0.5f * (zn.re - zk.re);
    }
  }
  fft_cols(spec, p->dftH, hw, p->twH, p->brH, 0, p->ztmp);
}

/* spec *= conj(tspec), pointwise. */
CVM_HOT
static void mul_conj(cf *spec, const cf *tspec, size_t count) {
  for (size_t i = 0; i < count; i++) {
    float ar = spec[i].re, ai = spec[i].im;
    float br = tspec[i].re, bi = tspec[i].im;
    spec[i].re = ar * br + ai * bi;
    spec[i].im = ai * br - ar * bi;
  }
}

/* Inverse 2D DFT of spec and add/store the top-left bh x bw corner into the
 * result tile at (x0, y0). Two output rows come out of each row IFFT. */
CVM_HOT
static void block_inverse_emit(const Plan *p, cf *spec, float *res, int rw,
                               int x0, int y0, int bw, int bh, int add) {
  fft_cols(spec, p->dftH, p->hw, p->twH, p->brH, 1, p->ztmp);
  int n = p->dftW, hw = p->hw;
  cf *z = p->ztmp;
  for (int r = 0; r < bh; r += 2) {
    const cf *sa = spec + (size_t)r * hw;
    const cf *sb = sa + hw;
    for (int k = 0; k < hw; k++) {
      z[k].re = sa[k].re - sb[k].im;
      z[k].im = sa[k].im + sb[k].re;
    }
    for (int k = hw; k < n; k++) {
      int m = n - k;
      z[k].re = sa[m].re + sb[m].im;
      z[k].im = sb[m].re - sa[m].im;
    }
    fft(z, n, p->twW, p->brW, 1);
    float *o = res + (size_t)(y0 + r) * rw + x0;
    if (add) {
      for (int x = 0; x < bw; x++) o[x] += z[x].re;
      if (r + 1 < bh)
        for (int x = 0; x < bw; x++) o[x + rw] += z[x].im;
    } else {
      for (int x = 0; x < bw; x++) o[x] = z[x].re;
      if (r + 1 < bh)
        for (int x = 0; x < bw; x++) o[x + rw] = z[x].im;
    }
  }
}

/* Raw cross-correlation of one channel accumulated into result. */
static void cross_corr_channel(const uint8_t *img, size_t istride,
                               const uint8_t *tpl, size_t tstride, int step,
                               int tw, int th, int rw, int rh, const Plan *p,
                               int add, float *result) {
  /* Template spectrum once, pre-scaled by 1/(dftW*dftH) so the inverse
   * transform needs no extra normalization pass. */
  block_forward(tpl, tstride, step, 0, 0, tw, th, p, p->tspec);
  float scale = 1.0f / ((float)p->dftW * (float)p->dftH);
  size_t count = (size_t)p->dftH * p->hw;
  for (size_t i = 0; i < count; i++) {
    p->tspec[i].re *= scale;
    p->tspec[i].im *= scale;
  }
  for (int y0 = 0; y0 < rh; y0 += p->blockH) {
    int bh = rh - y0 < p->blockH ? rh - y0 : p->blockH;
    for (int x0 = 0; x0 < rw; x0 += p->blockW) {
      int bw = rw - x0 < p->blockW ? rw - x0 : p->blockW;
      block_forward(img, istride, step, x0, y0, bw + tw - 1, bh + th - 1, p,
                    p->spec);
      mul_conj(p->spec, p->tspec, count);
      block_inverse_emit(p, p->spec, result, rw, x0, y0, bw, bh, add);
    }
  }
}

/* ------------------------------------------------- normalization + scan -- */

/* OpenCV common_matchTemplate() for TM_CCOEFF_NORMED, but with O(iw) sliding
 * column sums instead of full integral images, fused with the minMaxLoc scan
 * (strict comparisons keep OpenCV's first-occurrence tie semantics). */
CVM_HOT
static void normalize_and_scan(const uint8_t *img, size_t istride, int iw,
                               const uint8_t *tpl, size_t tstride, int tw,
                               int th, int cn, int step, int rw, int rh,
                               float *result, double *colSum, double *colSum2,
                               CvmExtrema *out) {
  double area = (double)tw * th, invArea = 1.0 / area;
  double mean[4] = {0, 0, 0, 0};
  double templNorm = 0;

  for (int k = 0; k < cn; k++) {
    double s = 0, s2 = 0;
    for (int y = 0; y < th; y++) {
      const uint8_t *row = tpl + (size_t)y * tstride + k;
      for (int x = 0; x < tw; x++) {
        double v = row[(size_t)x * step];
        s += v;
        s2 += v * v;
      }
    }
    mean[k] = s * invArea;
    templNorm += s2 * invArea - mean[k] * mean[k];
  }
  if (templNorm < DBL_EPSILON) { /* flat template: OpenCV defines result=1 */
    size_t total = (size_t)rw * rh;
    for (size_t i = 0; i < total; i++) result[i] = 1.0f;
    out->min_val = out->max_val = 1.0;
    out->min_x = out->min_y = out->max_x = out->max_y = 0;
    return;
  }
  templNorm = sqrt(templNorm * area);

  memset(colSum, 0, (size_t)iw * cn * sizeof(double));
  memset(colSum2, 0, (size_t)iw * sizeof(double));
  for (int y = 0; y < th; y++) {
    const uint8_t *row = img + (size_t)y * istride;
    for (int x = 0; x < iw; x++)
      for (int k = 0; k < cn; k++) {
        double v = row[(size_t)x * step + k];
        colSum[(size_t)x * cn + k] += v;
        colSum2[x] += v * v;
      }
  }

  double minv = DBL_MAX, maxv = -DBL_MAX;
  int minx = 0, miny = 0, maxx = 0, maxy = 0;
  double eps = 10.0 * FLT_EPSILON;

  for (int y = 0;; y++) {
    double s[4] = {0, 0, 0, 0}, s2 = 0;
    for (int x = 0; x < tw; x++) {
      for (int k = 0; k < cn; k++) s[k] += colSum[(size_t)x * cn + k];
      s2 += colSum2[x];
    }
    float *rrow = result + (size_t)y * rw;
    for (int x = 0;; x++) {
      double num = rrow[x], wndMean2 = 0;
      for (int k = 0; k < cn; k++) {
        double t = s[k];
        wndMean2 += t * t;
        num -= t * mean[k];
      }
      wndMean2 *= invArea;
      double diff2 = s2 - wndMean2;
      if (diff2 < 0) diff2 = 0;
      double lim = 0.5 < eps * s2 ? 0.5 : eps * s2;
      double den = diff2 <= lim ? 0.0 : sqrt(diff2) * templNorm;
      if (fabs(num) < den)
        num /= den;
      else if (fabs(num) < den * 1.125)
        num = num > 0 ? 1 : -1;
      else
        num = 0;
      rrow[x] = (float)num;
      if (num < minv) {
        minv = num;
        minx = x;
        miny = y;
      }
      if (num > maxv) {
        maxv = num;
        maxx = x;
        maxy = y;
      }
      if (x + 1 >= rw) break;
      for (int k = 0; k < cn; k++)
        s[k] += colSum[(size_t)(x + tw) * cn + k] - colSum[(size_t)x * cn + k];
      s2 += colSum2[x + tw] - colSum2[x];
    }
    if (y + 1 >= rh) break;
    const uint8_t *sub = img + (size_t)y * istride;
    const uint8_t *addr = img + (size_t)(y + th) * istride;
    for (int x = 0; x < iw; x++)
      for (int k = 0; k < cn; k++) {
        double a = addr[(size_t)x * step + k], b = sub[(size_t)x * step + k];
        colSum[(size_t)x * cn + k] += a - b;
        colSum2[x] += a * a - b * b;
      }
  }

  out->min_val = minv;
  out->max_val = maxv;
  out->min_x = minx;
  out->min_y = miny;
  out->max_x = maxx;
  out->max_y = maxy;
}

/* ----------------------------------------------------------- entrypoint -- */

int cvm_match_ccoeff_normed_u8(const uint8_t *img, int img_stride, int iw,
                               int ih, const uint8_t *tpl, int tpl_stride,
                               int tw, int th, int cn, int step, float *result,
                               CvmExtrema *out) {
  if (!img || !tpl || !out || cn < 1 || cn > 4 || step < cn || tw < 1 ||
      th < 1 || tw > iw || th > ih || img_stride < iw * step ||
      tpl_stride < tw * step)
    return CVM_ERR_BADARG;

  int rw = iw - tw + 1, rh = ih - th + 1;
  float *res = result;
  if (!res) {
    res = (float *)malloc((size_t)rw * rh * sizeof(float));
    if (!res) return CVM_ERR_NOMEM;
  }
  double *colSum = (double *)malloc((size_t)iw * (cn + 1) * sizeof(double));
  if (!colSum) {
    if (!result) free(res);
    return CVM_ERR_NOMEM;
  }

  Plan p;
  int rc = plan_init(&p, tw, th, rw, rh);
  if (rc != CVM_OK) {
    free(colSum);
    if (!result) free(res);
    return rc;
  }
  for (int k = 0; k < cn; k++)
    cross_corr_channel(img + k, (size_t)img_stride, tpl + k,
                       (size_t)tpl_stride, step, tw, th, rw, rh, &p, k > 0,
                       res);
  plan_free(&p);

  normalize_and_scan(img, (size_t)img_stride, iw, tpl, (size_t)tpl_stride, tw,
                     th, cn, step, rw, rh, res, colSum,
                     colSum + (size_t)iw * cn, out);
  free(colSum);
  if (!result) free(res);
  return CVM_OK;
}
