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

#define CVM_MAX_THREADS 16

typedef struct {
  float re, im;
} cf;

static int next_pow2(int v) {
  int n = 1;
  while (n < v) n <<= 1;
  return n;
}

/* ------------------------------------------------------------ threading -- */
/* Workers receive (ctx, worker_index) and pick their share by static
 * partition, so results never depend on scheduling. If thread creation
 * fails, the share runs inline — same output, just slower. Define
 * CVM_NO_THREADS to build without pthreads. */

#ifndef CVM_NO_THREADS
#include <pthread.h>

typedef struct {
  void (*fn)(void *, int);
  void *ctx;
  int idx;
} CvmThreadArg;

static void *cvm_thread_tramp(void *a) {
  CvmThreadArg *t = (CvmThreadArg *)a;
  t->fn(t->ctx, t->idx);
  return NULL;
}

static void run_parallel(int n, void (*fn)(void *, int), void *ctx) {
  if (n <= 1) {
    fn(ctx, 0);
    return;
  }
  pthread_t th[CVM_MAX_THREADS - 1];
  CvmThreadArg args[CVM_MAX_THREADS];
  int live[CVM_MAX_THREADS] = {0};
  for (int i = 1; i < n; i++) {
    args[i].fn = fn;
    args[i].ctx = ctx;
    args[i].idx = i;
    if (pthread_create(&th[i - 1], NULL, cvm_thread_tramp, &args[i]) == 0)
      live[i] = 1;
    else
      fn(ctx, i);
  }
  fn(ctx, 0);
  for (int i = 1; i < n; i++)
    if (live[i]) pthread_join(th[i - 1], NULL);
}
#else
static void run_parallel(int n, void (*fn)(void *, int), void *ctx) {
  for (int i = 0; i < n; i++) fn(ctx, i);
}
#endif

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
} Plan;

static void plan_free(Plan *p) {
  free(p->twW);
  if (p->twH != p->twW) free(p->twH);
  free(p->brW);
  if (p->brH != p->brW) free(p->brH);
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
  while ((long long)dftW * dftH > (1 << 21)) { /* cap scratch per buffer */
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
  if (!p->twW || !p->brW || !p->twH || !p->brH) {
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
                          cf *spec, cf *z) {
  int n = p->dftW, hw = p->hw, mask = n - 1;
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
  fft_cols(spec, p->dftH, hw, p->twH, p->brH, 0, z);
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
static void block_inverse_emit(const Plan *p, cf *spec, cf *z, float *res,
                               int rw, int x0, int y0, int bw, int bh,
                               int add) {
  fft_cols(spec, p->dftH, p->hw, p->twH, p->brH, 1, z);
  int n = p->dftW, hw = p->hw;
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

/* Tile-parallel raw cross-correlation. Each tile owns a disjoint result
 * region and runs every channel in order (store, then adds), so per-element
 * arithmetic is identical for any worker count. */
typedef struct {
  const uint8_t *img, *tpl;
  size_t istride, tstride;
  int step, cn, tw, th, rw, rh;
  const Plan *p;
  cf *tspec; /* cn spectra, each dftH*hw */
  float *result;
  cf *scratch; /* nw * (dftH*hw + dftW) */
  int ntx, ntiles, nw;
} CorrCtx;

static void corr_worker(void *vc, int w) {
  CorrCtx *c = (CorrCtx *)vc;
  const Plan *p = c->p;
  size_t specN = (size_t)p->dftH * p->hw;
  cf *spec = c->scratch + (size_t)w * (specN + p->dftW);
  cf *z = spec + specN;
  for (int t = w; t < c->ntiles; t += c->nw) {
    int x0 = (t % c->ntx) * p->blockW, y0 = (t / c->ntx) * p->blockH;
    int bw = c->rw - x0 < p->blockW ? c->rw - x0 : p->blockW;
    int bh = c->rh - y0 < p->blockH ? c->rh - y0 : p->blockH;
    for (int k = 0; k < c->cn; k++) {
      block_forward(c->img + k, c->istride, c->step, x0, y0, bw + c->tw - 1,
                    bh + c->th - 1, p, spec, z);
      mul_conj(spec, c->tspec + (size_t)k * specN, specN);
      block_inverse_emit(p, spec, z, c->result, c->rw, x0, y0, bw, bh, k > 0);
    }
  }
}

/* ------------------------------------------------- normalization + scan -- */

/* OpenCV common_matchTemplate() for TM_CCOEFF_NORMED over rows [y0, y1),
 * with O(iw) sliding column sums instead of integral images, fused with the
 * minMaxLoc scan (strict comparisons keep OpenCV's first-occurrence tie
 * semantics). corrF/corrD: exactly one is non-NULL (float32 FFT result or
 * exact double correlation). Because all window statistics are integers held
 * in doubles, a band starting at any y0 computes bit-identical values. */
CVM_HOT
static void normalize_band(const uint8_t *img, size_t istride, int iw, int cn,
                           int step, int tw, int th, int rw, int y0, int y1,
                           const double mean[4], double templNorm,
                           const float *corrF, const double *corrD,
                           float *result, double *colSum, double *colSum2,
                           CvmExtrema *out) {
  double invArea = 1.0 / ((double)tw * th);
  memset(colSum, 0, (size_t)iw * cn * sizeof(double));
  memset(colSum2, 0, (size_t)iw * sizeof(double));
  for (int y = y0; y < y0 + th; y++) {
    const uint8_t *row = img + (size_t)y * istride;
    for (int x = 0; x < iw; x++)
      for (int k = 0; k < cn; k++) {
        double v = row[(size_t)x * step + k];
        colSum[(size_t)x * cn + k] += v;
        colSum2[x] += v * v;
      }
  }

  float minv = FLT_MAX, maxv = -FLT_MAX;
  int minx = 0, miny = y0, maxx = 0, maxy = y0;
  double eps = 10.0 * FLT_EPSILON;

  for (int y = y0;; y++) {
    double s[4] = {0, 0, 0, 0}, s2 = 0;
    for (int x = 0; x < tw; x++) {
      for (int k = 0; k < cn; k++) s[k] += colSum[(size_t)x * cn + k];
      s2 += colSum2[x];
    }
    float *rrow = result + (size_t)y * rw;
    const float *cfr = corrF ? corrF + (size_t)y * rw : NULL;
    const double *cdr = corrD ? corrD + (size_t)y * rw : NULL;
    for (int x = 0;; x++) {
      double num = cfr ? (double)cfr[x] : cdr[x];
      double wndMean2 = 0;
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
      /* Compare the float32 value actually stored in the result map, not
       * the double intermediate — OpenCV's minMaxLoc scans the rounded
       * CV_32F data, and near-ties must resolve the same way. */
      float v = (float)num;
      rrow[x] = v;
      if (v < minv) {
        minv = v;
        minx = x;
        miny = y;
      }
      if (v > maxv) {
        maxv = v;
        maxx = x;
        maxy = y;
      }
      if (x + 1 >= rw) break;
      for (int k = 0; k < cn; k++)
        s[k] += colSum[(size_t)(x + tw) * cn + k] - colSum[(size_t)x * cn + k];
      s2 += colSum2[x + tw] - colSum2[x];
    }
    if (y + 1 >= y1) break;
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

typedef struct {
  const uint8_t *img;
  size_t istride;
  int iw, cn, step, tw, th, rw;
  const double *mean;
  double templNorm;
  const float *corrF;
  const double *corrD;
  float *result;
  double *colSums; /* nb bands * iw*(cn+1) doubles */
  int *bandY;      /* nb+1 boundaries */
  CvmExtrema *ext; /* nb entries */
  int nb;
} NormCtx;

static void norm_worker(void *vc, int w) {
  NormCtx *c = (NormCtx *)vc;
  if (w >= c->nb) return; /* one band per worker */
  double *cs = c->colSums + (size_t)w * c->iw * (c->cn + 1);
  normalize_band(c->img, c->istride, c->iw, c->cn, c->step, c->tw, c->th,
                 c->rw, c->bandY[w], c->bandY[w + 1], c->mean, c->templNorm,
                 c->corrF, c->corrD, c->result, cs,
                 cs + (size_t)c->iw * c->cn, &c->ext[w]);
}

/* ----------------------------------------------- NTT (exact) correlation -- */
/* Cross-correlation computed exactly over Z_p, p = 29*2^57+1 (2-adicity 57),
 * Montgomery arithmetic. Correlation values are true integers bounded by
 * area*255^2 < 2^53, so the double corr buffer holds them exactly. No
 * real-input packing exists in Z_p (a+I*b collapses to one residue), so rows
 * are transformed at full length and columns at full width — slower than the
 * float path by design; the payoff is exactness. */

typedef uint64_t u64;
typedef unsigned __int128 u128;

#define NTT_P 4179340454199820289ULL /* 29*2^57 + 1 */

typedef struct {
  u64 pinv;  /* -p^-1 mod 2^64 */
  u64 r2;    /* 2^128 mod p */
  u64 one;   /* 2^64 mod p (Montgomery 1) */
} Mont;

static u64 mont_mul(const Mont *m, u64 a, u64 b) {
  u128 t = (u128)a * b;
  u64 lo = (u64)t;
  u64 q = lo * m->pinv;
  u128 t2 = t + (u128)q * NTT_P;
  u64 r = (u64)(t2 >> 64);
  if (r >= NTT_P) r -= NTT_P;
  return r;
}

static u64 mont_add(u64 a, u64 b) {
  u64 r = a + b;
  if (r >= NTT_P || r < a) r -= NTT_P;
  return r;
}

static u64 mont_sub(u64 a, u64 b) { return a >= b ? a - b : a + NTT_P - b; }

static u64 mont_pow(const Mont *m, u64 base_m, u64 e) {
  u64 r = m->one, b = base_m;
  while (e) {
    if (e & 1) r = mont_mul(m, r, b);
    b = mont_mul(m, b, b);
    e >>= 1;
  }
  return r;
}

static void mont_init(Mont *m) {
  u64 inv = NTT_P; /* Newton iteration for p^-1 mod 2^64 */
  for (int i = 0; i < 6; i++) inv *= 2 - NTT_P * inv;
  m->pinv = (u64)(0 - inv);
  m->one = (u64)(((u128)1 << 64) % NTT_P);
  m->r2 = (u64)(((u128)m->one * m->one) % NTT_P);
}

static u64 to_mont(const Mont *m, u64 a) { return mont_mul(m, a, m->r2); }
static u64 from_mont(const Mont *m, u64 a) { return mont_mul(m, a, 1); }

/* Primitive root of p (p-1 = 2^57 * 29): smallest g failing both proper
 * subgroup tests. */
static u64 find_generator(const Mont *m) {
  for (u64 g = 2;; g++) {
    u64 gm = to_mont(m, g);
    if (mont_pow(m, gm, (NTT_P - 1) / 2) != m->one &&
        mont_pow(m, gm, (NTT_P - 1) / 29) != m->one)
      return gm;
  }
}

/* wtab[half+j] = w_{2*half}^j (Montgomery), for each power-of-two stage. */
static u64 *make_ntt_tab(const Mont *m, u64 gen_m, int n, int inverse) {
  u64 *tab = (u64 *)malloc((size_t)n * sizeof(u64));
  if (!tab) return NULL;
  tab[0] = m->one;
  for (int half = 1; half < n; half <<= 1) {
    u64 e = (NTT_P - 1) / (u64)(2 * half);
    u64 w = mont_pow(m, gen_m, inverse ? NTT_P - 1 - e : e);
    u64 cur = m->one;
    for (int j = 0; j < half; j++) {
      tab[half + j] = cur;
      cur = mont_mul(m, cur, w);
    }
  }
  return tab;
}

static void ntt(const Mont *m, u64 *a, int n, const u64 *tab, const int *br) {
  for (int i = 0; i < n; i++) {
    int j = br[i];
    if (j > i) {
      u64 t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
  for (int half = 1; half < n; half <<= 1) {
    const u64 *w = tab + half;
    for (int i = 0; i < n; i += half << 1) {
      u64 *p = a + i, *q = p + half;
      for (int j = 0; j < half; j++) {
        u64 v = mont_mul(m, q[j], w[j]);
        u64 u = p[j];
        p[j] = mont_add(u, v);
        q[j] = mont_sub(u, v);
      }
    }
  }
}

static void ntt_cols(const Mont *m, u64 *d, int n, int width, const u64 *tab,
                     const int *br, u64 *rowtmp) {
  size_t rb = (size_t)width * sizeof(u64);
  for (int i = 0; i < n; i++) {
    int j = br[i];
    if (j > i) {
      memcpy(rowtmp, d + (size_t)i * width, rb);
      memcpy(d + (size_t)i * width, d + (size_t)j * width, rb);
      memcpy(d + (size_t)j * width, rowtmp, rb);
    }
  }
  for (int half = 1; half < n; half <<= 1) {
    const u64 *w = tab + half;
    for (int i = 0; i < n; i += half << 1) {
      for (int j = 0; j < half; j++) {
        u64 wj = w[j];
        u64 *p = d + (size_t)(i + j) * width;
        u64 *q = d + (size_t)(i + j + half) * width;
        for (int c = 0; c < width; c++) {
          u64 v = mont_mul(m, q[c], wj);
          u64 u = p[c];
          p[c] = mont_add(u, v);
          q[c] = mont_sub(u, v);
        }
      }
    }
  }
}

typedef struct {
  Mont m;
  u64 *fwdW, *fwdH, *invW, *invH; /* stage tables */
  int *brW, *brH;
  int dftW, dftH, blockW, blockH;
} NttPlan;

static void ntt_plan_free(NttPlan *p) {
  free(p->fwdW);
  free(p->invW);
  if (p->fwdH != p->fwdW) free(p->fwdH);
  if (p->invH != p->invW) free(p->invH);
  free(p->brW);
  if (p->brH != p->brW) free(p->brH);
}

static int ntt_plan_init(NttPlan *p, const Plan *fp) {
  memset(p, 0, sizeof(*p));
  p->dftW = fp->dftW;
  p->dftH = fp->dftH;
  p->blockW = fp->blockW;
  p->blockH = fp->blockH;
  mont_init(&p->m);
  u64 g = find_generator(&p->m);
  p->fwdW = make_ntt_tab(&p->m, g, p->dftW, 0);
  p->invW = make_ntt_tab(&p->m, g, p->dftW, 1);
  if (p->dftH == p->dftW) {
    p->fwdH = p->fwdW;
    p->invH = p->invW;
  } else {
    p->fwdH = make_ntt_tab(&p->m, g, p->dftH, 0);
    p->invH = make_ntt_tab(&p->m, g, p->dftH, 1);
  }
  p->brW = make_bitrev(p->dftW);
  p->brH = p->dftH == p->dftW ? p->brW : make_bitrev(p->dftH);
  if (!p->fwdW || !p->invW || !p->fwdH || !p->invH || !p->brW || !p->brH) {
    ntt_plan_free(p);
    return CVM_ERR_NOMEM;
  }
  return CVM_OK;
}

/* Forward 2D NTT of a uint8 block (channel chan, values converted straight
 * into Montgomery form via a 256-entry LUT). */
static void ntt_block_forward(const NttPlan *p, const u64 *lut,
                              const uint8_t *chan, size_t stride, int step,
                              int x0, int y0, int loadW, int loadH, u64 *spec,
                              u64 *z) {
  int nW = p->dftW;
  for (int r = 0; r < p->dftH; r++) {
    u64 *row = spec + (size_t)r * nW;
    if (r >= loadH) {
      memset(row, 0, (size_t)(p->dftH - r) * nW * sizeof(u64));
      break;
    }
    const uint8_t *src = chan + (size_t)(y0 + r) * stride + (size_t)x0 * step;
    for (int x = 0; x < loadW; x++) row[x] = lut[src[(size_t)x * step]];
    memset(row + loadW, 0, (size_t)(nW - loadW) * sizeof(u64));
    ntt(&p->m, row, nW, p->fwdW, p->brW);
  }
  ntt_cols(&p->m, spec, p->dftH, nW, p->fwdH, p->brH, z);
}

typedef struct {
  const uint8_t *img;
  size_t istride;
  int step, cn, tw, th, rw, rh;
  const NttPlan *p;
  const u64 *lut;
  u64 *tspec; /* cn spectra, each dftH*dftW, pre-scaled by n^-1 */
  double *corrD;
  u64 *scratch; /* nw * (dftH*dftW + dftW) */
  int ntx, ntiles, nw;
} NttCorrCtx;

static void ntt_corr_worker(void *vc, int w) {
  NttCorrCtx *c = (NttCorrCtx *)vc;
  const NttPlan *p = c->p;
  size_t specN = (size_t)p->dftH * p->dftW;
  u64 *spec = c->scratch + (size_t)w * (specN + p->dftW);
  u64 *z = spec + specN;
  for (int t = w; t < c->ntiles; t += c->nw) {
    int x0 = (t % c->ntx) * p->blockW, y0 = (t / c->ntx) * p->blockH;
    int bw = c->rw - x0 < p->blockW ? c->rw - x0 : p->blockW;
    int bh = c->rh - y0 < p->blockH ? c->rh - y0 : p->blockH;
    for (int k = 0; k < c->cn; k++) {
      ntt_block_forward(p, c->lut, c->img + k, c->istride, c->step, x0, y0,
                        bw + c->tw - 1, bh + c->th - 1, spec, z);
      const u64 *ts = c->tspec + (size_t)k * specN;
      for (size_t i = 0; i < specN; i++) spec[i] = mont_mul(&p->m, spec[i], ts[i]);
      /* inverse 2D */
      ntt_cols(&p->m, spec, p->dftH, p->dftW, p->invH, p->brH, z);
      for (int r = 0; r < bh; r++) {
        u64 *row = spec + (size_t)(r + c->th - 1) * p->dftW;
        ntt(&p->m, row, p->dftW, p->invW, p->brW);
        double *o = c->corrD + (size_t)(y0 + r) * c->rw + x0;
        if (k == 0)
          for (int x = 0; x < bw; x++)
            o[x] = (double)from_mont(&p->m, row[x + c->tw - 1]);
        else
          for (int x = 0; x < bw; x++)
            o[x] += (double)from_mont(&p->m, row[x + c->tw - 1]);
      }
    }
  }
}

/* ----------------------------------------------------------- entrypoint -- */

static void template_stats(const uint8_t *tpl, size_t tstride, int tw, int th,
                           int cn, int step, double mean[4],
                           double *templNorm) {
  double invArea = 1.0 / ((double)tw * th);
  double tn = 0;
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
    tn += s2 * invArea - mean[k] * mean[k];
  }
  *templNorm = tn;
}

/* Shared normalization driver: bands the rows over nw workers and merges
 * extrema in band order (preserving first-occurrence tie semantics). */
static int normalize_parallel(const uint8_t *img, size_t istride, int iw,
                              const uint8_t *tpl, size_t tstride, int tw,
                              int th, int cn, int step, int rw, int rh,
                              int nw, const float *corrF, const double *corrD,
                              float *result, CvmExtrema *out) {
  double mean[4];
  double templNorm;
  template_stats(tpl, tstride, tw, th, cn, step, mean, &templNorm);
  if (templNorm < DBL_EPSILON) { /* flat template: OpenCV defines result=1 */
    size_t total = (size_t)rw * rh;
    for (size_t i = 0; i < total; i++) result[i] = 1.0f;
    out->min_val = out->max_val = 1.0;
    out->min_x = out->min_y = out->max_x = out->max_y = 0;
    return CVM_OK;
  }
  templNorm = sqrt(templNorm * ((double)tw * th));

  int nb = nw;
  int maxb = rh / (th > 32 ? th : 32);
  if (maxb < 1) maxb = 1;
  if (nb > maxb) nb = maxb;
  if (nb > rh) nb = rh;

  double *colSums =
      (double *)malloc((size_t)nb * iw * (cn + 1) * sizeof(double));
  if (!colSums) return CVM_ERR_NOMEM;
  int bandY[CVM_MAX_THREADS + 1];
  CvmExtrema ext[CVM_MAX_THREADS];
  for (int b = 0; b <= nb; b++) bandY[b] = (int)((long long)rh * b / nb);

  NormCtx nc = {img,   istride, iw,    cn,      step, tw,  th, rw,
                mean,  templNorm, corrF, corrD, result, colSums, bandY,
                ext,   nb};
  run_parallel(nb, norm_worker, &nc);
  free(colSums);

  CvmExtrema r = ext[0];
  for (int b = 1; b < nb; b++) { /* strict compares keep first occurrence */
    if (ext[b].min_val < r.min_val) {
      r.min_val = ext[b].min_val;
      r.min_x = ext[b].min_x;
      r.min_y = ext[b].min_y;
    }
    if (ext[b].max_val > r.max_val) {
      r.max_val = ext[b].max_val;
      r.max_x = ext[b].max_x;
      r.max_y = ext[b].max_y;
    }
  }
  *out = r;
  return CVM_OK;
}

static int check_args(const uint8_t *img, const uint8_t *tpl,
                      const CvmExtrema *out, int iw, int ih, int tw, int th,
                      int cn, int step, int img_stride, int tpl_stride) {
  if (!img || !tpl || !out || cn < 1 || cn > 4 || step < cn || tw < 1 ||
      th < 1 || tw > iw || th > ih || img_stride < iw * step ||
      tpl_stride < tw * step)
    return CVM_ERR_BADARG;
  return CVM_OK;
}

static int clamp_threads(int n) {
  if (n < 1) n = 1;
  if (n > CVM_MAX_THREADS) n = CVM_MAX_THREADS;
  return n;
}

int cvm_match_ccoeff_normed_u8(const uint8_t *img, int img_stride, int iw,
                               int ih, const uint8_t *tpl, int tpl_stride,
                               int tw, int th, int cn, int step, int nthreads,
                               float *result, CvmExtrema *out) {
  int rc = check_args(img, tpl, out, iw, ih, tw, th, cn, step, img_stride,
                      tpl_stride);
  if (rc != CVM_OK) return rc;
  int nw = clamp_threads(nthreads);

  int rw = iw - tw + 1, rh = ih - th + 1;
  float *res = result;
  if (!res) {
    res = (float *)malloc((size_t)rw * rh * sizeof(float));
    if (!res) return CVM_ERR_NOMEM;
  }

  Plan p;
  rc = plan_init(&p, tw, th, rw, rh);
  if (rc != CVM_OK) goto done_res;

  {
    size_t specN = (size_t)p.dftH * p.hw;
    int ntx = (rw + p.blockW - 1) / p.blockW;
    int nty = (rh + p.blockH - 1) / p.blockH;
    int ntiles = ntx * nty;
    int cw = nw < ntiles ? nw : ntiles; /* corr workers */
    /* keep worker scratch bounded (~64MB) */
    while (cw > 1 && (size_t)cw * (specN + p.dftW) * sizeof(cf) > (64u << 20))
      cw--;

    cf *tspec = (cf *)malloc((size_t)cn * specN * sizeof(cf));
    cf *scratch =
        (cf *)malloc((size_t)cw * (specN + p.dftW) * sizeof(cf));
    if (!tspec || !scratch) {
      free(tspec);
      free(scratch);
      plan_free(&p);
      rc = CVM_ERR_NOMEM;
      goto done_res;
    }
    /* template spectra once per channel, pre-scaled by 1/(dftW*dftH) so the
     * inverse transform needs no extra pass */
    float scale = 1.0f / ((float)p.dftW * (float)p.dftH);
    for (int k = 0; k < cn; k++) {
      cf *ts = tspec + (size_t)k * specN;
      block_forward(tpl + k, (size_t)tpl_stride, step, 0, 0, tw, th, &p, ts,
                    scratch);
      for (size_t i = 0; i < specN; i++) {
        ts[i].re *= scale;
        ts[i].im *= scale;
      }
    }
    CorrCtx cc = {img,   tpl,   (size_t)img_stride, (size_t)tpl_stride,
                  step,  cn,    tw,
                  th,    rw,    rh,
                  &p,    tspec, res,
                  scratch, ntx, ntiles, cw};
    run_parallel(cw, corr_worker, &cc);
    free(tspec);
    free(scratch);
  }
  plan_free(&p);

  rc = normalize_parallel(img, (size_t)img_stride, iw, tpl,
                          (size_t)tpl_stride, tw, th, cn, step, rw, rh, nw,
                          res, NULL, res, out);

done_res:
  if (!result) free(res);
  return rc;
}

int cvm_match_exact_u8(const uint8_t *img, int img_stride, int iw, int ih,
                       const uint8_t *tpl, int tpl_stride, int tw, int th,
                       int cn, int step, int nthreads, float *result,
                       CvmExtrema *out) {
  int rc = check_args(img, tpl, out, iw, ih, tw, th, cn, step, img_stride,
                      tpl_stride);
  if (rc != CVM_OK) return rc;
  int nw = clamp_threads(nthreads);

  int rw = iw - tw + 1, rh = ih - th + 1;
  float *res = result;
  if (!res) {
    res = (float *)malloc((size_t)rw * rh * sizeof(float));
    if (!res) return CVM_ERR_NOMEM;
  }
  double *corrD = (double *)malloc((size_t)rw * rh * sizeof(double));
  if (!corrD) {
    rc = CVM_ERR_NOMEM;
    goto done_res;
  }

  {
    Plan fp;
    rc = plan_init(&fp, tw, th, rw, rh);
    if (rc != CVM_OK) goto done_corr;
    NttPlan p;
    rc = ntt_plan_init(&p, &fp);
    plan_free(&fp);
    if (rc != CVM_OK) goto done_corr;

    size_t specN = (size_t)p.dftH * p.dftW;
    int ntx = (rw + p.blockW - 1) / p.blockW;
    int nty = (rh + p.blockH - 1) / p.blockH;
    int ntiles = ntx * nty;
    int cw = nw < ntiles ? nw : ntiles;
    while (cw > 1 && (size_t)cw * (specN + p.dftW) * sizeof(u64) > (128u << 20))
      cw--;

    u64 lut[256];
    for (int i = 0; i < 256; i++) lut[i] = to_mont(&p.m, (u64)i);

    u64 *tspec = (u64 *)malloc((size_t)cn * specN * sizeof(u64));
    u64 *scratch = (u64 *)malloc((size_t)cw * (specN + p.dftW) * sizeof(u64));
    if (!tspec || !scratch) {
      free(tspec);
      free(scratch);
      ntt_plan_free(&p);
      rc = CVM_ERR_NOMEM;
      goto done_corr;
    }
    /* reversed template spectra (correlation = convolution with reversed
     * kernel), pre-scaled by (dftW*dftH)^-1 mod p */
    u64 ninv = mont_pow(&p.m, to_mont(&p.m, (u64)p.dftW * (u64)p.dftH),
                        NTT_P - 2);
    for (int k = 0; k < cn; k++) {
      u64 *ts = tspec + (size_t)k * specN;
      for (int y = 0; y < p.dftH; y++) {
        u64 *row = ts + (size_t)y * p.dftW;
        if (y >= th) {
          memset(row, 0, (size_t)(p.dftH - y) * p.dftW * sizeof(u64));
          break;
        }
        const uint8_t *src = tpl + (size_t)(th - 1 - y) * tpl_stride + k;
        for (int x = 0; x < tw; x++)
          row[x] = lut[src[(size_t)(tw - 1 - x) * step]];
        memset(row + tw, 0, (size_t)(p.dftW - tw) * sizeof(u64));
        ntt(&p.m, row, p.dftW, p.fwdW, p.brW);
      }
      ntt_cols(&p.m, ts, p.dftH, p.dftW, p.fwdH, p.brH, scratch);
      for (size_t i = 0; i < specN; i++) ts[i] = mont_mul(&p.m, ts[i], ninv);
    }
    NttCorrCtx cc = {img,   (size_t)img_stride, step, cn, tw, th, rw, rh,
                     &p,    lut,               tspec, corrD, scratch,
                     ntx,   ntiles,            cw};
    run_parallel(cw, ntt_corr_worker, &cc);
    free(tspec);
    free(scratch);
    ntt_plan_free(&p);
  }

  rc = normalize_parallel(img, (size_t)img_stride, iw, tpl,
                          (size_t)tpl_stride, tw, th, cn, step, rw, rh, nw,
                          NULL, corrD, res, out);

done_corr:
  free(corrD);
done_res:
  if (!result) free(res);
  return rc;
}
