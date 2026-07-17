/* See cvmatch.h for the algorithm outline. C99, no dependencies. */
#include "cvmatch.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Results must not depend on the compiler's FMA-contraction mood: every
 * float/double op rounds individually (this also keeps the C and Go cores'
 * arithmetic aligned). GCC in -std=c99 mode already behaves this way; the
 * pragmas pin it for compilers/modes where contraction defaults on. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

/* Function multi-versioning: compile the hot loops for both baseline x86-64
 * and AVX2/FMA (arch=haswell), picked at load time via ifunc. Requires glibc;
 * define CVM_NO_TARGET_CLONES to opt out. */
#if defined(__x86_64__) && defined(__GLIBC__) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(CVM_NO_TARGET_CLONES)
#define CVM_CLONES __attribute__((target_clones("arch=haswell", "default")))
#else
#define CVM_CLONES
#endif

#define CVM_HOT CVM_CLONES

/* The normalize scan's guarded sqrt/divide cannot auto-vectorize under
 * default errno/trapping models, so an explicit AVX2 kernel (runtime
 * dispatched, scalar fallback elsewhere) handles it. Every vector lane
 * performs exactly the scalar op sequence — vsqrtpd/vdivpd are correctly
 * rounded IEEE ops — so results are bit-identical to the scalar path. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(CVM_NO_AVX2)
#define CVM_X86_KERNEL 1
#include <immintrin.h>
#endif

#define CVM_MAX_THREADS 16

/* Result-row chunk for the normalize scan: window sums are spilled per
 * chunk (not per row) so the spill buffer stays L1-resident. */
#define CVM_NORM_CHUNK 256

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
#include <unistd.h>

/* Persistent worker pool: workers are created on demand (up to the largest
 * n-1 ever requested), then parked on a condvar between jobs, so a call
 * costs two futex hops instead of pthread_create/join per worker. Share
 * assignment stays the static index partition. A forked child never sees
 * the parent's workers, so it detects the pid change and falls back to
 * spawn-per-call; if thread creation fails anywhere, the caller runs the
 * missing shares inline. Outputs are identical on every path. */
typedef struct {
  void (*fn)(void *, int);
  void *ctx;
  int n;                  /* shares in the current job */
  unsigned long long gen; /* job generation counter */
  int done;               /* worker shares finished for current gen */
  int nthreads;           /* live pool workers */
  int busy;               /* a job is in flight */
} CvmPool;

static CvmPool cvm_pool;
static pthread_mutex_t cvm_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cvm_pool_job = PTHREAD_COND_INITIALIZER;
static pthread_cond_t cvm_pool_done = PTHREAD_COND_INITIALIZER;
static pid_t cvm_pool_pid;

static void *cvm_pool_worker(void *arg) {
  int idx = (int)(intptr_t)arg; /* this worker owns share idx (1-based) */
  unsigned long long seen = 0;
  pthread_mutex_lock(&cvm_pool_mu);
  for (;;) {
    while (cvm_pool.gen == seen || idx >= cvm_pool.n)
      pthread_cond_wait(&cvm_pool_job, &cvm_pool_mu);
    seen = cvm_pool.gen;
    void (*fn)(void *, int) = cvm_pool.fn;
    void *ctx = cvm_pool.ctx;
    pthread_mutex_unlock(&cvm_pool_mu);
    fn(ctx, idx);
    pthread_mutex_lock(&cvm_pool_mu);
    if (++cvm_pool.done == cvm_pool.n - 1) pthread_cond_signal(&cvm_pool_done);
  }
  return NULL;
}

/* Spawn-per-call fallback (pool unavailable: forked child or create failure). */
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

static void run_parallel_spawn(int n, void (*fn)(void *, int), void *ctx) {
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

static void run_parallel(int n, void (*fn)(void *, int), void *ctx) {
  if (n <= 1) {
    fn(ctx, 0);
    return;
  }
  pid_t pid = getpid();
  pthread_mutex_lock(&cvm_pool_mu);
  if (cvm_pool.nthreads == 0) cvm_pool_pid = pid;
  if (cvm_pool.busy || cvm_pool_pid != pid) {
    /* Pool taken by a concurrent caller, or we are a forked child that
     * inherited no workers: run this call on its own threads instead. */
    pthread_mutex_unlock(&cvm_pool_mu);
    run_parallel_spawn(n, fn, ctx);
    return;
  }
  while (cvm_pool.nthreads < n - 1) { /* grow pool to n-1 workers */
    pthread_t th;
    if (pthread_create(&th, NULL, cvm_pool_worker,
                       (void *)(intptr_t)(cvm_pool.nthreads + 1)) != 0)
      break;
    pthread_detach(th);
    cvm_pool.nthreads++;
  }
  int pooled = cvm_pool.nthreads + 1; /* shares the pool can cover */
  if (pooled > n) pooled = n;
  cvm_pool.busy = 1;
  cvm_pool.fn = fn;
  cvm_pool.ctx = ctx;
  cvm_pool.n = pooled;
  cvm_pool.done = 0;
  cvm_pool.gen++;
  pthread_cond_broadcast(&cvm_pool_job);
  pthread_mutex_unlock(&cvm_pool_mu);
  fn(ctx, 0);
  for (int i = pooled; i < n; i++) fn(ctx, i); /* shares beyond the pool */
  pthread_mutex_lock(&cvm_pool_mu);
  while (cvm_pool.done < pooled - 1)
    pthread_cond_wait(&cvm_pool_done, &cvm_pool_mu);
  cvm_pool.busy = 0;
  pthread_mutex_unlock(&cvm_pool_mu);
}
#else
static void run_parallel(int n, void (*fn)(void *, int), void *ctx) {
  for (int i = 0; i < n; i++) fn(ctx, i);
}
#endif

/* ---------------------------------------------------------------- FFT ---- */
/* Iterative radix-2 complex FFT, natural order in and out.
 * tw holds forward twiddles: tw[half + j] = exp(-2*pi*i*j/(2*half)). */

/* Deterministic cos/sin of pi*j/half (0 <= j < half, half a power of two):
 * exact dyadic range reduction plus fdlibm-style minimax kernels in plain
 * sequenced double ops — no libm trig. The Go core runs the identical op
 * sequence, so both cores' twiddle tables (and therefore their outputs)
 * are bit-identical on every platform regardless of the system libm. */
static void sincospi_frac(int j, int half, float *cs, float *sn) {
  int m = 2 * j;      /* pi*j/half = k*(pi/2) + u*(pi/2), u in [0,1) */
  int k = m / half;   /* 0 or 1 */
  int rem = m - k * half;
  double u = (double)rem / (double)half; /* exact: half is a power of two */
  int swap = 0;
  if (u > 0.5) { /* co-function fold, exact dyadic subtraction */
    u = 1.0 - u;
    swap = 1;
  }
  double y = u * 1.57079632679489661923; /* u*(pi/2), one rounding */
  double z = y * y;
  double r = z * (4.16666666666666019037e-02 +
                  z * (-1.38888888888741095749e-03 +
                       z * (2.48015872894767294178e-05 +
                            z * (-2.75573143513906633035e-07 +
                                 z * (2.08757232129817482790e-09 +
                                      z * -1.13596475577881948265e-11)))));
  double hz = 0.5 * z;
  double w = 1.0 - hz;
  double c = w + (((1.0 - w) - hz) + z * r);
  double r2 = 8.33333333332248946124e-03 +
              z * (-1.98412698298579493134e-04 +
                   z * (2.75573137070700676789e-06 +
                        z * (-2.50507602534068634195e-08 +
                             z * 1.58969099521155010221e-10)));
  double v = z * y;
  double s = y + v * (-1.66666666666666324348e-01 + z * r2);
  if (swap) {
    double t = c;
    c = s;
    s = t;
  }
  if (k) { /* rotate by pi/2 */
    double t = c;
    c = -s;
    s = t;
  }
  *cs = (float)c;
  *sn = (float)s;
}

static cf *make_twiddles(int n) {
  cf *tw = (cf *)malloc((size_t)n * sizeof(cf));
  if (!tw) return NULL;
  for (int half = 1; half < n; half <<= 1) {
    for (int j = 0; j < half; j++) {
      float c, s;
      sincospi_frac(j, half, &c, &s);
      tw[half + j].re = c;
      tw[half + j].im = -s; /* forward twiddle: exp(-i*pi*j/half) */
    }
  }
  tw[0].re = 1.0f;
  tw[0].im = 0.0f;
  return tw;
}

/* Bit-reversal permutation as a precomputed list of swap pairs (i < br[i]),
 * terminated by the pair count in pairs[0]; iterating pairs avoids the
 * branchy per-element compare of the classic loop. Pure data movement —
 * results are unaffected. */
static int *make_swap_pairs(int n) {
  int *pairs = (int *)malloc((size_t)(n + 1) * sizeof(int));
  if (!pairs) return NULL;
  int np = 0, br = 0;
  for (int i = 0; i < n; i++) {
    if (br > i) {
      pairs[1 + 2 * np] = i;
      pairs[2 + 2 * np] = br;
      np++;
    }
    /* increment br in reversed order */
    int bit = n >> 1;
    while (br & bit) {
      br ^= bit;
      bit >>= 1;
    }
    br |= bit;
  }
  pairs[0] = np;
  return pairs;
}

#ifdef CVM_X86_KERNEL
/* All radix-2 stages with half >= 8, interleaved complex layout, 4
 * butterflies per vector. Each lane executes exactly the scalar butterfly
 * ops: t1 = q*wr and t2 = swap(q)*wi are one rounded multiply per
 * component, addsub gives (t1-t2, t1+t2) — the same single-rounded add/sub
 * the scalar code performs (the vi sum is commuted, which IEEE addition
 * permits bit-exactly). The twiddle sign flip (s = -1) is an exact xor. */
__attribute__((target("avx2"))) static void fft_stages_avx2(
    cf *restrict a, int n, const cf *restrict tw, int inverse) {
  const __m256 signmask = _mm256_castsi256_ps(
      _mm256_set1_epi32(inverse ? (int)0x80000000 : 0));
  for (int half = 8; half < n; half <<= 1) {
    const cf *restrict w = tw + half;
    for (int i = 0; i < n; i += half << 1) {
      cf *restrict p = a + i;
      cf *restrict q = p + half;
      for (int j = 0; j + 4 <= half; j += 4) {
        __m256 qv = _mm256_loadu_ps((const float *)(q + j));
        __m256 wv = _mm256_loadu_ps((const float *)(w + j));
        __m256 wr = _mm256_moveldup_ps(wv);
        __m256 wi = _mm256_xor_ps(_mm256_movehdup_ps(wv), signmask);
        __m256 qs = _mm256_permute_ps(qv, 0xB1);
        __m256 v = _mm256_addsub_ps(_mm256_mul_ps(qv, wr),
                                    _mm256_mul_ps(qs, wi));
        __m256 pv = _mm256_loadu_ps((const float *)(p + j));
        _mm256_storeu_ps((float *)(p + j), _mm256_add_ps(pv, v));
        _mm256_storeu_ps((float *)(q + j), _mm256_sub_ps(pv, v));
      }
    }
  }
}
#endif /* CVM_X86_KERNEL */

CVM_HOT
static void fft(cf *restrict a, int n, const cf *restrict tw,
                const int *restrict pairs, int inverse) {
  int np = pairs[0];
  for (int k = 0; k < np; k++) {
    int i = pairs[1 + 2 * k], j = pairs[2 + 2 * k];
    cf t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
  float s = inverse ? -1.0f : 1.0f;
  /* The half=1 and half=2 stages are flattened into single loops with the
   * (tiny) twiddle sets hoisted so they vectorize across butterflies; each
   * element still sees exactly the arithmetic of the generic stage loop
   * below, so results are bit-identical. */
  if (n >= 2) {
    const float w0r = tw[1].re, w0i = s * tw[1].im;
    for (int i = 0; i < n; i += 2) {
      cf *restrict p = a + i;
      float vr = p[1].re * w0r - p[1].im * w0i;
      float vi = p[1].re * w0i + p[1].im * w0r;
      float ur = p[0].re, ui = p[0].im;
      p[0].re = ur + vr;
      p[0].im = ui + vi;
      p[1].re = ur - vr;
      p[1].im = ui - vi;
    }
  }
  if (n >= 4) {
    const float w0r = tw[2].re, w0i = s * tw[2].im;
    const float w1r = tw[3].re, w1i = s * tw[3].im;
    for (int i = 0; i < n; i += 4) {
      cf *restrict p = a + i;
      float v0r = p[2].re * w0r - p[2].im * w0i;
      float v0i = p[2].re * w0i + p[2].im * w0r;
      float u0r = p[0].re, u0i = p[0].im;
      p[0].re = u0r + v0r;
      p[0].im = u0i + v0i;
      p[2].re = u0r - v0r;
      p[2].im = u0i - v0i;
      float v1r = p[3].re * w1r - p[3].im * w1i;
      float v1i = p[3].re * w1i + p[3].im * w1r;
      float u1r = p[1].re, u1i = p[1].im;
      p[1].re = u1r + v1r;
      p[1].im = u1i + v1i;
      p[3].re = u1r - v1r;
      p[3].im = u1i - v1i;
    }
  }
  if (n >= 8) {
    const cf *restrict w = tw + 4;
    for (int i = 0; i < n; i += 8) {
      cf *restrict p = a + i;
      cf *restrict q = p + 4;
      for (int j = 0; j < 4; j++) {
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
#ifdef CVM_X86_KERNEL
  if (__builtin_cpu_supports("avx2")) {
    fft_stages_avx2(a, n, tw, inverse);
    return;
  }
#endif
  for (int half = 8; half < n; half <<= 1) {
    const cf *restrict w = tw + half;
    for (int i = 0; i < n; i += half << 1) {
      cf *restrict p = a + i;
      cf *restrict q = p + half;
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

#ifdef CVM_X86_KERNEL
/* One column-FFT butterfly row pair with a broadcast twiddle: identical
 * per-element ops to the scalar loop (wi arrives pre-multiplied by the
 * exact sign flip). */
__attribute__((target("avx2"))) static void fft_cols_bfly_avx2(
    cf *restrict p, cf *restrict q, float wr, float wi, int width) {
  const __m256 vwr = _mm256_set1_ps(wr), vwi = _mm256_set1_ps(wi);
  int c = 0;
  for (; c + 4 <= width; c += 4) {
    __m256 qv = _mm256_loadu_ps((const float *)(q + c));
    __m256 qs = _mm256_permute_ps(qv, 0xB1);
    __m256 v = _mm256_addsub_ps(_mm256_mul_ps(qv, vwr),
                                _mm256_mul_ps(qs, vwi));
    __m256 pv = _mm256_loadu_ps((const float *)(p + c));
    _mm256_storeu_ps((float *)(p + c), _mm256_add_ps(pv, v));
    _mm256_storeu_ps((float *)(q + c), _mm256_sub_ps(pv, v));
  }
  for (; c < width; c++) {
    float vr = q[c].re * wr - q[c].im * wi;
    float vi = q[c].re * wi + q[c].im * wr;
    float ur = p[c].re, ui = p[c].im;
    p[c].re = ur + vr;
    p[c].im = ui + vi;
    q[c].re = ur - vr;
    q[c].im = ui - vi;
  }
}
#endif /* CVM_X86_KERNEL */

/* FFT along columns of a row-major [n x width] complex array. The butterfly
 * inner loop runs across contiguous row elements, so it vectorizes and stays
 * cache friendly. rowtmp must hold width elements. */
CVM_HOT
static void fft_cols(cf *restrict d, int n, int width, const cf *restrict tw,
                     const int *restrict pairs, int inverse,
                     cf *restrict rowtmp) {
  size_t rb = (size_t)width * sizeof(cf);
  int np = pairs[0];
  for (int k = 0; k < np; k++) {
    int i = pairs[1 + 2 * k], j = pairs[2 + 2 * k];
    memcpy(rowtmp, d + (size_t)i * width, rb);
    memcpy(d + (size_t)i * width, d + (size_t)j * width, rb);
    memcpy(d + (size_t)j * width, rowtmp, rb);
  }
  float s = inverse ? -1.0f : 1.0f;
#ifdef CVM_X86_KERNEL
  if (__builtin_cpu_supports("avx2")) {
    for (int half = 1; half < n; half <<= 1) {
      const cf *restrict w = tw + half;
      for (int i = 0; i < n; i += half << 1)
        for (int j = 0; j < half; j++)
          fft_cols_bfly_avx2(d + (size_t)(i + j) * width,
                             d + (size_t)(i + j + half) * width, w[j].re,
                             s * w[j].im, width);
    }
    return;
  }
#endif
  for (int half = 1; half < n; half <<= 1) {
    const cf *restrict w = tw + half;
    for (int i = 0; i < n; i += half << 1) {
      for (int j = 0; j < half; j++) {
        float wr = w[j].re, wi = s * w[j].im;
        cf *restrict p = d + (size_t)(i + j) * width;
        cf *restrict q = d + (size_t)(i + j + half) * width;
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
  int *prW, *prH; /* swap-pair lists */
  int ownW, ownH; /* tables owned by this plan (not from the cache) */
} Plan;

/* Twiddle/swap-pair tables are immutable per FFT size, and real workloads
 * cycle through a handful of power-of-two sizes, so completed tables are
 * published to a small process-lifetime cache instead of being rebuilt
 * every call. Sizes above 2^12 stay per-call to bound the cache. */
#ifndef CVM_NO_THREADS
#define CVM_TAB_CACHE_MAX 4096
typedef struct {
  cf *tw;
  int *pairs;
} CvmFftTab;
static CvmFftTab cvm_tabs[13]; /* index = log2(n) for n = 2..4096 */
static pthread_mutex_t cvm_tab_mu = PTHREAD_MUTEX_INITIALIZER;
#endif

static int plan_tables(int n, cf **tw, int **pairs, int *owned) {
#ifndef CVM_NO_THREADS
  if (n <= CVM_TAB_CACHE_MAX) {
    int idx = 0;
    while ((1 << idx) < n) idx++;
    pthread_mutex_lock(&cvm_tab_mu);
    if (!cvm_tabs[idx].tw) {
      cf *t = make_twiddles(n);
      int *pr = make_swap_pairs(n);
      if (!t || !pr) {
        free(t);
        free(pr);
        pthread_mutex_unlock(&cvm_tab_mu);
        return CVM_ERR_NOMEM;
      }
      cvm_tabs[idx].tw = t;
      cvm_tabs[idx].pairs = pr;
    }
    *tw = cvm_tabs[idx].tw;
    *pairs = cvm_tabs[idx].pairs;
    *owned = 0;
    pthread_mutex_unlock(&cvm_tab_mu);
    return CVM_OK;
  }
#endif
  *tw = make_twiddles(n);
  *pairs = make_swap_pairs(n);
  if (!*tw || !*pairs) {
    free(*tw);
    free(*pairs);
    *tw = NULL;
    *pairs = NULL;
    *owned = 0;
    return CVM_ERR_NOMEM;
  }
  *owned = 1;
  return CVM_OK;
}

static void plan_free(Plan *p) {
  if (p->ownW) {
    free(p->twW);
    free(p->prW);
  }
  if (p->ownH) {
    free(p->twH);
    free(p->prH);
  }
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

  int rc = plan_tables(dftW, &p->twW, &p->prW, &p->ownW);
  if (rc != CVM_OK) return rc;
  if (dftH == dftW) {
    p->twH = p->twW;
    p->prH = p->prW;
    p->ownH = 0;
  } else {
    rc = plan_tables(dftH, &p->twH, &p->prH, &p->ownH);
    if (rc != CVM_OK) {
      plan_free(p);
      return rc;
    }
  }
  return CVM_OK;
}

/* Real 2D forward DFT of one channel of a uint8 block into spec[dftH][hw].
 * Two real rows are packed into one complex FFT (re/im) and untangled, so
 * only ceil(loadH/2) row FFTs run; fully-zero padding rows are memset. */
CVM_HOT
static void block_forward(const uint8_t *restrict chan, size_t stride,
                          int step, int x0, int y0, int loadW, int loadH,
                          const Plan *p, cf *restrict spec, cf *restrict z) {
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
    fft(z, n, p->twW, p->prW, 0);
    for (int k = 0; k < hw; k++) {
      cf zk = z[k], zn = z[(n - k) & mask];
      sa[k].re = 0.5f * (zk.re + zn.re);
      sa[k].im = 0.5f * (zk.im - zn.im);
      sb[k].re = 0.5f * (zk.im + zn.im);
      sb[k].im = 0.5f * (zn.re - zk.re);
    }
  }
  fft_cols(spec, p->dftH, hw, p->twH, p->prH, 0, z);
}

/* spec *= conj(tspec), pointwise. */
CVM_HOT
static void mul_conj(cf *restrict spec, const cf *restrict tspec,
                     size_t count) {
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
static void block_inverse_emit(const Plan *p, cf *restrict spec,
                               cf *restrict z, float *restrict res, int rw,
                               int x0, int y0, int bw, int bh, int add) {
  fft_cols(spec, p->dftH, p->hw, p->twH, p->prH, 1, z);
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
    fft(z, n, p->twW, p->prW, 1);
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

/* Sliding column sums are integer-valued (colSum <= 255*th, colSum2 <=
 * 255^2*th*cn), so accumulating them as integers instead of doubles feeds
 * the double window math below bit-identical inputs — every value is an
 * exact integer either way — while integer SIMD runs several times wider.
 * colSum is int32 (255*th cannot overflow it for any real image), colSum2
 * int64. The (step,cn) pairs the public API produces get specialized loops;
 * anything else takes the generic path. */

CVM_HOT
static void col_build_u1(int32_t *restrict s, int64_t *restrict s2,
                         const uint8_t *restrict img, size_t istride, int iw,
                         int y0, int th) {
  memset(s, 0, (size_t)iw * sizeof(*s));
  memset(s2, 0, (size_t)iw * sizeof(*s2));
  for (int y = y0; y < y0 + th; y++) {
    const uint8_t *restrict row = img + (size_t)y * istride;
    for (int x = 0; x < iw; x++) {
      int v = row[x];
      s[x] += v;
      s2[x] += v * v;
    }
  }
}

CVM_HOT
static void col_slide_u1(int32_t *restrict s, int64_t *restrict s2,
                         const uint8_t *restrict rsub,
                         const uint8_t *restrict radd, int iw) {
  for (int x = 0; x < iw; x++) {
    int a = radd[x], b = rsub[x];
    s[x] += a - b;
    s2[x] += a * a - b * b;
  }
}

/* step==4 variants keep colSum at stride 4; with cn==3 the alpha lane is
 * accumulated too (the bytes are right there) but never read downstream.
 * The per-channel sums and the square sums run as separate loops so each
 * vectorizes cleanly (mixing int32 and int64 updates in one loop defeats
 * the vectorizer); integer sums are order-free, so splitting changes
 * nothing. */
CVM_HOT
static void col_build_u4_3(int32_t *restrict s, int64_t *restrict s2,
                           const uint8_t *restrict img, size_t istride,
                           int iw, int y0, int th) {
  memset(s, 0, (size_t)iw * 4 * sizeof(*s));
  memset(s2, 0, (size_t)iw * sizeof(*s2));
  for (int y = y0; y < y0 + th; y++) {
    const uint8_t *restrict row = img + (size_t)y * istride;
    int n = iw * 4;
    for (int i = 0; i < n; i++) s[i] += row[i];
    for (int x = 0; x < iw; x++) {
      int r = row[x * 4], g = row[x * 4 + 1], b = row[x * 4 + 2];
      s2[x] += r * r + g * g + b * b;
    }
  }
}

CVM_HOT
static void col_slide_u4_3(int32_t *restrict s, int64_t *restrict s2,
                           const uint8_t *restrict rsub,
                           const uint8_t *restrict radd, int iw) {
  int n = iw * 4;
  for (int i = 0; i < n; i++) s[i] += radd[i] - rsub[i];
  for (int x = 0; x < iw; x++) {
    int ar = radd[x * 4], ag = radd[x * 4 + 1], ab = radd[x * 4 + 2];
    int br = rsub[x * 4], bg = rsub[x * 4 + 1], bb = rsub[x * 4 + 2];
    s2[x] += ar * ar + ag * ag + ab * ab - br * br - bg * bg - bb * bb;
  }
}

CVM_HOT
static void col_build_u4_4(int32_t *restrict s, int64_t *restrict s2,
                           const uint8_t *restrict img, size_t istride,
                           int iw, int y0, int th) {
  memset(s, 0, (size_t)iw * 4 * sizeof(*s));
  memset(s2, 0, (size_t)iw * sizeof(*s2));
  for (int y = y0; y < y0 + th; y++) {
    const uint8_t *restrict row = img + (size_t)y * istride;
    int n = iw * 4;
    for (int i = 0; i < n; i++) s[i] += row[i];
    for (int x = 0; x < iw; x++) {
      int r = row[x * 4], g = row[x * 4 + 1];
      int b = row[x * 4 + 2], a = row[x * 4 + 3];
      s2[x] += r * r + g * g + b * b + a * a;
    }
  }
}

CVM_HOT
static void col_slide_u4_4(int32_t *restrict s, int64_t *restrict s2,
                           const uint8_t *restrict rsub,
                           const uint8_t *restrict radd, int iw) {
  int n = iw * 4;
  for (int i = 0; i < n; i++) s[i] += radd[i] - rsub[i];
  for (int x = 0; x < iw; x++) {
    int ar = radd[x * 4], ag = radd[x * 4 + 1];
    int ab = radd[x * 4 + 2], aa = radd[x * 4 + 3];
    int br = rsub[x * 4], bg = rsub[x * 4 + 1];
    int bb = rsub[x * 4 + 2], ba = rsub[x * 4 + 3];
    s2[x] += ar * ar + ag * ag + ab * ab + aa * aa - br * br - bg * bg -
             bb * bb - ba * ba;
  }
}

CVM_HOT
static void col_build_gen(int32_t *restrict s, int64_t *restrict s2,
                          const uint8_t *restrict img, size_t istride, int iw,
                          int cn, int step, int y0, int th) {
  memset(s, 0, (size_t)iw * cn * sizeof(*s));
  memset(s2, 0, (size_t)iw * sizeof(*s2));
  for (int y = y0; y < y0 + th; y++) {
    const uint8_t *restrict row = img + (size_t)y * istride;
    for (int x = 0; x < iw; x++)
      for (int k = 0; k < cn; k++) {
        int v = row[(size_t)x * step + k];
        s[(size_t)x * cn + k] += v;
        s2[x] += v * v;
      }
  }
}

CVM_HOT
static void col_slide_gen(int32_t *restrict s, int64_t *restrict s2,
                          const uint8_t *restrict rsub,
                          const uint8_t *restrict radd, int iw, int cn,
                          int step) {
  for (int x = 0; x < iw; x++)
    for (int k = 0; k < cn; k++) {
      int a = radd[(size_t)x * step + k], b = rsub[(size_t)x * step + k];
      s[(size_t)x * cn + k] += a - b;
      s2[x] += a * a - b * b;
    }
}

/* Tail of the normalized-coefficient formula, written as straight-line
 * selects (no branches) so the surrounding loops if-convert and vectorize:
 * sqrt and the division run unconditionally — their inputs are clamped
 * non-negative resp. their results discarded by the select when the guard
 * fails — which leaves every selected value bit-identical to the branchy
 * reference formulation. */
static inline float norm_tail(double num, double wndMean2, double s2d,
                              double eps, double templNorm) {
  double diff2 = s2d - wndMean2;
  diff2 = diff2 < 0 ? 0 : diff2;
  double e = eps * s2d;
  double lim = 0.5 < e ? 0.5 : e;
  double sq = sqrt(diff2) * templNorm;
  double den = diff2 <= lim ? 0.0 : sq;
  double an = fabs(num);
  double dv = num / den;
  double sat = num > 0 ? 1.0 : -1.0;
  return (float)(an < den ? dv : (an < den * 1.125 ? sat : 0.0));
}

#ifdef CVM_X86_KERNEL
/* AVX2 norm_tail over one result row: four lanes each execute the scalar
 * op sequence (same order, correctly rounded vector ops), so stored values
 * are bit-identical to the scalar loop. cn is a compile-time constant in
 * each instantiation. Lanes whose select-guard fails may compute inf/NaN in
 * the discarded division exactly like the scalar reference. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((target("avx2"), always_inline))
#else
__attribute__((target("avx2")))
#endif
static inline void norm_row_avx2_cn(float *restrict rrow,
                                    const float *restrict crow,
                                    const double *restrict wt, size_t rwsz,
                                    int rw, int cn,
                                    const double *restrict mean,
                                    double invArea, double eps,
                                    double templNorm) {
  const __m256d vinv = _mm256_set1_pd(invArea);
  const __m256d veps = _mm256_set1_pd(eps);
  const __m256d vtn = _mm256_set1_pd(templNorm);
  const __m256d vzero = _mm256_setzero_pd();
  const __m256d vhalf = _mm256_set1_pd(0.5);
  const __m256d vone = _mm256_set1_pd(1.0);
  const __m256d vneg1 = _mm256_set1_pd(-1.0);
  const __m256d vsat125 = _mm256_set1_pd(1.125);
  const __m256d vabs =
      _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
  const __m256d vm0 = _mm256_set1_pd(mean[0]);
  const __m256d vm1 = _mm256_set1_pd(cn > 1 ? mean[1] : 0.0);
  const __m256d vm2 = _mm256_set1_pd(cn > 2 ? mean[2] : 0.0);
  const __m256d vm3 = _mm256_set1_pd(cn > 3 ? mean[3] : 0.0);
  const double *restrict t0 = wt;
  const double *restrict t1 = wt + rwsz;
  const double *restrict t2 = wt + 2 * rwsz;
  const double *restrict t3 = wt + 3 * rwsz;
  const double *restrict q2 = wt + (size_t)cn * rwsz;
  int x = 0;
  for (; x + 4 <= rw; x += 4) {
    __m256d num = _mm256_cvtps_pd(_mm_loadu_ps(crow + x));
    __m256d v0 = _mm256_loadu_pd(t0 + x);
    __m256d wm = _mm256_mul_pd(v0, v0);
    num = _mm256_sub_pd(num, _mm256_mul_pd(v0, vm0));
    if (cn > 1) {
      __m256d v1 = _mm256_loadu_pd(t1 + x);
      wm = _mm256_add_pd(wm, _mm256_mul_pd(v1, v1));
      num = _mm256_sub_pd(num, _mm256_mul_pd(v1, vm1));
    }
    if (cn > 2) {
      __m256d v2 = _mm256_loadu_pd(t2 + x);
      wm = _mm256_add_pd(wm, _mm256_mul_pd(v2, v2));
      num = _mm256_sub_pd(num, _mm256_mul_pd(v2, vm2));
    }
    if (cn > 3) {
      __m256d v3 = _mm256_loadu_pd(t3 + x);
      wm = _mm256_add_pd(wm, _mm256_mul_pd(v3, v3));
      num = _mm256_sub_pd(num, _mm256_mul_pd(v3, vm3));
    }
    wm = _mm256_mul_pd(wm, vinv);
    __m256d s2d = _mm256_loadu_pd(q2 + x);
    /* diff2 is never -0/NaN, so max() equals the scalar `<0 ? 0` clamp */
    __m256d diff2 = _mm256_max_pd(_mm256_sub_pd(s2d, wm), vzero);
    __m256d e = _mm256_mul_pd(veps, s2d);
    __m256d lim = _mm256_min_pd(vhalf, e);
    __m256d sq = _mm256_mul_pd(_mm256_sqrt_pd(diff2), vtn);
    __m256d leq = _mm256_cmp_pd(diff2, lim, _CMP_LE_OQ);
    __m256d den = _mm256_andnot_pd(leq, sq); /* le ? 0.0 : sq */
    __m256d an = _mm256_and_pd(num, vabs);
    __m256d m1 = _mm256_cmp_pd(an, den, _CMP_LT_OQ);
    /* Discarded lanes divide 0/1 so no special-value penalties arise; the
     * selected lanes' num/den is untouched. */
    __m256d dv = _mm256_div_pd(_mm256_and_pd(m1, num),
                               _mm256_blendv_pd(vone, den, m1));
    __m256d sat =
        _mm256_blendv_pd(vneg1, vone, _mm256_cmp_pd(num, vzero, _CMP_GT_OQ));
    __m256d m2 =
        _mm256_cmp_pd(an, _mm256_mul_pd(den, vsat125), _CMP_LT_OQ);
    __m256d inner = _mm256_and_pd(m2, sat); /* m2 ? sat : 0.0 */
    __m256d v = _mm256_blendv_pd(inner, dv, m1);
    _mm_storeu_ps(rrow + x, _mm256_cvtpd_ps(v));
  }
  for (; x < rw; x++) {
    double num = (double)crow[x];
    double wm = 0;
    num -= t0[x] * mean[0];
    wm += t0[x] * t0[x];
    if (cn > 1) {
      num -= t1[x] * mean[1];
      wm += t1[x] * t1[x];
    }
    if (cn > 2) {
      num -= t2[x] * mean[2];
      wm += t2[x] * t2[x];
    }
    if (cn > 3) {
      num -= t3[x] * mean[3];
      wm += t3[x] * t3[x];
    }
    rrow[x] = norm_tail(num, wm * invArea, q2[x], eps, templNorm);
  }
}

__attribute__((target("avx2"))) static void norm_row_avx2(
    float *restrict rrow, const float *restrict crow,
    const double *restrict wt, size_t rwsz, int rw, int cn,
    const double *restrict mean, double invArea, double eps,
    double templNorm) {
  if (cn == 3)
    norm_row_avx2_cn(rrow, crow, wt, rwsz, rw, 3, mean, invArea, eps,
                     templNorm);
  else if (cn == 1)
    norm_row_avx2_cn(rrow, crow, wt, rwsz, rw, 1, mean, invArea, eps,
                     templNorm);
  else if (cn == 4)
    norm_row_avx2_cn(rrow, crow, wt, rwsz, rw, 4, mean, invArea, eps,
                     templNorm);
  else
    norm_row_avx2_cn(rrow, crow, wt, rwsz, rw, 2, mean, invArea, eps,
                     templNorm);
}
#endif /* CVM_X86_KERNEL */

/* OpenCV common_matchTemplate() for TM_CCOEFF_NORMED over rows [y0, y1),
 * with O(iw) sliding column sums instead of integral images, fused with the
 * minMaxLoc scan (strict comparisons keep OpenCV's first-occurrence tie
 * semantics). Because all window statistics are exact integers, a band
 * starting at any y0 computes bit-identical values. */
CVM_HOT
static void normalize_band(const uint8_t *restrict img, size_t istride,
                           int iw, int cn, int step, int tw, int th, int rw,
                           int y0, int y1, const double *restrict mean,
                           double templNorm, const float *restrict corr,
                           float *restrict result, int32_t *restrict colSum,
                           int64_t *restrict colSum2, double *restrict wt,
                           CvmExtrema *out) {
  double invArea = 1.0 / ((double)tw * th);
  int u4 = step == 4 && (cn == 3 || cn == 4);
  int cs = u4 ? 4 : cn; /* colSum lane stride */
  if (step == 1 && cn == 1)
    col_build_u1(colSum, colSum2, img, istride, iw, y0, th);
  else if (u4 && cn == 3)
    col_build_u4_3(colSum, colSum2, img, istride, iw, y0, th);
  else if (u4)
    col_build_u4_4(colSum, colSum2, img, istride, iw, y0, th);
  else
    col_build_gen(colSum, colSum2, img, istride, iw, cn, step, y0, th);

  float minv = FLT_MAX, maxv = -FLT_MAX;
  int minx = 0, miny = y0, maxx = 0, maxy = y0;
  double eps = 10.0 * FLT_EPSILON;
#ifdef CVM_X86_KERNEL
  int use_avx2 = __builtin_cpu_supports("avx2");
#endif

  for (int y = y0;; y++) {
    /* Pass 1 (scalar, exact): slide the integer window sums across the row
     * and spill them as doubles (every value is an exact integer, so the
     * conversion is lossless and the spilled values are bit-identical to
     * what the fused loop computed). wt holds cn lanes of window sums plus
     * one lane of window square sums, each rw wide. */
    float *restrict rrow = result + (size_t)y * rw;
    const float *restrict crow = corr + (size_t)y * rw;
    const double *restrict q2 = wt + (size_t)cn * CVM_NORM_CHUNK;
    {
      int64_t s[4] = {0, 0, 0, 0}, s2 = 0;
      for (int x = 0; x < tw; x++) {
        for (int k = 0; k < cn; k++) s[k] += colSum[(size_t)x * cs + k];
        s2 += colSum2[x];
      }
      for (int x0 = 0; x0 < rw; x0 += CVM_NORM_CHUNK) {
        int len = rw - x0 < CVM_NORM_CHUNK ? rw - x0 : CVM_NORM_CHUNK;
        /* Pass 1 (scalar, exact): slide the integer window sums across the
         * chunk and spill them as doubles (every value is an exact integer,
         * so the conversion is lossless and the spilled values are
         * bit-identical to what the fused loop computed). */
        for (int i = 0;; i++) {
          for (int k = 0; k < cn; k++)
            wt[(size_t)k * CVM_NORM_CHUNK + i] = (double)s[k];
          wt[(size_t)cn * CVM_NORM_CHUNK + i] = (double)s2;
          int x = x0 + i;
          if (i + 1 >= len) {
            if (x + 1 < rw) { /* carry the slide into the next chunk */
              for (int k = 0; k < cn; k++)
                s[k] += colSum[(size_t)(x + tw) * cs + k] -
                        colSum[(size_t)x * cs + k];
              s2 += colSum2[x + tw] - colSum2[x];
            }
            break;
          }
          for (int k = 0; k < cn; k++)
            s[k] += colSum[(size_t)(x + tw) * cs + k] -
                    colSum[(size_t)x * cs + k];
          s2 += colSum2[x + tw] - colSum2[x];
        }
        /* Pass 2 (vector, element-wise): the normalized value. Same
         * expression tree per element as the reference formulation. */
        float *restrict rr = rrow + x0;
        const float *restrict cr = crow + x0;
#ifdef CVM_X86_KERNEL
        if (use_avx2) {
          norm_row_avx2(rr, cr, wt, (size_t)CVM_NORM_CHUNK, len, cn, mean,
                        invArea, eps, templNorm);
          continue;
        }
#endif
        if (cn == 3) {
          const double m0 = mean[0], m1 = mean[1], m2 = mean[2];
          const double *restrict t0 = wt;
          const double *restrict t1 = wt + CVM_NORM_CHUNK;
          const double *restrict t2 = wt + 2 * (size_t)CVM_NORM_CHUNK;
          for (int i = 0; i < len; i++) {
            double num =
                ((double)cr[i] - t0[i] * m0 - t1[i] * m1) - t2[i] * m2;
            double wndMean2 =
                ((t0[i] * t0[i] + t1[i] * t1[i]) + t2[i] * t2[i]) * invArea;
            rr[i] = norm_tail(num, wndMean2, q2[i], eps, templNorm);
          }
        } else if (cn == 1) {
          const double m0 = mean[0];
          const double *restrict t0 = wt;
          for (int i = 0; i < len; i++) {
            double num = (double)cr[i] - t0[i] * m0;
            double wndMean2 = (t0[i] * t0[i]) * invArea;
            rr[i] = norm_tail(num, wndMean2, q2[i], eps, templNorm);
          }
        } else if (cn == 4) {
          const double m0 = mean[0], m1 = mean[1], m2 = mean[2],
                       m3 = mean[3];
          const double *restrict t0 = wt;
          const double *restrict t1 = wt + CVM_NORM_CHUNK;
          const double *restrict t2 = wt + 2 * (size_t)CVM_NORM_CHUNK;
          const double *restrict t3 = wt + 3 * (size_t)CVM_NORM_CHUNK;
          for (int i = 0; i < len; i++) {
            double num = (((double)cr[i] - t0[i] * m0 - t1[i] * m1) -
                          t2[i] * m2) -
                         t3[i] * m3;
            double wndMean2 =
                (((t0[i] * t0[i] + t1[i] * t1[i]) + t2[i] * t2[i]) +
                 t3[i] * t3[i]) *
                invArea;
            rr[i] = norm_tail(num, wndMean2, q2[i], eps, templNorm);
          }
        } else {
          for (int i = 0; i < len; i++) {
            double num = (double)cr[i];
            double wndMean2 = 0;
            for (int k = 0; k < cn; k++) {
              double t = wt[(size_t)k * CVM_NORM_CHUNK + i];
              wndMean2 += t * t;
              num -= t * mean[k];
            }
            wndMean2 *= invArea;
            rr[i] = norm_tail(num, wndMean2, q2[i], eps, templNorm);
          }
        }
      }
    }
    /* Pass 3: min/max scan of the stored float32 row. Ascending x with
     * strict compares reproduces the fused loop's first-occurrence tie
     * semantics exactly. */
    for (int x = 0; x < rw; x++) {
      float v = rrow[x];
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
    }
    if (y + 1 >= y1) break;
    const uint8_t *restrict rsub = img + (size_t)y * istride;
    const uint8_t *restrict radd = img + (size_t)(y + th) * istride;
    if (step == 1 && cn == 1)
      col_slide_u1(colSum, colSum2, rsub, radd, iw);
    else if (u4 && cn == 3)
      col_slide_u4_3(colSum, colSum2, rsub, radd, iw);
    else if (u4)
      col_slide_u4_4(colSum, colSum2, rsub, radd, iw);
    else
      col_slide_gen(colSum, colSum2, rsub, radd, iw, cn, step);
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
  const float *corr;
  float *result;
  char *colSums;    /* nb bands * bandBytes:
                       [iw int64][(cn+1)*CHUNK double][iw*cs int32] */
  size_t bandBytes; /* per-band scratch, 8-byte aligned */
  int *bandY;       /* nb+1 boundaries */
  CvmExtrema *ext;  /* nb entries */
  int nb;
} NormCtx;

static void norm_worker(void *vc, int w) {
  NormCtx *c = (NormCtx *)vc;
  if (w >= c->nb) return; /* one band per worker */
  int64_t *cs2 = (int64_t *)(c->colSums + (size_t)w * c->bandBytes);
  double *wt = (double *)(cs2 + c->iw);
  int32_t *cs = (int32_t *)(wt + (size_t)(c->cn + 1) * CVM_NORM_CHUNK);
  normalize_band(c->img, c->istride, c->iw, c->cn, c->step, c->tw, c->th,
                 c->rw, c->bandY[w], c->bandY[w + 1], c->mean, c->templNorm,
                 c->corr, c->result, cs, cs2, wt, &c->ext[w]);
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
                              int nw, const float *corr, float *result,
                              CvmExtrema *out) {
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

  int cs = (step == 4 && (cn == 3 || cn == 4)) ? 4 : cn;
  size_t bandBytes = (size_t)iw * sizeof(int64_t) +
                     (size_t)(cn + 1) * CVM_NORM_CHUNK * sizeof(double) +
                     (((size_t)iw * cs + 1) & ~(size_t)1) * sizeof(int32_t);
  char *colSums = (char *)malloc((size_t)nb * bandBytes);
  if (!colSums) return CVM_ERR_NOMEM;
  int bandY[CVM_MAX_THREADS + 1];
  CvmExtrema ext[CVM_MAX_THREADS];
  for (int b = 0; b <= nb; b++) bandY[b] = (int)((long long)rh * b / nb);

  NormCtx nc = {img,   istride,   iw,   cn,     step,    tw,        th,
                rw,    mean,      templNorm,    corr,    result,    colSums,
                bandBytes, bandY, ext,  nb};
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
                          res, res, out);

done_res:
  if (!result) free(res);
  return rc;
}
