// Native OpenCV C++ benchmark: the same scenes as the Go benchmarks, linked
// against prebuilt static OpenCV 4.12 archives (WITH_IPP=OFF, Release,
// SIMD dispatch on), so the baseline is OpenCV itself with no wrapper.
//
// Timing is end-to-end per call for fairness with the Go side, which starts
// from an in-memory RGBA frame: Mat construction with a private copy of the
// RGBA buffer, matchTemplate into a fresh result Mat, minMaxLoc, and
// destruction. A core-only time (matchTemplate + minMaxLoc on prebuilt
// Mats) is reported as a diagnostic.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct RawImg {
  int w = 0, h = 0;
  std::vector<uint8_t> pix; // RGBA
};

static bool loadRaw(const std::string &path, RawImg &out) {
  std::ifstream f(path, std::ios::binary);
  char magic[4];
  int32_t wh[2];
  if (!f.read(magic, 4) || memcmp(magic, "CVMS", 4) != 0) return false;
  if (!f.read(reinterpret_cast<char *>(wh), 8)) return false;
  out.w = wh[0];
  out.h = wh[1];
  out.pix.resize(size_t(out.w) * out.h * 4);
  return bool(f.read(reinterpret_cast<char *>(out.pix.data()), out.pix.size()));
}

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// dumpResult writes the CV_32F response map so the Go side can verify
// cvmatch.MatchMap element-by-element against native OpenCV output
// ("CVMR" magic, int32 LE width/height, then float32 data).
static void dumpResult(const std::string &path, const cv::Mat &m) {
  std::ofstream f(path, std::ios::binary);
  int32_t wh[2] = {m.cols, m.rows};
  f.write("CVMR", 4);
  f.write(reinterpret_cast<const char *>(wh), 8);
  for (int y = 0; y < m.rows; y++)
    f.write(reinterpret_cast<const char *>(m.ptr<float>(y)), size_t(m.cols) * 4);
}

int main(int argc, char **argv) {
  std::string dir = argc > 1 ? argv[1] : "scenes";
  int iters = argc > 2 ? atoi(argv[2]) : 5;
  bool dump = argc > 3 && std::string(argv[3]) == "dump";
  std::ifstream mf(dir + "/manifest.tsv");
  if (!mf) {
    fprintf(stderr, "cannot open %s/manifest.tsv (run dumpscenes first)\n", dir.c_str());
    return 1;
  }
  printf("threads=%d  iters=%d  (times are best-of runs, ms)\n", cv::getNumThreads(), iters);
  printf("%-28s %12s %12s   %s\n", "scene", "end-to-end", "core-only", "check");

  std::string name, pf, sf;
  int px, py;
  while (mf >> name >> px >> py >> pf >> sf) {
    RawImg pi, si;
    if (!loadRaw(dir + "/" + pf, pi) || !loadRaw(dir + "/" + sf, si)) {
      fprintf(stderr, "%s: failed to load raw images\n", name.c_str());
      return 1;
    }

    double bestFull = 1e30, bestCore = 1e30;
    cv::Point maxLoc;
    double maxVal = 0;

    // End-to-end: from an RGBA byte buffer, exactly the per-call work the
    // wrapper performs minus Go: copy into a Mat, match, scan, release.
    for (int it = 0; it < iters + 1; it++) { // +1 warmup
      double t0 = now_ms();
      cv::Mat parent = cv::Mat(pi.h, pi.w, CV_8UC4, pi.pix.data()).clone();
      cv::Mat sub = cv::Mat(si.h, si.w, CV_8UC4, si.pix.data()).clone();
      cv::Mat result;
      cv::matchTemplate(parent, sub, result, cv::TM_CCOEFF_NORMED);
      double mn;
      cv::Point mnl;
      cv::minMaxLoc(result, &mn, &maxVal, &mnl, &maxLoc);
      double dt = now_ms() - t0;
      if (it > 0 && dt < bestFull) bestFull = dt;
    }

    // Core-only diagnostic: Mats prebuilt, time matchTemplate + minMaxLoc.
    cv::Mat parent = cv::Mat(pi.h, pi.w, CV_8UC4, pi.pix.data()).clone();
    cv::Mat sub = cv::Mat(si.h, si.w, CV_8UC4, si.pix.data()).clone();
    cv::Mat result;
    for (int it = 0; it < iters + 1; it++) {
      double t0 = now_ms();
      cv::matchTemplate(parent, sub, result, cv::TM_CCOEFF_NORMED);
      double mn;
      cv::Point mnl;
      cv::minMaxLoc(result, &mn, &maxVal, &mnl, &maxLoc);
      double dt = now_ms() - t0;
      if (it > 0 && dt < bestCore) bestCore = dt;
    }

    if (dump) dumpResult(dir + "/" + name + ".result.raw", result);

    bool check = px >= 0; /* low-score scenes carry px = -1 */
    bool ok = !check || (maxLoc.x == px && maxLoc.y == py && maxVal > 0.99);
    printf("%-28s %9.1f ms %9.1f ms   %s (%d,%d) max=%.6f\n", name.c_str(),
           bestFull, bestCore, !check ? "-- " : ok ? "OK " : "BAD", maxLoc.x,
           maxLoc.y, maxVal);
    if (!ok) return 1;
  }
  return 0;
}
