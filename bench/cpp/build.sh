#!/usr/bin/env bash
# Builds the native OpenCV benchmark against prebuilt static OpenCV 4.12.0
# archives (linux/amd64, WITH_IPP=OFF Release build fetched via the Go
# module proxy), using matching upstream headers. Run from bench/cpp/:
#
#   ./build.sh          # downloads headers on first use, then compiles
#   ./native_bench scenes 5
set -euo pipefail
cd "$(dirname "$0")"

CV2_VER="${CV2_VER:-v0.41200.0}" # -> OpenCV 4.12.0
OPENCV_VER="4.12.0"
MODCACHE="$(go env GOMODCACHE)"
LIBDIR="$MODCACHE/github.com/hkloudou/cv2/libs/linux_amd64@$CV2_VER"

if [ ! -f "$LIBDIR/libopencv_core.a" ]; then
  echo "fetching bundled OpenCV static libs ($CV2_VER)..."
  GOMODCACHE="$MODCACHE" go mod download github.com/hkloudou/cv2/libs/linux_amd64@"$CV2_VER" ||
    (cd .. && go mod download all)
fi

HDR="headers/opencv-$OPENCV_VER"
if [ ! -d "$HDR/modules/core/include" ]; then
  echo "fetching OpenCV $OPENCV_VER headers..."
  mkdir -p headers
  curl -fsSL "https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VER.tar.gz" |
    tar -xz -C headers \
      "opencv-$OPENCV_VER/modules/core/include" \
      "opencv-$OPENCV_VER/modules/imgproc/include"
  # The one cmake-generated header a consumer needs.
  cat > "$HDR/opencv_modules.hpp" <<'EOF'
#define HAVE_OPENCV_CORE
#define HAVE_OPENCV_IMGPROC
EOF
  mkdir -p "$HDR/opencv2"
  cp "$HDR/opencv_modules.hpp" "$HDR/opencv2/opencv_modules.hpp"
fi

g++ -O2 -std=c++17 native_bench.cpp -o native_bench \
  -I "$HDR" \
  -I "$HDR/modules/core/include" \
  -I "$HDR/modules/imgproc/include" \
  "$LIBDIR/libopencv_imgproc.a" "$LIBDIR/libopencv_core.a" "$LIBDIR/libzlib.a" \
  -lpthread -ldl -lm
echo "built ./native_bench (linked against $LIBDIR)"
