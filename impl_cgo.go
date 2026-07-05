//go:build cgo

package cvmatch

/*
#cgo CFLAGS: -O3 -std=c99
#cgo LDFLAGS: -lm -lpthread
#include "cvmatch.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Impl reports which core is compiled in: "cgo" (C fast path) here.
const Impl = "cgo"

func matchU8(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step, threads int, result []float32) (float32, int, int, float32, int, int) {
	if tw > iw || th > ih {
		panic(fmt.Sprintf("cvmatch: template %dx%d larger than image %dx%d", tw, th, iw, ih))
	}
	var resPtr *C.float
	if result != nil {
		resPtr = (*C.float)(unsafe.Pointer(&result[0]))
	}
	var out C.CvmExtrema
	rc := C.cvm_match_ccoeff_normed_u8(
		(*C.uint8_t)(unsafe.Pointer(&img[0])), C.int(istride), C.int(iw), C.int(ih),
		(*C.uint8_t)(unsafe.Pointer(&tpl[0])), C.int(tstride), C.int(tw), C.int(th),
		C.int(cn), C.int(step), C.int(threads), resPtr, &out)
	if rc != C.CVM_OK {
		panic(fmt.Sprintf("cvmatch: match failed (code %d)", int(rc)))
	}
	return float32(out.min_val), int(out.min_x), int(out.min_y),
		float32(out.max_val), int(out.max_x), int(out.max_y)
}

func matchExactU8(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step, threads int, result []float32) (float32, int, int, float32, int, int) {
	if tw > iw || th > ih {
		panic(fmt.Sprintf("cvmatch: template %dx%d larger than image %dx%d", tw, th, iw, ih))
	}
	var resPtr *C.float
	if result != nil {
		resPtr = (*C.float)(unsafe.Pointer(&result[0]))
	}
	var out C.CvmExtrema
	rc := C.cvm_match_exact_u8(
		(*C.uint8_t)(unsafe.Pointer(&img[0])), C.int(istride), C.int(iw), C.int(ih),
		(*C.uint8_t)(unsafe.Pointer(&tpl[0])), C.int(tstride), C.int(tw), C.int(th),
		C.int(cn), C.int(step), C.int(threads), resPtr, &out)
	if rc != C.CVM_OK {
		panic(fmt.Sprintf("cvmatch: exact match failed (code %d)", int(rc)))
	}
	return float32(out.min_val), int(out.min_x), int(out.min_y),
		float32(out.max_val), int(out.max_x), int(out.max_y)
}
