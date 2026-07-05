// dumpscenes writes every benchmark scene as raw RGBA files so the native
// C++ benchmark (bench/cpp) consumes byte-identical inputs.
//
// Format per image: "CVMS" magic, int32 LE width, int32 LE height, then
// width*height*4 RGBA bytes. A manifest.tsv lists name, expected match
// position and the file pair.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"

	"github.com/hkloudou/cvmatch/bench"
)

func writeRaw(path string, img *image.RGBA) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	hdr := make([]byte, 12)
	copy(hdr, "CVMS")
	binary.LittleEndian.PutUint32(hdr[4:], uint32(w))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(h))
	if _, err := f.Write(hdr); err != nil {
		return err
	}
	for y := 0; y < h; y++ {
		if _, err := f.Write(img.Pix[y*img.Stride : y*img.Stride+w*4]); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	out := flag.String("out", "cpp/scenes", "output directory")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	mf, err := os.Create(filepath.Join(*out, "manifest.tsv"))
	if err != nil {
		log.Fatal(err)
	}
	defer mf.Close()
	for _, s := range bench.Scenarios() {
		pf, sf := s.Name+".parent.raw", s.Name+".sub.raw"
		if err := writeRaw(filepath.Join(*out, pf), s.Parent); err != nil {
			log.Fatal(err)
		}
		if err := writeRaw(filepath.Join(*out, sf), s.Sub); err != nil {
			log.Fatal(err)
		}
		fmt.Fprintf(mf, "%s\t%d\t%d\t%s\t%s\n", s.Name, s.PX, s.PY, pf, sf)
	}
	fmt.Println("scenes dumped to", *out)
}
