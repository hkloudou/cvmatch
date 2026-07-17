#!/usr/bin/env python3
"""Parses raw benchmark outputs into docs/benchdata.json for genchart.py.

Inputs (any subset; missing values are carried over from the existing
benchdata.json so partial re-measurements stay coherent per field):

  --asm FILE      `go test -bench . -benchtime 5x -cpu 1,4` output
                  (default build with SIMD kernels; keys asm1/asm4 for
                  Match, agray1/agray4 for MatchGray)
  --go FILE       the same run with -tags purego — scalar safe mode
                  (keys go1/go4, gray1/gray4)
  --asm-arm64 / --go-arm64 FILE
                  the same two runs from an arm64 machine (keys gain an
                  'A' suffix: asmA1, goA4, grayA4, ...)
  --native FILE   `bench/cpp/native_bench cpp/scenes N` output (amd64)
  --native-arm64 FILE  the same run on the arm64 machine (key nativeA)
  --mem FILE      concatenated `memprobe -impl {baseline,cvmatch,gray}` lines
  --native-mem KB peak RSS (VmHWM kB) of native_bench on the 1080p/128 scene
  --host TEXT     one-line description of the machine
  --host-arm64 TEXT  same for the arm64 machine

Usage: python3 docs/collect.py [flags] -o docs/benchdata.json
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Everything genchart.py reads; carried-over keys outside this vocabulary
# (e.g. from retired implementations) are pruned on every run.
SCENE_KEYS = {"native", "nativeA",
              "asm1", "asm4", "go1", "go4", "agray1", "agray4", "gray1", "gray4",
              "asmA1", "asmA4", "goA1", "goA4",
              "agrayA1", "agrayA4", "grayA1", "grayA4"}
MEM_KEYS = {"native", "match", "gray", "baseline"}


def parse_go_bench(path):
    """BenchmarkMatch/scene[-N] <iters> <ns> ns/op -> {(kind, scene, cpus): ms}"""
    out = {}
    rx = re.compile(
        r"^Benchmark(Match|MatchGray)/([\w]+?)(?:-(\d+))?\s+\d+\s+([\d.]+) ns/op")
    for line in open(path):
        m = rx.match(line)
        if not m:
            continue
        kind, scene, cpus, ns = m.group(1), m.group(2), m.group(3) or "1", m.group(4)
        out[(kind, scene, int(cpus))] = float(ns) / 1e6
    return out


def parse_native(path):
    """scene <best> ms <core> ms ... -> {scene: end_to_end_ms}"""
    out = {}
    rx = re.compile(r"^(\w+)\s+([\d.]+) ms\s+([\d.]+) ms")
    for line in open(path):
        m = rx.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_mem(path):
    """impl=X ... peakHWM=N kB -> {impl: MB}"""
    out = {}
    rx = re.compile(r"impl=(\w+).*peakHWM=(\d+) kB")
    for line in open(path):
        m = rx.search(line)
        if m:
            out[m.group(1)] = float(m.group(2)) / 1024.0
    return out


def parse_parity(path):
    """paritystat lines: scene=X worst_abs=E ... -> {scene: worst_abs}"""
    out = {}
    rx = re.compile(r"scene=(\w+) worst_abs=([\d.eE+-]+)")
    for line in open(path):
        m = rx.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm")
    ap.add_argument("--go")
    ap.add_argument("--asm-arm64")
    ap.add_argument("--go-arm64")
    ap.add_argument("--native")
    ap.add_argument("--native-arm64")
    ap.add_argument("--mem")
    ap.add_argument("--native-mem", type=float, help="VmHWM kB")
    ap.add_argument("--parity", help="paritystat output (amd64)")
    ap.add_argument("--parity-arm64")
    ap.add_argument("--opencv", help="native OpenCV version string")
    ap.add_argument("--host")
    ap.add_argument("--host-arm64")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "benchdata.json"))
    args = ap.parse_args()

    data = {"host": "", "scenes": {}, "mem": {}}
    if os.path.exists(args.out):
        data = json.load(open(args.out))

    if args.host:
        data["host"] = args.host
    if args.host_arm64:
        data["hostArm64"] = args.host_arm64
    scenes = data.setdefault("scenes", {})

    def put_go(path, match_key, gray_key, suffix):
        for (kind, scene, cpus), ms in parse_go_bench(path).items():
            key = (match_key if kind == "Match" else gray_key) + suffix + str(cpus)
            scenes.setdefault(scene, {})[key] = round(ms, 1)

    if args.native:
        for scene, ms in parse_native(args.native).items():
            scenes.setdefault(scene, {})["native"] = round(ms, 1)
    if args.native_arm64:
        for scene, ms in parse_native(args.native_arm64).items():
            scenes.setdefault(scene, {})["nativeA"] = round(ms, 1)
    if args.asm:
        put_go(args.asm, "asm", "agray", "")
    if args.go:
        put_go(args.go, "go", "gray", "")
    if args.asm_arm64:
        put_go(args.asm_arm64, "asm", "agray", "A")
    if args.go_arm64:
        put_go(args.go_arm64, "go", "gray", "A")
    if args.mem:
        mem = parse_mem(args.mem)
        for k_src, k_dst in (("baseline", "baseline"), ("cvmatch", "match"), ("gray", "gray")):
            if k_src in mem:
                data.setdefault("mem", {})[k_dst] = round(mem[k_src], 1)
    if args.native_mem:
        data.setdefault("mem", {})["native"] = round(args.native_mem / 1024.0, 1)
    if args.parity:
        for scene, w in parse_parity(args.parity).items():
            data.setdefault("parity", {}).setdefault(scene, {})["amd64"] = w
    if args.parity_arm64:
        for scene, w in parse_parity(args.parity_arm64).items():
            data.setdefault("parity", {}).setdefault(scene, {})["arm64"] = w
    if args.opencv:
        data["opencv"] = args.opencv

    for s in scenes.values():
        for k in list(s):
            if k not in SCENE_KEYS:
                del s[k]
    for k in list(data.get("mem", {})):
        if k not in MEM_KEYS:
            del data["mem"][k]
    for per in data.get("parity", {}).values():
        for k in list(per):
            if k not in ("amd64", "arm64"):
                del per[k]

    json.dump(data, open(args.out, "w"), indent=1, sort_keys=True)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
