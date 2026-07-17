#!/usr/bin/env python3
"""Parses raw benchmark outputs into docs/benchdata.json for genchart.py.

Inputs (any subset; missing values are carried over from the existing
benchdata.json so partial re-measurements stay coherent per field):

  --cgo FILE      `go test -bench . -benchtime 5x -cpu 1,4` output (cgo core)
  --purego FILE   same with CGO_ENABLED=0
  --native FILE   `bench/cpp/native_bench cpp/scenes N` output
  --mem FILE      concatenated `memprobe -impl {baseline,cvmatch,gray}` lines
  --native-mem KB peak RSS (VmHWM kB) of native_bench on the 1080p/128 scene
  --host TEXT     one-line description of the machine

Usage: python3 docs/collect.py [flags] -o docs/benchdata.json
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgo")
    ap.add_argument("--purego")
    ap.add_argument("--native")
    ap.add_argument("--mem")
    ap.add_argument("--native-mem", type=float, help="VmHWM kB")
    ap.add_argument("--host")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "benchdata.json"))
    args = ap.parse_args()

    data = {"host": "", "scenes": {}, "mem": {}}
    if os.path.exists(args.out):
        data = json.load(open(args.out))

    if args.host:
        data["host"] = args.host
    scenes = data.setdefault("scenes", {})

    def put(scene, key, val):
        scenes.setdefault(scene, {})[key] = round(val, 1)

    if args.native:
        for scene, ms in parse_native(args.native).items():
            put(scene, "native", ms)
    if args.cgo:
        for (kind, scene, cpus), ms in parse_go_bench(args.cgo).items():
            put(scene, ("gray" if kind == "MatchGray" else "cgo") + str(cpus), ms)
    if args.purego:
        for (kind, scene, cpus), ms in parse_go_bench(args.purego).items():
            if kind == "Match":  # pure-Go gray tracks the same ratios; keep the chart focused
                put(scene, "go" + str(cpus), ms)
            else:
                put(scene, "pgray" + str(cpus), ms)
    if args.mem:
        mem = parse_mem(args.mem)
        for k_src, k_dst in (("baseline", "baseline"), ("cvmatch", "cgo"), ("gray", "gray")):
            if k_src in mem:
                data.setdefault("mem", {})[k_dst] = round(mem[k_src], 1)
    if args.native_mem:
        data.setdefault("mem", {})["native"] = round(args.native_mem / 1024.0, 1)

    json.dump(data, open(args.out, "w"), indent=1, sort_keys=True)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
