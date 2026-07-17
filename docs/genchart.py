#!/usr/bin/env python3
"""Renders the README benchmark SVGs (light + dark) and rewrites the
auto-generated README sections from docs/benchdata.json.

benchdata.json is produced by docs/collect.py from raw benchmark outputs
(`go test -bench . -benchtime 5x -cpu 1,4` for both cores,
`bench/cpp/native_bench`, and the memprobe peak-RSS runs) — all measured in
one session on one machine so the three-way comparison is coherent. The
bench-charts GitHub workflow re-measures on every push to main and commits
the refreshed charts, benchdata.json and README table.

Manual run: python3 docs/collect.py [flags] && python3 docs/genchart.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(HERE, "..", "README.md")

DATA = json.load(open(os.path.join(HERE, "benchdata.json")))
SC = DATA["scenes"]
MEM = DATA.get("mem", {})
HOST = DATA.get("host", "")

# (scene key, chart/table label, panel)
LAYOUT = [
    ("window1600_button96x32", "Window 1600×1000 · button 96×32", "hd"),
    ("window1600_icon24x24", "Window 1600×1000 · icon 24×24", "hd"),
    ("window1600_panel300x200", "Window 1600×1000 · panel 300×200", "hd"),
    ("noise720p_sub96", "Noise 1280×720 · sub 96×96", "hd"),
    ("noise1080p_sub128", "Noise 1920×1080 · sub 128×128", "hd"),
    ("noise1080p_sub32", "Noise 1920×1080 · sub 32×32", "hd"),
    ("noise640_alpha", "Noise 640×480 · varying alpha, 4ch", "hd"),
    ("window4k_button96x32", "Window 3840×2160 · button 96×32", "4k"),
    ("noise4k_sub256", "Noise 3840×2160 · sub 256×256", "4k"),
    ("photo_fruits", "fruits 512×480 · sub 80×80", "photo"),
    ("photo_baboon", "baboon 512×512 · sub 64×64", "photo"),
    ("photo_building", "building 868×600 · sub 100×100", "photo"),
    ("photo_graf1", "graf1 800×640 · sub 120×120", "photo"),
    ("photo_starry_night", "starry_night 752×600 · sub 128×128", "photo"),
]

SERIES = ["OpenCV C++ (native, 1 thread)", "cvmatch pure-Go (CGO_ENABLED=0)",
          "cvmatch.Match (cgo)", "cvmatch.MatchGray (cgo)"]
# amd64 chart: every Go bar is annotated with its speedup vs native (bar 0)
REFS = [None, 0, 0, 0]

# arm64 chart: no native baseline — pure-Go bars annotate vs their cgo twin
ARM_SERIES = ["cvmatch.Match (cgo)", "cvmatch.Match pure-Go (NEON)",
              "cvmatch.MatchGray (cgo)", "cvmatch.MatchGray pure-Go (NEON)"]
ARM_REFS = [None, 0, None, 2]

THEMES = {
    "light": dict(series=["#008300", "#2a78d6", "#1baf7a", "#eda100"], ink="#24292f",
                  sec="#57606a", muted="#6e7781", grid="#d0d7de", axis="#afb8c1"),
    "dark": dict(series=["#008300", "#3987e5", "#199e70", "#c98500"], ink="#e6edf3",
                 sec="#9198a1", muted="#8b949e", grid="#30363d", axis="#484f58"),
}

FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'
LEFT, RIGHT, W = 260, 128, 980
PLOT_W = W - LEFT - RIGHT
BAR, PITCH, GROUP_GAP = 14, 16, 18
NSER = len(SERIES)


def rows_for(panel_key, keys=("native", "go4", "cgo4", "gray4")):
    """(label, *values) rows for one chart panel, in `keys` order."""
    rows = []
    for key, label, panel in LAYOUT:
        if panel != panel_key or key not in SC:
            continue
        s = SC[key]
        if not all(k in s for k in keys):
            continue
        rows.append((label, *(s[k] for k in keys)))
    return rows


def axis_scale(rows):
    """Round the axis max up to a tidy step grid of four divisions."""
    vmax = max(v for _, *vals in rows for v in vals)
    for step in (25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800,
                 1000, 1500, 2000, 4000):
        if vmax <= step * 4:
            return step * 4, [step * i for i in range(5)]
    return vmax, [round(vmax / 4 * i) for i in range(5)]


def bar_path(x, y, w, h, r=4):
    """Bar anchored flat at the baseline, 4px rounding on the data end."""
    w = max(w, r + 1)
    return (f'M{x},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z')


def panel(out, t, rows, y, vmax, ticks, title, refs=REFS, note=None):
    out.append(f'<text x="{LEFT}" y="{y}" font-size="12" font-weight="600" '
               f'fill="{t["sec"]}" {FONT}>{title}</text>')
    note = note or 'ms — lower is better · ×  = speedup vs native C++ (&lt;1 = slower)'
    out.append(f'<text x="{W - 4}" y="{y}" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>{note}</text>')
    y += 10
    h = len(rows) * (NSER * PITCH + GROUP_GAP) - GROUP_GAP + 8
    for tick in ticks:
        x = LEFT + PLOT_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + h}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" {FONT} style="font-variant-numeric:tabular-nums">{tick:g}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT}" y2="{y + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    yy = y + 4
    for label, *vals in rows:
        out.append(f'<text x="{LEFT - 10}" y="{yy + NSER * PITCH / 2 + 4}" font-size="12" '
                   f'text-anchor="end" fill="{t["sec"]}" {FONT}>{label}</text>')
        for i, v in enumerate(vals):
            wpx = PLOT_W * min(v, vmax) / vmax
            out.append(f'<path d="{bar_path(LEFT + 0.5, yy, wpx, BAR)}" fill="{t["series"][i]}"/>')
            lab = f'{v:,.0f} ms' if v >= 100 else f'{v:,.1f} ms'
            if refs[i] is not None:
                lab += f' · {vals[refs[i]] / v:.2f}×'
            out.append(f'<text x="{LEFT + wpx + 7:.1f}" y="{yy + BAR - 3}" font-size="11" '
                       f'fill="{t["sec"]}" {FONT} style="font-variant-numeric:tabular-nums">{lab}</text>')
            yy += PITCH
        yy += GROUP_GAP
    return y + h + 34


def legend(out, t, y, series=SERIES):
    x = 20
    for i, name in enumerate(series):
        out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
        out.append(f'<text x="{x + 15}" y="{y}" font-size="12" fill="{t["ink"]}" {FONT}>{name}</text>')
        x += 15 + 7 * len(name) + 26
    return y + 20


def speed_chart(mode):
    t = THEMES[mode]
    out = []
    out.append(f'<text x="20" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Template matching speed — TM_CCOEFF_NORMED, end-to-end call, identical output</text>')
    out.append(f'<text x="20" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               f'{HOST}, one session · Go rows use default internal threading '
               '· OpenCV matchTemplate does not use extra cores</text>')
    y = legend(out, t, 60)
    for panel_key, title in (("hd", "HD desktop + noise scenes"), ("4k", "4K scenes"),
                             ("photo", "Real photographs (OpenCV samples/data)")):
        rows = rows_for(panel_key)
        if not rows:
            continue
        vmax, ticks = axis_scale(rows)
        y = panel(out, t, rows, y + 8, vmax, ticks, title)
    return svg(out, y)


ARM_KEYS = ("cgoA4", "goA4", "grayA4", "pgrayA4")


def arm_speed_chart(mode):
    """arm64 twin of the speed chart: the two cvmatch cores head to head
    (no native OpenCV baseline is built on the arm runner)."""
    t = THEMES[mode]
    out = []
    out.append(f'<text x="20" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'arm64 — cgo core vs pure-Go NEON core, identical output</text>')
    out.append(f'<text x="20" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               f'{DATA.get("hostArm64", "arm64 CI runner")}, one session · '
               'default internal threading (4 workers)</text>')
    y = legend(out, t, 60, ARM_SERIES)
    note = 'ms — lower is better · ×  = pure-Go speedup vs the cgo core (&lt;1 = slower)'
    for panel_key, title in (("hd", "HD desktop + noise scenes"), ("4k", "4K scenes"),
                             ("photo", "Real photographs (OpenCV samples/data)")):
        rows = rows_for(panel_key, ARM_KEYS)
        if not rows:
            continue
        vmax, ticks = axis_scale(rows)
        y = panel(out, t, rows, y + 8, vmax, ticks, title, ARM_REFS, note)
    return svg(out, y)


def mem_chart(mode):
    t = THEMES[mode]
    rows = [("OpenCV C++ (native)", MEM.get("native"), 0),
            ("cvmatch.Match (cgo)", MEM.get("cgo"), 2),
            ("cvmatch.MatchGray (cgo)", MEM.get("gray"), 3),
            ("idle Go process (baseline)", MEM.get("baseline"), 1)]
    rows = [r for r in rows if r[1]]
    out = []
    out.append(f'<text x="{LEFT}" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Peak process memory — one 1920×1080 / 128×128 match</text>')
    out.append(f'<text x="{LEFT}" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               'whole-process peak RSS (VmHWM), fresh process per run</text>')
    out.append(f'<text x="{W - 4}" y="38" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>MB — lower is better</text>')
    vmax = 50 * max(1, -(-int(max(v for _, v, _ in rows)) // 50))
    y = 56
    h = len(rows) * PITCH + 8
    for tick in range(0, vmax + 1, vmax // 4):
        x = LEFT + PLOT_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + h}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" {FONT} style="font-variant-numeric:tabular-nums">{tick}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT}" y2="{y + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    yy = y + 4
    base = rows[0][1]
    for i, (label, v, slot) in enumerate(rows):
        wpx = PLOT_W * v / vmax
        out.append(f'<text x="{LEFT - 10}" y="{yy + BAR - 3}" font-size="12" text-anchor="end" '
                   f'fill="{t["sec"]}" {FONT}>{label}</text>')
        out.append(f'<path d="{bar_path(LEFT + 0.5, yy, wpx, BAR)}" fill="{t["series"][slot]}"/>')
        lab = f'{v:.1f} MB'
        if 0 < i < len(rows) - 1:
            lab += f' · {base / v:.1f}× less'
        out.append(f'<text x="{LEFT + wpx + 7:.1f}" y="{yy + BAR - 3}" font-size="11" '
                   f'fill="{t["sec"]}" {FONT} style="font-variant-numeric:tabular-nums">{lab}</text>')
        yy += PITCH
    return svg(out, y + h + 30)


def svg(body, height):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img">\n' + "\n".join(body) + "\n</svg>\n")


def matrix_markdown():
    """The README 'full matrix' table between the benchmatrix markers."""
    lines = [
        "`Match`, milliseconds, measured on " + (HOST or "the CI runner") + ".",
        "Native C++ is the best of 7 end-to-end runs, single-threaded because",
        "that is how OpenCV's `matchTemplate` runs; the Go columns are",
        "`go test -benchtime 5x` averages from `-cpu 1,4`:",
        "",
        "| scene | native C++ | cgo 1T | cgo 4T | pure-Go 1T | pure-Go 4T |",
        "|---|---|---|---|---|---|",
    ]
    for key, label, _ in LAYOUT:
        s = SC.get(key)
        if not s or not all(k in s for k in ("native", "cgo1", "cgo4", "go1", "go4")):
            continue
        cells = [s["native"], s["cgo1"], s["cgo4"], s["go1"], s["go4"]]
        best = min(cells[1:])
        fmt = [f"{cells[0]:.1f}"] + [
            f"**{v:.1f}**" if v == best else f"{v:.1f}" for v in cells[1:]]
        lines.append(f"| {label.replace('·', '·')} | " + " | ".join(fmt) + " |")
    g = SC.get("noise1080p_sub128", {})
    if all(k in g for k in ("gray1", "gray4", "pgray1", "pgray4")):
        lines += [
            "",
            f"`MatchGray` at 1080p/128 for scale: cgo {g['gray1']:.1f} / {g['gray4']:.1f} ms, "
            f"pure-Go {g['pgray1']:.1f} / {g['pgray4']:.1f} ms (1T / 4T). Native C++ has no gray",
            "row in this suite — a fair baseline would need cvtColor + 1-channel",
            "matchTemplate timed end-to-end, which was not measured; treat MatchGray",
            "numbers as cvmatch-internal.",
        ]
    if any("cgoA1" in s for s in SC.values()):
        lines += [
            "",
            "**arm64** — the same `Match` matrix from the arm64 CI leg (" +
            (DATA.get("hostArm64") or "ubuntu-24.04-arm") + "). No native",
            "OpenCV baseline is built there, so the columns compare the two",
            "cvmatch cores — NEON pure-Go vs the C core:",
            "",
            "| scene | cgo 1T | cgo 4T | pure-Go 1T | pure-Go 4T |",
            "|---|---|---|---|---|",
        ]
        for key, label, _ in LAYOUT:
            s = SC.get(key)
            if not s or not all(k in s for k in ("cgoA1", "cgoA4", "goA1", "goA4")):
                continue
            cells = [s["cgoA1"], s["cgoA4"], s["goA1"], s["goA4"]]
            best = min(cells)
            fmt = [f"**{v:.1f}**" if v == best else f"{v:.1f}" for v in cells]
            lines.append(f"| {label} | " + " | ".join(fmt) + " |")
        ga = SC.get("noise1080p_sub128", {})
        if all(k in ga for k in ("grayA1", "grayA4", "pgrayA1", "pgrayA4")):
            lines += [
                "",
                f"`MatchGray` at 1080p/128 on arm64: cgo {ga['grayA1']:.1f} / {ga['grayA4']:.1f} ms, "
                f"pure-Go {ga['pgrayA1']:.1f} / {ga['pgrayA4']:.1f} ms (1T / 4T).",
            ]
    return "\n".join(lines)


def rewrite_readme():
    text = open(README).read()
    block = ("<!-- benchmatrix:begin — auto-generated by docs/genchart.py, do not edit by hand -->\n"
             + matrix_markdown() + "\n<!-- benchmatrix:end -->")
    new, n = re.subn(r"<!-- benchmatrix:begin[^>]*-->.*?<!-- benchmatrix:end -->",
                     block, text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("README benchmatrix markers not found")
    if new != text:
        open(README, "w").write(new)
        return True
    return False


HAVE_ARM = any(all(k in s for k in ARM_KEYS) for s in SC.values())
for mode in ("light", "dark"):
    open(os.path.join(HERE, f"bench-{mode}.svg"), "w").write(speed_chart(mode))
    open(os.path.join(HERE, f"mem-{mode}.svg"), "w").write(mem_chart(mode))
    if HAVE_ARM:
        open(os.path.join(HERE, f"bench-arm64-{mode}.svg"), "w").write(arm_speed_chart(mode))
changed = rewrite_readme()
print("charts written; README table " + ("updated" if changed else "unchanged"))
