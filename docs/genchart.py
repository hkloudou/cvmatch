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


def rows_for(panel_key):
    """(label, native, go4, cgo4, gray4) rows for one chart panel."""
    rows = []
    for key, label, panel in LAYOUT:
        if panel != panel_key or key not in SC:
            continue
        s = SC[key]
        if not all(k in s for k in ("native", "go4", "cgo4", "gray4")):
            continue
        rows.append((label, s["native"], s["go4"], s["cgo4"], s["gray4"]))
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


def panel(out, t, rows, y, vmax, ticks, title):
    out.append(f'<text x="{LEFT}" y="{y}" font-size="12" font-weight="600" '
               f'fill="{t["sec"]}" {FONT}>{title}</text>')
    out.append(f'<text x="{W - 4}" y="{y}" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>ms — lower is better · ×  = speedup vs native C++ (&lt;1 = slower)</text>')
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
            lab = f'{v:,.0f} ms'
            if i >= 1:
                lab += f' · {vals[0] / v:.1f}×'
            out.append(f'<text x="{LEFT + wpx + 7:.1f}" y="{yy + BAR - 3}" font-size="11" '
                       f'fill="{t["sec"]}" {FONT} style="font-variant-numeric:tabular-nums">{lab}</text>')
            yy += PITCH
        yy += GROUP_GAP
    return y + h + 34


def legend(out, t, y):
    x = 20
    for i, name in enumerate(SERIES):
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
        "`Match`, milliseconds, measured on " + (HOST or "the CI runner") +
        " (best of `-benchtime 5x`; native C++ is single-threaded because that",
        "is how OpenCV's `matchTemplate` runs; the Go columns come from `-cpu 1,4`):",
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


for mode in ("light", "dark"):
    open(os.path.join(HERE, f"bench-{mode}.svg"), "w").write(speed_chart(mode))
    open(os.path.join(HERE, f"mem-{mode}.svg"), "w").write(mem_chart(mode))
changed = rewrite_readme()
print("charts written; README table " + ("updated" if changed else "unchanged"))
