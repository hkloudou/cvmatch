#!/usr/bin/env python3
"""Generates the README benchmark SVGs (light + dark variants).

Data: `go test -bench . -benchtime 5x -cpu 1,4` (both cores),
`bench/cpp/native_bench` (native OpenCV C++, end-to-end best-of-7), and the
peak-RSS probes — all measured in one session on a 4-core Intel Xeon
@ 2.10 GHz (linux/amd64). Go values are the 4-thread (default) runs; the
single-thread matrix is in the README table. Update the tables below when
re-measuring, then run: python3 docs/genchart.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, native C++ ms, pure-Go 4T ms, cgo 4T ms, cgo MatchGray 4T ms)
PANEL_HD = [
    ("Window 1600×1000 · button 96×32", 259.8, 141.3, 33.9, 20.3),
    ("Window 1600×1000 · icon 24×24", 236.7, 106.7, 28.2, 16.7),
    ("Window 1600×1000 · panel 300×200", 260.6, 556.5, 121.0, 42.8),
    ("Noise 1280×720 · sub 96×96", 151.8, 254.9, 51.0, 18.4),
    ("Noise 1920×1080 · sub 128×128", 428.1, 291.6, 66.3, 29.1),
    ("Noise 1920×1080 · sub 32×32", 306.3, 149.3, 40.2, 20.4),
    ("Noise 640×480 · varying alpha, 4ch", 34.6, 83.6, 17.1, 5.4),
]
PANEL_4K = [
    ("Window 3840×2160 · button 96×32", 1274.5, 630.0, 164.6, 89.0),
    ("Noise 3840×2160 · sub 256×256", 1815.0, 1362.4, 282.5, 136.4),
]
PANEL_PHOTO = [
    ("fruits 512×480 · sub 80×80", 62.7, 61.3, 13.2, 5.0),
    ("baboon 512×512 · sub 64×64", 38.3, 62.7, 12.6, 5.4),
    ("building 868×600 · sub 100×100", 94.6, 239.6, 42.1, 15.2),
    ("graf1 800×640 · sub 120×120", 109.3, 244.0, 43.2, 16.5),
    ("starry_night 752×600 · sub 128×128", 82.8, 235.2, 50.7, 15.8),
]
MEM = [  # (label, MB, series color slot)
    ("OpenCV C++ (native)", 165.8, 0),
    ("cvmatch.Match (cgo)", 48.2, 2),
    ("cvmatch.MatchGray (cgo)", 42.0, 3),
    ("idle Go process (baseline)", 12.8, 1),
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
    x = LEFT
    for i, name in enumerate(SERIES):
        out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{t["series"][i]}"/>')
        out.append(f'<text x="{x + 15}" y="{y}" font-size="12" fill="{t["ink"]}" {FONT}>{name}</text>')
        x += 15 + 7 * len(name) + 26
    return y + 20


def speed_chart(mode):
    t = THEMES[mode]
    out = []
    out.append(f'<text x="{LEFT}" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Template matching speed — TM_CCOEFF_NORMED, end-to-end call, identical output</text>')
    out.append(f'<text x="{LEFT}" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               '4-core Xeon 2.10 GHz, one session · Go values use default internal threading (4 workers, bit-identical output) · OpenCV matchTemplate does not use extra cores</text>')
    y = legend(out, t, 60)
    y = panel(out, t, PANEL_HD, y + 8, 600, range(0, 601, 150), "HD desktop + noise scenes")
    y = panel(out, t, PANEL_4K, y + 6, 2000, range(0, 2001, 400), "4K scenes")
    y = panel(out, t, PANEL_PHOTO, y + 6, 250, range(0, 251, 50), "Real photographs (OpenCV samples/data)")
    return svg(out, y)


def mem_chart(mode):
    t = THEMES[mode]
    out = []
    out.append(f'<text x="{LEFT}" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Peak process memory — one 1920×1080 / 128×128 match</text>')
    out.append(f'<text x="{LEFT}" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               'whole-process peak RSS (VmHWM), fresh process per run</text>')
    out.append(f'<text x="{W - 4}" y="38" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>MB — lower is better</text>')
    y, vmax = 56, 200
    h = len(MEM) * PITCH + 8
    for tick in range(0, 201, 50):
        x = LEFT + PLOT_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + h}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" {FONT} style="font-variant-numeric:tabular-nums">{tick}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT}" y2="{y + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    yy = y + 4
    for i, (label, v, slot) in enumerate(MEM):
        color = t["series"][slot]
        wpx = PLOT_W * v / vmax
        out.append(f'<text x="{LEFT - 10}" y="{yy + BAR - 3}" font-size="12" text-anchor="end" '
                   f'fill="{t["sec"]}" {FONT}>{label}</text>')
        out.append(f'<path d="{bar_path(LEFT + 0.5, yy, wpx, BAR)}" fill="{color}"/>')
        lab = f'{v:.1f} MB'
        if 0 < i < 3:
            lab += f' · {MEM[0][1] / v:.1f}× less'
        out.append(f'<text x="{LEFT + wpx + 7:.1f}" y="{yy + BAR - 3}" font-size="11" '
                   f'fill="{t["sec"]}" {FONT} style="font-variant-numeric:tabular-nums">{lab}</text>')
        yy += PITCH
    return svg(out, y + h + 30)


def svg(body, height):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}" role="img">\n' + "\n".join(body) + "\n</svg>\n")


for mode in ("light", "dark"):
    open(os.path.join(HERE, f"bench-{mode}.svg"), "w").write(speed_chart(mode))
    open(os.path.join(HERE, f"mem-{mode}.svg"), "w").write(mem_chart(mode))
print("charts written")
