#!/usr/bin/env python3
"""Generates the README benchmark SVGs (light + dark variants).

Data comes from `cd bench && go test -bench . -benchtime 5x` and the memprobe
tool on a 4-core Intel Xeon @ 2.10 GHz (linux/amd64). Update the tables below
when re-measuring, then run: python3 docs/genchart.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, cv2 ms, cvmatch ms, gray ms)
PANEL_HD = [
    ("Window 1600×1000 · button 96×32", 248.6, 97.3, 40.8),
    ("Window 1600×1000 · icon 24×24", 217.7, 82.7, 34.6),
    ("Window 1600×1000 · panel 300×200", 240.1, 113.3, 48.0),
    ("Noise 1280×720 · sub 96×96", 130.3, 83.3, 33.8),
    ("Noise 1920×1080 · sub 128×128", 385.8, 153.0, 58.0),
    ("Noise 1920×1080 · sub 32×32", 295.8, 108.9, 46.3),
]
PANEL_4K = [
    ("Window 3840×2160 · button 96×32", 1284.0, 506.4, 206.9),
    ("Noise 3840×2160 · sub 256×256", 1728.2, 721.9, 300.4),
]
MEM = [  # (label, MB)
    ("cv2.Match", 145.5),
    ("cvmatch.Match", 14.8),
    ("cvmatch.MatchGray", 16.8),
]

SERIES = ["cv2.Match", "cvmatch.Match", "cvmatch.MatchGray"]

THEMES = {
    "light": dict(series=["#2a78d6", "#1baf7a", "#eda100"], ink="#24292f",
                  sec="#57606a", muted="#6e7781", grid="#d0d7de", axis="#afb8c1"),
    "dark": dict(series=["#3987e5", "#199e70", "#c98500"], ink="#e6edf3",
                 sec="#9198a1", muted="#8b949e", grid="#30363d", axis="#484f58"),
}

FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'
LEFT, RIGHT, W = 250, 118, 960
PLOT_W = W - LEFT - RIGHT
BAR, PITCH, GROUP_GAP = 14, 16, 18


def bar_path(x, y, w, h, r=4):
    """Bar anchored flat at the baseline, 4px rounding on the data end."""
    w = max(w, r + 1)
    return (f'M{x},{y} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z')


def panel(out, t, rows, y, vmax, ticks, title):
    out.append(f'<text x="{LEFT}" y="{y}" font-size="12" font-weight="600" '
               f'fill="{t["sec"]}" {FONT}>{title}</text>')
    out.append(f'<text x="{W - 4}" y="{y}" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>ms — lower is better</text>')
    y += 10
    h = len(rows) * (3 * PITCH + GROUP_GAP) - GROUP_GAP + 8
    for tick in ticks:
        x = LEFT + PLOT_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + h}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" {FONT} style="font-variant-numeric:tabular-nums">{tick:g}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT}" y2="{y + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    yy = y + 4
    for label, *vals in rows:
        out.append(f'<text x="{LEFT - 10}" y="{yy + 3 * PITCH / 2 + 4}" font-size="12" '
                   f'text-anchor="end" fill="{t["sec"]}" {FONT}>{label}</text>')
        for i, v in enumerate(vals):
            wpx = PLOT_W * v / vmax
            out.append(f'<path d="{bar_path(LEFT + 0.5, yy, wpx, BAR)}" fill="{t["series"][i]}"/>')
            lab = f'{v:,.0f} ms'
            if i > 0:
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
        x += 15 + 9 * len(name) + 26
    return y + 20


def speed_chart(mode):
    t = THEMES[mode]
    out = []
    out.append(f'<text x="{LEFT}" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Template matching speed — TM_CCOEFF_NORMED, one call</text>')
    out.append(f'<text x="{LEFT}" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               '4-core Intel Xeon 2.10 GHz · linux/amd64 · identical output values · speedups vs cv2</text>')
    y = legend(out, t, 60)
    y = panel(out, t, PANEL_HD, y + 8, 400, range(0, 401, 100), "HD scenes")
    y = panel(out, t, PANEL_4K, y + 6, 1800, range(0, 1801, 300), "4K scenes")
    return svg(out, y)


def mem_chart(mode):
    t = THEMES[mode]
    out = []
    out.append(f'<text x="{LEFT}" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Peak native memory — one 1920×1080 / 128×128 match</text>')
    out.append(f'<text x="{LEFT}" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               'process peak-RSS delta (VmHWM), fresh process per run</text>')
    out.append(f'<text x="{W - 4}" y="38" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>MB — lower is better</text>')
    y, vmax = 56, 160
    h = len(MEM) * PITCH + 8
    for tick in range(0, 161, 40):
        x = LEFT + PLOT_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + h}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" {FONT} style="font-variant-numeric:tabular-nums">{tick}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT}" y2="{y + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    yy = y + 4
    for i, (label, v) in enumerate(MEM):
        wpx = PLOT_W * v / vmax
        out.append(f'<text x="{LEFT - 10}" y="{yy + BAR - 3}" font-size="12" text-anchor="end" '
                   f'fill="{t["sec"]}" {FONT}>{label}</text>')
        out.append(f'<path d="{bar_path(LEFT + 0.5, yy, wpx, BAR)}" fill="{t["series"][i]}"/>')
        lab = f'{v:.1f} MB' + ('' if i == 0 else f' · {MEM[0][1] / v:.1f}× less')
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
