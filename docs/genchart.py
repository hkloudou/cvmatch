#!/usr/bin/env python3
"""Renders the README benchmark SVGs (light + dark) and rewrites the
auto-generated README sections from docs/benchdata.json.

benchdata.json is produced by docs/collect.py from raw benchmark outputs
(`go test -bench . -benchtime 5x -cpu 1,4` for both build modes on the
amd64 and arm64 runners, `bench/cpp/native_bench`, and the memprobe
peak-RSS runs) — the amd64 numbers are all measured in one session on one
machine so the comparison against native OpenCV is coherent. The
bench-charts GitHub workflow re-measures on every push to main and
commits the refreshed charts, benchdata.json and README table.

Series colors are meaning-stable across every chart: green = native
OpenCV C++, blue family = cvmatch.Match, orange family =
cvmatch.MatchGray; the solid shade is the SIMD build (-tags cvmatch_asm),
the light shade is the default pure-Go build.

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

# (legend label, color slot); the row value order everywhere follows these.
SERIES = [("OpenCV C++ (native)", "native"),
          ("Match (asm)", "match"),
          ("Match (pure Go)", "matchgo"),
          ("MatchGray (asm)", "gray"),
          ("MatchGray (pure Go)", "graygo")]
KEYS = ("native", "asm4", "go4", "agray4", "gray4")
REFS = [None, 0, 0, 0, 0]  # annotate cvmatch bars with speedup vs native

ARM_SERIES = SERIES[1:]
ARM_KEYS = ("asmA4", "goA4", "agrayA4", "grayA4")
# no native baseline on the arm runner: annotate each asm bar with its
# speedup over the matching default-build bar instead
ARM_REFS = [1, None, 3, None]

THEMES = {
    "light": dict(series=dict(native="#008300",
                              match="#2a78d6", matchgo="#8ab8ec",
                              gray="#eda100", graygo="#f3cd77",
                              baseline="#8c959f"),
                  ink="#24292f", sec="#57606a", muted="#6e7781",
                  grid="#d0d7de", axis="#afb8c1"),
    "dark": dict(series=dict(native="#008300",
                             match="#3987e5", matchgo="#79a8dd",
                             gray="#c98500", graygo="#c2a05c",
                             baseline="#6e7681"),
                 ink="#e6edf3", sec="#9198a1", muted="#8b949e",
                 grid="#30363d", axis="#484f58"),
}

FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'
LEFT, RIGHT, W = 260, 128, 980
PLOT_W = W - LEFT - RIGHT
BAR, PITCH, GROUP_GAP = 14, 16, 18


def rows_for(panel_key, keys):
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


def panel(out, t, rows, y, vmax, ticks, title, series, refs, note):
    nser = len(series)
    out.append(f'<text x="{LEFT}" y="{y}" font-size="12" font-weight="600" '
               f'fill="{t["sec"]}" {FONT}>{title}</text>')
    out.append(f'<text x="{W - 4}" y="{y}" font-size="11" text-anchor="end" '
               f'fill="{t["muted"]}" {FONT}>{note}</text>')
    y += 10
    h = len(rows) * (nser * PITCH + GROUP_GAP) - GROUP_GAP + 8
    for tick in ticks:
        x = LEFT + PLOT_W * tick / vmax
        out.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + h}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{y + h + 16}" font-size="11" text-anchor="middle" '
                   f'fill="{t["muted"]}" {FONT} style="font-variant-numeric:tabular-nums">{tick:g}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT}" y2="{y + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    yy = y + 4
    for label, *vals in rows:
        out.append(f'<text x="{LEFT - 10}" y="{yy + nser * PITCH / 2 + 4}" font-size="12" '
                   f'text-anchor="end" fill="{t["sec"]}" {FONT}>{label}</text>')
        for i, v in enumerate(vals):
            wpx = PLOT_W * min(v, vmax) / vmax
            color = t["series"][series[i][1]]
            out.append(f'<path d="{bar_path(LEFT + 0.5, yy, wpx, BAR)}" fill="{color}"/>')
            lab = f'{v:,.0f} ms' if v >= 100 else f'{v:,.1f} ms'
            if refs[i] is not None:
                lab += f' · {vals[refs[i]] / v:.2f}×'
            out.append(f'<text x="{LEFT + wpx + 7:.1f}" y="{yy + BAR - 3}" font-size="11" '
                       f'fill="{t["sec"]}" {FONT} style="font-variant-numeric:tabular-nums">{lab}</text>')
            yy += PITCH
        yy += GROUP_GAP
    return y + h + 34


def legend(out, t, y, series):
    x = 20
    for name, slot in series:
        out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{t["series"][slot]}"/>')
        out.append(f'<text x="{x + 15}" y="{y}" font-size="12" fill="{t["ink"]}" {FONT}>{name}</text>')
        x += 15 + 7 * len(name) + 26
    return y + 20


PANELS = (("hd", "HD desktop + noise scenes"), ("4k", "4K scenes"),
          ("photo", "Real photographs (OpenCV samples/data)"))


def speed_chart(mode):
    t = THEMES[mode]
    out = []
    out.append(f'<text x="20" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'Template matching speed — TM_CCOEFF_NORMED, end-to-end call, identical output</text>')
    out.append(f'<text x="20" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               f'{HOST}, one session · asm = -tags cvmatch_asm build, pure Go = default build '
               '· OpenCV matchTemplate is single-threaded by design</text>')
    y = legend(out, t, 60, SERIES)
    note = 'ms — lower is better · ×  = speedup vs native C++ (&lt;1 = slower)'
    for panel_key, title in PANELS:
        rows = rows_for(panel_key, KEYS)
        if not rows:
            continue
        vmax, ticks = axis_scale(rows)
        y = panel(out, t, rows, y + 8, vmax, ticks, title, SERIES, REFS, note)
    return svg(out, y)


def arm_speed_chart(mode):
    """arm64 twin of the speed chart (no native OpenCV baseline is built on
    the arm runner, so the asm bars annotate vs the default build)."""
    t = THEMES[mode]
    out = []
    out.append(f'<text x="20" y="20" font-size="15" font-weight="600" fill="{t["ink"]}" {FONT}>'
               'arm64 — the same pipeline, NEON kernels vs default build, identical output bits</text>')
    out.append(f'<text x="20" y="38" font-size="12" fill="{t["muted"]}" {FONT}>'
               f'{DATA.get("hostArm64", "arm64 CI runner")}, one session · '
               'asm = -tags cvmatch_asm build, pure Go = default build</text>')
    y = legend(out, t, 60, ARM_SERIES)
    note = 'ms — lower is better · ×  = asm speedup vs the default build'
    for panel_key, title in PANELS:
        rows = rows_for(panel_key, ARM_KEYS)
        if not rows:
            continue
        vmax, ticks = axis_scale(rows)
        y = panel(out, t, rows, y + 8, vmax, ticks, title, ARM_SERIES, ARM_REFS, note)
    return svg(out, y)


def mem_chart(mode):
    t = THEMES[mode]
    rows = [("OpenCV C++ (native)", MEM.get("native"), "native"),
            ("cvmatch.Match", MEM.get("match"), "match"),
            ("cvmatch.MatchGray", MEM.get("gray"), "gray"),
            ("idle Go process (baseline)", MEM.get("baseline"), "baseline")]
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


def match_table(lines, keys, header):
    """One Match matrix table; bold = best cvmatch cell per scene."""
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for key, label, _ in LAYOUT:
        s = SC.get(key)
        if not s or not all(k in s for k in keys):
            continue
        cells = [s[k] for k in keys]
        go_cells = cells[1:] if keys[0] == "native" else cells
        best = min(go_cells)
        fmt = [f"{cells[0]:.1f}"] if keys[0] == "native" else []
        fmt += [f"**{v:.1f}**" if v == best else f"{v:.1f}" for v in go_cells]
        lines.append(f"| {label} | " + " | ".join(fmt) + " |")


def matrix_markdown():
    """The README 'full matrix' table between the benchmatrix markers."""
    lines = [
        "`Match`, milliseconds, measured on " + (HOST or "the CI runner") + ".",
        "Native C++ is the best of 7 end-to-end runs, single-threaded because",
        "that is how OpenCV's `matchTemplate` runs; the cvmatch columns are",
        "`go test -benchtime 5x` averages from `-cpu 1,4` — the asm columns",
        "are the `-tags cvmatch_asm` build, the pure-Go columns the default",
        "build:",
        "",
    ]
    match_table(lines, ("native", "asm1", "asm4", "go1", "go4"),
                "| scene | native C++ | asm 1T | asm 4T | pure-Go 1T | pure-Go 4T |")
    g = SC.get("noise1080p_sub128", {})
    if all(k in g for k in ("agray1", "agray4", "gray1", "gray4")):
        lines += [
            "",
            f"`MatchGray` at 1080p/128 for scale: asm {g['agray1']:.1f} / {g['agray4']:.1f} ms,",
            f"pure Go {g['gray1']:.1f} / {g['gray4']:.1f} ms (1T / 4T). Native C++ has no gray",
            "row in this suite — a fair baseline would need cvtColor + 1-channel",
            "matchTemplate timed end-to-end, which was not measured; treat MatchGray",
            "numbers as cvmatch-internal.",
        ]
    if any("asmA1" in s for s in SC.values()):
        lines += [
            "",
            "**arm64** — the same `Match` matrix from the arm64 CI leg (" +
            (DATA.get("hostArm64") or "ubuntu-24.04-arm") + "),",
            "NEON kernels vs the default build, bit-identical output to the",
            "amd64 rows above:",
            "",
        ]
        match_table(lines, ("asmA1", "asmA4", "goA1", "goA4"),
                    "| scene | asm 1T | asm 4T | pure-Go 1T | pure-Go 4T |")
        ga = SC.get("noise1080p_sub128", {})
        if all(k in ga for k in ("agrayA1", "agrayA4", "grayA1", "grayA4")):
            lines += [
                "",
                f"`MatchGray` at 1080p/128 on arm64: asm {ga['agrayA1']:.1f} / "
                f"{ga['agrayA4']:.1f} ms, pure Go {ga['grayA1']:.1f} / "
                f"{ga['grayA4']:.1f} ms (1T / 4T).",
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
