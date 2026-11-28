"""Hand-drawn SVGs for the results site: how the architecture changed from v1 to v8, and the inside
of the 3D model (v8). Used by build_report.py (front page and architecture.html)."""

from __future__ import annotations

from html import escape

BLUE, ORANGE, GREEN, INK, INK2, LINE, FILL = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#c9c8c2", "#f3f2ee"
FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'

# rows: (group, label, [(first version, last version, text, changed?)])
EVOLUTION = [
    ("network", "patch embedding", [(1, 6, "linear, 4°×4° patches", False),
                                    (7, 7, "anti-aliased (conv + low-pass)", True),
                                    (8, 8, "anti-aliased, per level", True)]),
    ("network", "tokens", [(1, 7, "4050: one per patch, all levels folded in", False),
                           (8, 8, "9 × 4050: levels + surface", True)]),
    ("network", "positions", [(1, 7, "learned, per patch", False),
                              (8, 8, "rotary + level", True)]),
    ("network", "attention", [(1, 7, "6 global heads per block, across the globe", False),
                              (8, 8, "4 horizontal + 2 vertical", True)]),
    ("network", "LayerNorms", [(1, 2, "plain", False),
                               (3, 8, "noise-conditioned: a stochastic model", True)]),
    ("network", "output head", [(1, 1, "linear un-patchify", False),
                                (2, 7, "+ 3×3 conv, + 10 diagnostics", True),
                                (8, 8, "per-level heads", True)]),
    ("network", "size", [(1, 7, "dim 384 · 8 blocks · 16.7 M", False),
                         (8, 8, "384 · 8 · 14.4 M", True)]),
    ("training", "loss", [(1, 1, "MSE", False), (2, 2, "MSE after correctors", True),
                          (3, 4, "fair CRPS, 2 members", True), (5, 8, "CRPS + spectral energy score", True)]),
    ("training", "rollouts", [(1, 3, "1 → 4 → 8 steps", False), (4, 4, "4–16 steps", True),
                              (5, 8, "4–24 steps, last-step grad", True)]),
    ("training", "grid shifts", [(1, 5, "none", False), (6, 8, "random sub-patch jitter", True)]),
    ("outside", "correctors", [(1, 1, "dry air, positivity", False),
                               (2, 8, "+ moisture and energy budgets (ACE)", True)]),
    ("outside", "long runs", [(1, 6, "as is (seam filter optional)", False),
                              (7, 8, "patch jitter + seam filter", True)]),
]
GROUPS = {"network": "the network", "training": "how it is trained", "outside": "around the network (every step)"}


def evolution_svg() -> str:
    X0, CW, RH, TOP = 190, 118, 34, 58
    W = X0 + 8 * CW + 10
    rows, y, last_group = [], TOP, None
    for group, label, spans in EVOLUTION:
        if group != last_group:
            y += 14 if last_group else 0
            rows.append(f'<text x="6" y="{y + 4}" font-size="11.5" font-weight="700" fill="{INK2}" '
                        f'letter-spacing=".04em">{escape(GROUPS[group].upper())}</text>')
            y += 14
            last_group = group
        rows.append(f'<text x="14" y="{y + RH / 2 + 4.5}" font-size="13" fill="{INK}">{escape(label)}</text>')
        for a, b, text, changed in spans:
            x = X0 + (a - 1) * CW + 3
            w = (b - a + 1) * CW - 6
            fill = "#e6f0fb" if changed else FILL
            stroke = BLUE if changed else LINE
            rows.append(f'<rect x="{x}" y="{y + 3}" width="{w}" height="{RH - 6}" rx="7" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="{1.4 if changed else 1}"/>')
            fs = 12 if w > 200 else 11
            # wrap long text in narrow cells onto two lines
            if len(text) * fs * 0.52 > w - 10 and " " in text:
                cut = text.rfind(" ", 0, len(text) // 2 + 6) or len(text) // 2
                parts = [text[:cut], text[cut + 1:]]
                for k, t in enumerate(parts):
                    rows.append(f'<text x="{x + w / 2}" y="{y + RH / 2 - 1 + (k - 0.5) * 12.5 + 4}" font-size="{fs - 0.5}" '
                                f'text-anchor="middle" fill="{INK}">{escape(t)}</text>')
            else:
                rows.append(f'<text x="{x + w / 2}" y="{y + RH / 2 + 4.5}" font-size="{fs}" text-anchor="middle" '
                            f'fill="{INK}">{escape(text)}</text>')
        y += RH
    H = y + 34
    head = [f'<text x="{X0 + (v - 0.5) * CW}" y="{TOP - 14}" font-size="14" font-weight="700" text-anchor="middle" '
            f'fill="{ORANGE if v == 8 else INK}">v{v}</text>' for v in range(1, 9)]
    head += [f'<line x1="{X0 + v * CW}" x2="{X0 + v * CW}" y1="{TOP - 6}" y2="{H - 30}" stroke="{LINE}" '
             f'stroke-dasharray="2 4"/>' for v in range(1, 8)]
    notes = {1: "first model", 2: "30 years", 3: "stochastic", 4: "long runs", 5: "spectral",
             6: "jitter", 7: "decade-stable", 8: "3D"}
    head += [f'<text x="{X0 + (v - 0.5) * CW}" y="{TOP - 1}" font-size="10.5" text-anchor="middle" fill="{INK2}">'
             f'{notes[v]}</text>' for v in range(1, 9)]
    legend = (f'<rect x="{X0}" y="{H - 22}" width="16" height="12" rx="3" fill="#e6f0fb" stroke="{BLUE}"/>'
              f'<text x="{X0 + 22}" y="{H - 12}" font-size="11.5" fill="{INK2}">changed in this version</text>'
              f'<rect x="{X0 + 190}" y="{H - 22}" width="16" height="12" rx="3" fill="{FILL}" stroke="{LINE}"/>'
              f'<text x="{X0 + 212}" y="{H - 12}" font-size="11.5" fill="{INK2}">as before</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" {FONT} role="img" '
            f'aria-label="How the architecture, training and per-step corrections changed from v1 to v8" '
            f'style="width:100%;height:auto;background:#fcfcfb">' + "".join(head + rows) + legend + "</svg>")


def v8_svg() -> str:
    """Left: the 3D token stack and the two kinds of attention. Right: the inside of one v8 block."""
    out = []
    # --- token stack: 9 parallelogram layers
    x0, y0, w, d, gap = 30, 70, 250, 60, 30
    names = ["level 0 (top)"] + [f"level {k}" for k in range(1, 7)] + ["level 7 (near surface)", "surface + forcings"]
    for k in range(9):
        y = y0 + k * gap
        col = "#fdf1e8" if k == 8 else "#eef4fc"
        st = ORANGE if k == 8 else BLUE
        out.append(f'<path d="M{x0},{y + d / 2} L{x0 + d},{y} L{x0 + d + w},{y} L{x0 + w},{y + d / 2} Z" '
                   f'fill="{col}" stroke="{st}" stroke-width="1" opacity="0.95"/>')
        out.append(f'<text x="{x0 + d + w + 10}" y="{y + 14}" font-size="11.5" fill="{INK2}">{names[k]}</text>')
    # a column through the stack
    cx = x0 + d / 2 + 110
    out.append(f'<line x1="{cx + 14}" y1="{y0 + 10}" x2="{cx + 14}" y2="{y0 + 8 * gap + 22}" stroke="{ORANGE}" '
               f'stroke-width="2.4" marker-start="url(#va)" marker-end="url(#va)"/>')
    out.append(f'<text x="{x0 + 6}" y="{y0 + 8 * gap + d / 2 + 30}" font-size="12" fill="{ORANGE}" font-weight="600">'
               f'vertical heads: one 4°×4° column, its 9 tokens (orange arrow)</text>')
    # horizontal arrow on one level
    yl = y0 + 3 * gap + 16
    out.append(f'<path d="M{x0 + 40},{yl} C{x0 + 90},{yl - 34} {x0 + 170},{yl - 34} {x0 + 220},{yl}" fill="none" '
               f'stroke="{BLUE}" stroke-width="2.2" marker-end="url(#ha)"/>')
    out.append(f'<text x="{x0 + 6}" y="{y0 - 22}" font-size="12" fill="{BLUE}" font-weight="600">horizontal heads: one level, '
               f'all 4050 patches</text>')
    out.append(f'<text x="{x0}" y="{y0 - 44}" font-size="13.5" font-weight="700" fill="{INK}">tokens: 9 levels × 4050 patches '
               f'(residual stream 9 × 4050 × 384)</text>')
    # --- one block
    bx, by = 560, 60
    out.append(f'<rect x="{bx}" y="{by - 18}" width="540" height="372" rx="12" fill="#fafaf8" stroke="{LINE}"/>')
    out.append(f'<text x="{bx + 14}" y="{by + 2}" font-size="13" font-weight="700" fill="{INK}">one v8 block (8 in series)</text>')
    sy = by + 280  # stream
    out.append(f'<line x1="{bx + 16}" y1="{sy}" x2="{bx + 522}" y2="{sy}" stroke="{BLUE}" stroke-width="3.5" '
               f'marker-end="url(#ha)"/>')
    out.append(f'<text x="{bx + 18}" y="{sy + 20}" font-size="11" fill="{INK2}" font-family="ui-monospace,monospace">'
               f'hook_resid_pre</text><text x="{bx + 440}" y="{sy + 20}" font-size="11" fill="{INK2}" '
               f'font-family="ui-monospace,monospace">hook_resid_post</text>')

    def box(x, y, w, h, title, sub="", stroke=INK):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="#fff" stroke="{stroke}" stroke-width="1.2"/>'
                   f'<text x="{x + w / 2}" y="{y + 17}" font-size="12" font-weight="700" text-anchor="middle" fill="{INK}">{title}</text>')
        if sub:
            out.append(f'<text x="{x + w / 2}" y="{y + 32}" font-size="10.5" text-anchor="middle" fill="{INK2}">{sub}</text>')

    box(bx + 30, by + 150, 56, 44, "LN", "← noise")
    # attention with 6 heads
    ax, ay = bx + 108, by + 40
    out.append(f'<rect x="{ax}" y="{ay}" width="232" height="160" rx="9" fill="#fff" stroke="{INK}" stroke-width="1.2"/>')
    out.append(f'<text x="{ax + 116}" y="{ay + 18}" font-size="12.5" font-weight="700" text-anchor="middle" fill="{INK}">'
               f'self-attention · 6 heads × 64</text>')
    for h in range(6):
        vert = h >= 4
        hx = ax + 12 + h * 36
        out.append(f'<rect x="{hx}" y="{ay + 30}" width="30" height="64" rx="5" fill="{"#fdf1e8" if vert else "#eef4fc"}" '
                   f'stroke="{ORANGE if vert else BLUE}" stroke-width="1.3"/>'
                   f'<text x="{hx + 15}" y="{ay + 66}" font-size="12" text-anchor="middle" fill="{INK}">h{h}</text>')
    out.append(f'<text x="{ax + 70}" y="{ay + 112}" font-size="10.5" text-anchor="middle" fill="{BLUE}">across the level</text>'
               f'<text x="{ax + 70}" y="{ay + 125}" font-size="10.5" text-anchor="middle" fill="{BLUE}">4050 tokens, rotary</text>'
               f'<text x="{ax + 190}" y="{ay + 112}" font-size="10.5" text-anchor="middle" fill="{ORANGE}">up and down</text>'
               f'<text x="{ax + 190}" y="{ay + 125}" font-size="10.5" text-anchor="middle" fill="{ORANGE}">a column (9)</text>')
    out.append(f'<text x="{ax + 116}" y="{ay + 142}" font-size="9.5" text-anchor="middle" fill="{INK2}" '
               f'font-family="ui-monospace,monospace">hook_pattern_h · hook_pattern_v</text>'
               f'<text x="{ax + 116}" y="{ay + 154}" font-size="9.5" text-anchor="middle" fill="{INK2}" '
               f'font-family="ui-monospace,monospace">hook_result (per head)</text>')
    box(bx + 362, by + 150, 56, 44, "LN", "← noise")
    box(bx + 428, by + 120, 100, 74, "MLP", "384→1536→384")
    # wiring
    def arrow(x1, y1, x2, y2, c=INK):
        out.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c}" stroke-width="1.3" fill="none" marker-end="url(#ka)"/>')
    arrow(bx + 58, sy, bx + 58, by + 196)
    arrow(bx + 86, by + 172, ax, by + 172)
    arrow(ax + 232, by + 160, ax + 246, by + 160)
    out.append(f'<path d="M{ax + 246},{by + 160} L{ax + 246},{sy - 9}" stroke="{INK}" stroke-width="1.3" fill="none" marker-end="url(#ka)"/>'
               f'<circle cx="{ax + 246}" cy="{sy}" r="8" fill="#fff" stroke="{INK}"/><text x="{ax + 246}" y="{sy + 4}" '
               f'font-size="12" text-anchor="middle">+</text>')
    arrow(bx + 390, sy, bx + 390, by + 196)
    arrow(bx + 418, by + 172, bx + 428, by + 172)
    out.append(f'<path d="M{bx + 478},{by + 194} L{bx + 478},{sy - 9}" stroke="{INK}" stroke-width="1.3" fill="none" '
               f'marker-end="url(#ka)"/><circle cx="{bx + 478}" cy="{sy}" r="8" fill="#fff" stroke="{INK}"/>'
               f'<text x="{bx + 478}" y="{sy + 4}" font-size="12" text-anchor="middle">+</text>')
    out.append(f'<text x="{bx + 270}" y="{by + 330}" font-size="11" text-anchor="middle" fill="{INK2}">'
               f'MLPs and LayerNorms act per token; only attention moves information:</text>'
               f'<text x="{bx + 270}" y="{by + 345}" font-size="11" text-anchor="middle" fill="{INK2}">'
               f'heads 0–3 across a level, heads 4–5 up and down a column</text>')
    defs = (f'<defs><marker id="ha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>'
            f'<marker id="va" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{ORANGE}"/></marker>'
            f'<marker id="ka" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 430" {FONT} role="img" '
            f'aria-label="v8: tokens for 8 levels and the surface over 4050 patches; in each block 4 heads attend across '
            f'a level and 2 up and down a column" style="width:100%;height:auto;background:#fcfcfb">'
            + defs + "".join(out) + "</svg>")
