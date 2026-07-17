"""Generate a code-aligned Phase-Path Network overview figure.

The figure is intentionally produced from native drawing primitives rather than
embedding the earlier reference image. It exports both an editable SVG and a
high-resolution PNG for manuscript or slide use.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("diagnostics/paper_artifacts/figures")
PNG_PATH = OUT_DIR / "optimized_phase_path_flowchart.png"
SVG_PATH = OUT_DIR / "optimized_phase_path_flowchart.svg"

W, H = 3200, 1800


PALETTE = {
    "ink": "#20242a",
    "muted": "#5f6670",
    "line": "#6f7782",
    "grid": "#e9edf2",
    "input": ("#eef6ff", "#3f7fc4"),
    "window": ("#fff7e8", "#d49b27"),
    "encoder": ("#eff8ec", "#6d9d56"),
    "embed": ("#edf7f7", "#4b9aa0"),
    "proto": ("#f4f0ff", "#876cb6"),
    "assign": ("#f5f1ff", "#8a70bd"),
    "path": ("#fff1e9", "#e3873c"),
    "unc": ("#eaf8f8", "#52a4a8"),
    "fusion": ("#f5f7fb", "#57677c"),
    "pred": ("#edf4ff", "#4f83c4"),
    "train": ("#fff3f5", "#d27a8d"),
    "memory": ("#f0faef", "#65a35e"),
    "loss": ("#fff8ed", "#cc8a2c"),
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = {
        "regular": [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\times.ttf",
        ],
        "bold": [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf",
            r"C:\Windows\Fonts\timesbd.ttf",
        ],
        "mono": [
            r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\cour.ttf",
        ],
    }
    for item in candidates.get(name, candidates["regular"]):
        if Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


FONT_TITLE = font("bold", 42)
FONT_SECTION = font("bold", 25)
FONT_BOX = font("bold", 25)
FONT_TEXT = font("regular", 22)
FONT_SMALL = font("regular", 18)
FONT_MONO = font("mono", 20)


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#38414c"/></marker>',
            '<marker id="arrowDashed" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#9b6571"/></marker>',
            "</defs>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str,
        sw: float = 3,
        rx: float = 22,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def line(
        self,
        points: Sequence[tuple[float, float]],
        color: str = "#38414c",
        width: float = 4,
        dash: str | None = None,
        arrow: bool = True,
    ) -> None:
        d = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrowDashed)"' if dash and arrow else (' marker-end="url(#arrow)"' if arrow else "")
        self.items.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 22,
        weight: str = "400",
        color: str = "#20242a",
        anchor: str = "start",
        family: str = "Arial",
    ) -> None:
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def tspans(
        self,
        x: float,
        y: float,
        lines: Iterable[str],
        size: int = 20,
        color: str = "#20242a",
        family: str = "Arial",
        anchor: str = "middle",
        leading: int = 28,
        weight: str = "400",
    ) -> None:
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        )
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else leading
            self.items.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>')
        self.items.append("</text>")

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str, sw: float = 3) -> None:
        self.items.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def polyline(self, pts: Sequence[tuple[float, float]], color: str, width: float = 3) -> None:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.items.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def save(self, path: Path) -> None:
        self.items.append("</svg>")
        path.write_text("\n".join(self.items), encoding="utf-8")


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=font_obj, fill=fill)


def wrap_lines(text: str, chars: int) -> list[str]:
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=chars) if part else [""])
    return lines


def rounded_box(
    draw: ImageDraw.ImageDraw,
    svg: Svg,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: Sequence[str],
    fill: str,
    stroke: str,
    title_color: str = PALETTE["ink"],
    body_color: str = PALETTE["muted"],
    dash: tuple[int, ...] | None = None,
    rx: int = 22,
) -> None:
    if dash:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=rx, fill=fill, outline=stroke, width=3)
        # Overlay simple dashed border on the PNG.
        step = dash[0] + dash[1]
        for xx in range(x + rx, x + w - rx, step):
            draw.line((xx, y, min(xx + dash[0], x + w - rx), y), fill=stroke, width=4)
            draw.line((xx, y + h, min(xx + dash[0], x + w - rx), y + h), fill=stroke, width=4)
        for yy in range(y + rx, y + h - rx, step):
            draw.line((x, yy, x, min(yy + dash[0], y + h - rx)), fill=stroke, width=4)
            draw.line((x + w, yy, x + w, min(yy + dash[0], y + h - rx)), fill=stroke, width=4)
        svg.rect(x, y, w, h, fill, stroke, dash=f"{dash[0]} {dash[1]}")
    else:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=rx, fill=fill, outline=stroke, width=3)
        svg.rect(x, y, w, h, fill, stroke)

    title_lines = wrap_lines(title, max(10, int(w / 15)))
    title_y = y + 34 if len(title_lines) == 1 else y + 25
    for idx, title_line in enumerate(title_lines[:2]):
        draw_centered_text(draw, (x + w // 2, title_y + idx * 28), title_line, FONT_BOX, title_color)
    svg.tspans(
        x + w / 2,
        title_y + 8,
        title_lines[:2],
        size=25,
        weight="700",
        color=title_color,
        leading=28,
    )
    start_y = y + 72 + max(0, len(title_lines[:2]) - 1) * 20
    flat_lines: list[str] = []
    for line in body:
        flat_lines.extend(wrap_lines(line, max(12, int(w / 13))))
    for idx, line in enumerate(flat_lines[:6]):
        f = FONT_MONO if any(token in line for token in ["=", "q_", "S_", "z_", "U_", "A_", "pi_"]) else FONT_TEXT
        draw_centered_text(draw, (x + w // 2, start_y + idx * 30), line, f, body_color)
    svg.tspans(x + w / 2, start_y + 8, flat_lines[:6], size=20, color=body_color, leading=28)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    svg: Svg,
    points: Sequence[tuple[int, int]],
    color: str = "#38414c",
    width: int = 5,
    dashed: bool = False,
) -> None:
    line_color = "#9b6571" if dashed else color
    if dashed:
        for a, b in zip(points[:-1], points[1:]):
            draw_dashed_segment(draw, a, b, line_color, width)
        svg.line(points, color="#9b6571", width=4, dash="13 10")
    else:
        draw.line(points, fill=line_color, width=width, joint="curve")
        svg.line(points, color=line_color, width=width)

    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 17
    left = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
    right = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill=line_color)


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: str,
    width: int,
    dash_len: int = 16,
    gap_len: int = 12,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    ux, uy = (x2 - x1) / dist, (y2 - y1) / dist
    pos = 0
    while pos < dist:
        end = min(pos + dash_len, dist)
        draw.line((x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end), fill=color, width=width)
        pos += dash_len + gap_len


def draw_input_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    chart = (x + 30, y + 84, x + 190, y + 178)
    draw.rounded_rectangle(chart, radius=8, fill="#ffffff", outline="#aab5c0", width=2)
    svg.rect(chart[0], chart[1], chart[2] - chart[0], chart[3] - chart[1], "#ffffff", "#aab5c0", sw=2, rx=8)
    colors = ["#3f7fc4", "#ef8b2c", "#60a653"]
    for cidx, color in enumerate(colors):
        pts = []
        baseline = chart[1] + 22 + cidx * 28
        for i in range(58):
            px = chart[0] + 8 + i * 2.5
            py = baseline + 7 * math.sin(i * 0.45 + cidx) + 4 * math.sin(i * 1.4)
            pts.append((px, py))
        draw.line(pts, fill=color, width=3)
        svg.polyline(pts, color, width=3)


def draw_window_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    colors = ["#3f7fc4", "#ef8b2c", "#60a653"]
    for idx, dx in enumerate([0, 34, 68]):
        bx, by = x + 42 + dx, y + 78 + idx * 20
        draw.rounded_rectangle((bx, by, bx + 132, by + 82), radius=10, fill="#ffffff", outline=colors[idx], width=3)
        svg.rect(bx, by, 132, 82, "#ffffff", colors[idx], sw=3, rx=10)
        pts = []
        for i in range(40):
            px = bx + 10 + i * 2.8
            py = by + 42 + 8 * math.sin(i * 0.5 + idx) + 3 * math.sin(i * 1.2)
            pts.append((px, py))
        draw.line(pts, fill=colors[idx], width=3)
        svg.polyline(pts, colors[idx], width=3)


def draw_encoder_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    base_x, base_y = x + 54, y + 96
    for i in range(5):
        xx = base_x + i * 26
        h = 96 - i * 8
        poly = [(xx, base_y + i * 4), (xx + 46, base_y + 20 + i * 4), (xx + 46, base_y + h), (xx, base_y + h - 20)]
        draw.polygon(poly, fill="#d8ecd1", outline="#6d9d56")
        pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in poly)
        svg.items.append(f'<polygon points="{pts}" fill="#d8ecd1" stroke="#6d9d56" stroke-width="2"/>')


def draw_embedding_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    for i in range(5):
        bx = x + 42 + i * 32
        by = y + 110
        height = [58, 82, 45, 70, 54][i]
        draw.rounded_rectangle((bx, by - height, bx + 20, by), radius=4, fill="#8fc7c9", outline="#4b9aa0", width=2)
        svg.rect(bx, by - height, 20, height, "#8fc7c9", "#4b9aa0", sw=2, rx=4)
    draw_centered_text(draw, (x + 120, y + 154), "h_1 ... h_N", FONT_MONO, PALETTE["muted"])
    svg.text(x + 120, y + 160, "h_1 ... h_N", size=20, family="Consolas", color=PALETTE["muted"], anchor="middle")


def draw_proto_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    for row in range(3):
        for col in range(4):
            fill = ["#d9ccf4", "#efe7ff", "#cfc0ef"][(row + col) % 3]
            bx, by = x + 76 + col * 30, y + 174 + row * 26
            draw.rectangle((bx, by, bx + 22, by + 18), fill=fill, outline="#876cb6")
            svg.rect(bx, by, 22, 18, fill, "#876cb6", sw=1, rx=2)
    draw_centered_text(draw, (x + 160, y + 268), "P_y,k = U_y,k V_y,k^T", FONT_MONO, "#614b8f")
    svg.text(x + 160, y + 274, "P_y,k = U_y,k V_y,k^T", size=19, family="Consolas", color="#614b8f", anchor="middle")


def draw_assignment_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    vals = [
        ["#ede5ff", "#cdbcef", "#a991dc", "#dfd5f8"],
        ["#c5b2ed", "#e8e1fb", "#d2c3f2", "#9c83d4"],
        ["#e3daf9", "#aa91dd", "#cdbcef", "#eee8fb"],
    ]
    gx, gy = x + 82, y + 150
    for r in range(3):
        for c in range(4):
            draw.rectangle((gx + c * 32, gy + r * 28, gx + c * 32 + 26, gy + r * 28 + 22), fill=vals[r][c], outline="#8a70bd")
            svg.rect(gx + c * 32, gy + r * 28, 26, 22, vals[r][c], "#8a70bd", sw=1, rx=2)
    draw_centered_text(draw, (x + 150, y + 238), "q_t,y,k", FONT_MONO, "#6c549f")
    svg.text(x + 150, y + 244, "q_t,y,k", size=20, family="Consolas", color="#6c549f", anchor="middle")


def draw_prediction_icon(draw: ImageDraw.ImageDraw, svg: Svg, x: int, y: int) -> None:
    labels = ["Class 1", "Class 2", "Class C"]
    widths = [110, 66, 38]
    colors = ["#4f83c4", "#ef8b2c", "#60a653"]
    for idx, label in enumerate(labels):
        yy = y + 94 + idx * 42
        draw.text((x + 38, yy - 10), label, font=FONT_SMALL, fill=PALETTE["ink"])
        svg.text(x + 38, yy + 7, label, size=18, color=PALETTE["ink"])
        draw.rounded_rectangle((x + 120, yy - 12, x + 120 + widths[idx], yy + 12), radius=4, fill=colors[idx])
        svg.rect(x + 120, yy - 12, widths[idx], 24, colors[idx], colors[idx], sw=1, rx=4)
        draw.text((x + 240, yy - 11), f"{[0.62, 0.27, 0.11][idx]:.2f}", font=FONT_SMALL, fill=PALETTE["ink"])
        svg.text(x + 240, yy + 7, f"{[0.62, 0.27, 0.11][idx]:.2f}", size=18, color=PALETTE["ink"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    svg = Svg(W, H)

    # Section backgrounds.
    draw.rounded_rectangle((50, 130, 3150, 920), radius=30, fill="#fbfcfe", outline="#dfe5ec", width=2)
    svg.rect(50, 130, 3100, 790, "#fbfcfe", "#dfe5ec", sw=2, rx=30)
    draw.rounded_rectangle((50, 1010, 3150, 1680), radius=30, fill="#fffafa", outline="#ead9dd", width=2)
    svg.rect(50, 1010, 3100, 670, "#fffafa", "#ead9dd", sw=2, rx=30)

    draw.text((80, 48), "Phase-Path Network: code-aligned method overview", font=FONT_TITLE, fill=PALETTE["ink"])
    svg.text(80, 90, "Phase-Path Network: code-aligned method overview", size=42, weight="700")
    draw.text((80, 146), "Inference path: solid arrows.  Training-only objectives and memory update: dashed arrows.", font=FONT_SECTION, fill=PALETTE["muted"])
    svg.text(80, 176, "Inference path: solid arrows. Training-only objectives and memory update: dashed arrows.", size=25, weight="700", color=PALETTE["muted"])
    draw.text((80, 1038), "Training side objectives", font=FONT_SECTION, fill="#9b6571")
    svg.text(80, 1068, "Training side objectives", size=25, weight="700", color="#9b6571")

    # Main boxes.
    boxes = {
        "input": (90, 260, 250, 270),
        "windows": (410, 260, 270, 270),
        "encoder": (750, 260, 250, 270),
        "embed": (1060, 260, 270, 270),
        "proto": (1400, 225, 320, 330),
        "assign": (1795, 260, 300, 270),
        "score": (2160, 205, 400, 620),
        "score_proto": (2185, 305, 350, 135),
        "score_path": (2185, 485, 350, 145),
        "score_unc": (2185, 675, 350, 125),
        "fusion": (2635, 405, 270, 210),
        "pred": (2930, 390, 240, 240),
    }

    rounded_box(draw, svg, *boxes["input"], "Time series", [], *PALETTE["input"])
    draw_input_icon(draw, svg, *boxes["input"][:2])
    rounded_box(draw, svg, *boxes["windows"], "Windows", [], *PALETTE["window"])
    draw_window_icon(draw, svg, *boxes["windows"][:2])
    rounded_box(draw, svg, *boxes["encoder"], "Encoder", [], *PALETTE["encoder"])
    draw_encoder_icon(draw, svg, *boxes["encoder"][:2])
    rounded_box(draw, svg, *boxes["embed"], "Embeddings", [], *PALETTE["embed"])
    draw_embedding_icon(draw, svg, *boxes["embed"][:2])
    rounded_box(
        draw,
        svg,
        *boxes["proto"],
        "Phase dictionary",
        ["class-conditioned", "low-rank prototypes", "template dist + residual"],
        *PALETTE["proto"],
    )
    draw_proto_icon(draw, svg, *boxes["proto"][:2])
    rounded_box(
        draw,
        svg,
        *boxes["assign"],
        "Soft phase assignment",
        ["q_t,y,k = softmax", "over K phases"],
        *PALETTE["assign"],
    )
    draw_assignment_icon(draw, svg, *boxes["assign"][:2])

    rounded_box(draw, svg, *boxes["score"], "Score builder", ["inputs: q_t,y,k, distances,", "and class phase graph"], *PALETTE["fusion"])
    rounded_box(draw, svg, *boxes["score_proto"], "Prototype term", ["S_proto(y)", "-mean template distance"], *PALETTE["proto"])
    rounded_box(draw, svg, *boxes["score_path"], "Path term", ["learn pi_y, A_y", "HMM forward", "G_y = logP - iid"], *PALETTE["path"])
    rounded_box(draw, svg, *boxes["score_unc"], "Uncertainty term", ["entropy(q) + residual", "U_y penalty"], *PALETTE["unc"])
    rounded_box(draw, svg, *boxes["fusion"], "VETO fusion", ["z_y = wp S_proto", "+ wg G_y - wu U_y"], *PALETTE["fusion"])
    rounded_box(draw, svg, *boxes["pred"], "Prediction", [], *PALETTE["pred"])
    draw_prediction_icon(draw, svg, *boxes["pred"][:2])

    # Solid inference arrows.
    chain = ["input", "windows", "encoder", "embed", "proto", "assign", "score", "fusion", "pred"]
    for a, b in zip(chain[:-1], chain[1:]):
        ax, ay, aw, ah = boxes[a]
        bx, by, bw, bh = boxes[b]
        draw_arrow(draw, svg, [(ax + aw, ay + ah // 2), (bx - 18, by + bh // 2)])

    # Training side boxes.
    train_boxes = {
        "cf": (410, 1170, 270, 175),
        "cf_loss": (740, 1170, 270, 175),
        "rel": (1220, 1150, 285, 225),
        "memory": (1560, 1150, 285, 225),
        "mem_loss": (1900, 1170, 300, 175),
        "total": (2635, 1170, 270, 175),
    }
    rounded_box(draw, svg, *train_boxes["cf"], "Counterfactual path", ["swap / delete / insert", "transition break"], *PALETTE["train"], dash=(14, 10))
    rounded_box(draw, svg, *train_boxes["cf_loss"], "Margin loss", ["real score > cf score", "L_cf"], *PALETTE["loss"], dash=(14, 10))
    rounded_box(draw, svg, *train_boxes["rel"], "Reliability gate", ["residual + entropy", "+ transition residual", "rel > threshold"], *PALETTE["train"], dash=(14, 10))
    rounded_box(draw, svg, *train_boxes["memory"], "Confirmed memory", ["candidate buffer", "evidence counter", "EMA update"], *PALETTE["memory"], dash=(14, 10))
    rounded_box(draw, svg, *train_boxes["mem_loss"], "Memory / transition", ["L_mem and L_trans", "regularize q, A, M"], *PALETTE["loss"], dash=(14, 10))
    rounded_box(draw, svg, *train_boxes["total"], "Training loss", ["L = L_cls + L_cf", "+ L_trans + L_mem"], *PALETTE["loss"], dash=(14, 10))

    # Dashed training arrows.
    draw_arrow(draw, svg, [(boxes["windows"][0] + boxes["windows"][2] // 2, boxes["windows"][1] + boxes["windows"][3]), (boxes["windows"][0] + boxes["windows"][2] // 2, 1035), (train_boxes["cf"][0] + 90, train_boxes["cf"][1] - 22)], dashed=True)
    draw_arrow(draw, svg, [(train_boxes["cf"][0] + train_boxes["cf"][2], train_boxes["cf"][1] + 85), (train_boxes["cf_loss"][0] - 18, train_boxes["cf_loss"][1] + 85)], dashed=True)
    draw_arrow(draw, svg, [(train_boxes["cf_loss"][0] + train_boxes["cf_loss"][2], train_boxes["cf_loss"][1] + 85), (1050, train_boxes["cf_loss"][1] + 85), (1050, 1535), (train_boxes["total"][0] - 35, 1535), (train_boxes["total"][0] - 35, train_boxes["total"][1] + 85), (train_boxes["total"][0] - 18, train_boxes["total"][1] + 85)], dashed=True)
    draw_arrow(draw, svg, [(boxes["assign"][0] + boxes["assign"][2] // 2, boxes["assign"][1] + boxes["assign"][3]), (boxes["assign"][0] + boxes["assign"][2] // 2, 1035), (train_boxes["rel"][0] + 105, train_boxes["rel"][1] - 22)], dashed=True)
    draw_arrow(draw, svg, [(boxes["score"][0] + boxes["score"][2] // 2, boxes["score"][1] + boxes["score"][3]), (boxes["score"][0] + boxes["score"][2] // 2, 1035), (train_boxes["mem_loss"][0] + 140, train_boxes["mem_loss"][1] - 22)], dashed=True)
    draw_arrow(draw, svg, [(train_boxes["rel"][0] + train_boxes["rel"][2], train_boxes["rel"][1] + 110), (train_boxes["memory"][0] - 18, train_boxes["memory"][1] + 110)], dashed=True)
    draw_arrow(draw, svg, [(train_boxes["memory"][0] + train_boxes["memory"][2], train_boxes["memory"][1] + 110), (train_boxes["mem_loss"][0] - 18, train_boxes["mem_loss"][1] + 85)], dashed=True)
    draw_arrow(draw, svg, [(train_boxes["mem_loss"][0] + train_boxes["mem_loss"][2], train_boxes["mem_loss"][1] + 85), (train_boxes["total"][0] - 20, train_boxes["total"][1] + 110)], dashed=True)
    draw_arrow(draw, svg, [(boxes["fusion"][0] + boxes["fusion"][2] // 2, boxes["fusion"][1] + boxes["fusion"][3]), (boxes["fusion"][0] + boxes["fusion"][2] // 2, 1035), (train_boxes["total"][0] + 120, train_boxes["total"][1] - 22)], dashed=True)

    # Small code alignment note.
    note = "Code alignment: memory is a training/update side branch; prediction uses prototype score, transition gain/path score, and uncertainty penalty."
    draw.text((80, 1715), note, font=FONT_SMALL, fill=PALETTE["muted"])
    svg.text(80, 1740, note, size=18, color=PALETTE["muted"])

    img.save(PNG_PATH, dpi=(300, 300))
    svg.save(SVG_PATH)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
