from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "docs" / "figures" / "bridge_navigation_demo.png"
OUT_GIF = ROOT / "docs" / "figures" / "bridge_navigation_demo.gif"

W, H = 1200, 520
SCALE = 2
BG = (246, 242, 235)
PANEL = (255, 253, 249)
BORDER = (216, 207, 191)
EDGE = (199, 190, 178)
NODE = (63, 110, 168)
NODE_STROKE = (39, 72, 109)
START = (198, 58, 85)
START_STROKE = (139, 33, 55)
TARGET = (31, 140, 97)
PATH = (204, 75, 55)
BRIDGE = (234, 151, 66)
BRIDGE_STROKE = (151, 92, 31)
SUBTLE = (118, 107, 95)
TEXT = (34, 34, 34)


def _font(size: int):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FT = _font(25 * SCALE)
FC = _font(17 * SCALE)

POS = {
    "A": (90, 240),
    "B": (140, 170),
    "C": (145, 315),
    "D": (220, 205),
    "E": (285, 135),
    "F": (290, 300),
    "G": (360, 165),
    "H": (380, 320),
    "I": (455, 255),
    "J": (520, 185),
    "K": (535, 335),
    "L": (590, 215),
    "M": (610, 305),
    "S": (475, 165),
}

EDGES = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("C", "D"),
    ("D", "E"),
    ("D", "F"),
    ("E", "G"),
    ("F", "H"),
    ("H", "I"),
    ("I", "J"),
    ("I", "K"),
    ("J", "L"),
    ("K", "M"),
    ("L", "M"),
]


def p(panel_x: int, key: str):
    x, y = POS[key]
    return (
        int((panel_x + 8 + x * 0.73) * SCALE),
        int((84 + 18 + y * 0.92) * SCALE),
    )


def circle(draw: ImageDraw.ImageDraw, xy, r, fill, outline=None, width=1):
    x, y = xy
    draw.ellipse(
        (x - r * SCALE, y - r * SCALE, x + r * SCALE, y + r * SCALE),
        fill=fill,
        outline=outline,
        width=max(1, int(width * SCALE)),
    )


def _interp(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
    )


def draw_partial_path(draw: ImageDraw.ImageDraw, panel_x: int, keys: list[str], progress: float):
    points = [p(panel_x, key) for key in keys]
    total_segments = len(points) - 1
    completed = max(0, min(total_segments, int(progress)))
    partial = max(0.0, min(1.0, progress - completed))

    if completed > 0:
        draw.line(
            points[: completed + 1],
            fill=PATH,
            width=8 * SCALE,
            joint="curve",
        )
    if completed < total_segments and partial > 0:
        seg = [points[completed], _interp(points[completed], points[completed + 1], partial)]
        draw.line(seg, fill=PATH, width=8 * SCALE, joint="curve")

    visible_visits = completed + (1 if partial > 0 else 0)
    for idx, key in enumerate(keys[1:]):
        if idx < visible_visits:
            circle(draw, p(panel_x, key), 7, PATH)


def panel(
    draw: ImageDraw.ImageDraw,
    x0: int,
    title: str,
    bridged: bool,
    *,
    left_progress: float | None = None,
    right_progress: float | None = None,
    show_deadend: bool = False,
    show_success: bool = False,
    show_bridge: bool = False,
):
    draw.rounded_rectangle(
        (x0 * SCALE, 84 * SCALE, (x0 + 520) * SCALE, 444 * SCALE),
        radius=26 * SCALE,
        fill=PANEL,
        outline=BORDER,
        width=2 * SCALE,
    )
    draw.text((int((x0 + 30) * SCALE), 54 * SCALE), title, fill=TEXT, font=FT)

    for a, b in EDGES:
        draw.line((p(x0, a), p(x0, b)), fill=EDGE, width=3 * SCALE)

    if bridged and show_bridge:
        draw.line((p(x0, "G"), p(x0, "S")), fill=BRIDGE, width=5 * SCALE)
        draw.line((p(x0, "S"), p(x0, "J")), fill=BRIDGE, width=5 * SCALE)

    for key in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]:
        if key == "A":
            circle(draw, p(x0, key), 12, START, START_STROKE, 3)
        else:
            circle(draw, p(x0, key), 10, NODE, NODE_STROKE, 2)

    if bridged and show_bridge:
        circle(draw, p(x0, "S"), 12, BRIDGE, BRIDGE_STROKE, 3)

    circle(draw, p(x0, "L"), 18, None, TARGET, 5)
    circle(draw, p(x0, "L"), 5, TARGET)

    if not bridged and left_progress is not None:
        draw_partial_path(draw, x0, ["A", "B", "D", "E", "G"], left_progress)
    if bridged and right_progress is not None:
        draw_partial_path(draw, x0, ["A", "B", "D", "E", "G", "S", "J", "L"], right_progress)

    if show_success:
        circle(draw, p(x0, "L"), 28, None, TARGET, 5)
    if show_deadend:
        circle(draw, p(x0, "G"), 22, None, PATH, 5)


def _base_frame():
    img = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    draw = ImageDraw.Draw(img)
    draw.text((406 * SCALE, 490 * SCALE), "orange = routing-only Steiner bridge", fill=SUBTLE, font=FC)
    return img, draw


def render_png():
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img, draw = _base_frame()
    panel(
        draw,
        40,
        "Connected graph, greedy path gets trapped",
        False,
        left_progress=4.0,
        show_deadend=True,
    )
    panel(
        draw,
        640,
        "Same graph plus one bridge node",
        True,
        right_progress=7.0,
        show_bridge=True,
        show_success=True,
    )
    img = img.resize((W, H), resample=Image.Resampling.LANCZOS)
    img.save(OUT_PNG)


def render_gif():
    frame_specs = [
        {"left": 0.6, "right": 0.0, "bridge": False, "deadend": False, "success": False, "duration": 180},
        {"left": 1.4, "right": 0.0, "bridge": False, "deadend": False, "success": False, "duration": 180},
        {"left": 2.2, "right": 0.0, "bridge": False, "deadend": False, "success": False, "duration": 180},
        {"left": 3.2, "right": 0.0, "bridge": False, "deadend": False, "success": False, "duration": 200},
        {"left": 4.0, "right": 0.0, "bridge": False, "deadend": True, "success": False, "duration": 420},
        {"left": 4.0, "right": 0.0, "bridge": True, "deadend": True, "success": False, "duration": 220},
        {"left": 4.0, "right": 1.2, "bridge": True, "deadend": False, "success": False, "duration": 180},
        {"left": 4.0, "right": 2.4, "bridge": True, "deadend": False, "success": False, "duration": 180},
        {"left": 4.0, "right": 3.6, "bridge": True, "deadend": False, "success": False, "duration": 180},
        {"left": 4.0, "right": 4.8, "bridge": True, "deadend": False, "success": False, "duration": 180},
        {"left": 4.0, "right": 6.0, "bridge": True, "deadend": False, "success": False, "duration": 180},
        {"left": 4.0, "right": 7.0, "bridge": True, "deadend": False, "success": True, "duration": 480},
    ]

    frames: list[Image.Image] = []
    durations: list[int] = []
    for spec in frame_specs:
        img, draw = _base_frame()
        panel(
            draw,
            40,
            "Connected graph, greedy path gets trapped",
            False,
            left_progress=float(spec["left"]),
            show_deadend=bool(spec["deadend"]),
        )
        panel(
            draw,
            640,
            "Same graph plus one bridge node",
            True,
            right_progress=float(spec["right"]),
            show_bridge=bool(spec["bridge"]),
            show_success=bool(spec["success"]),
        )
        frame = img.resize((W, H), resample=Image.Resampling.LANCZOS)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=255))
        durations.append(int(spec["duration"]))

    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main():
    render_png()
    render_gif()
    print(OUT_PNG)
    print(OUT_GIF)


if __name__ == "__main__":
    main()
