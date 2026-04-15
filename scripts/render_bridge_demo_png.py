from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "bridge_navigation_demo.png"

W, H = 1200, 520
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
TEXT = (34, 34, 34)
SUBTLE = (118, 107, 95)


def _font(size: int):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FT = _font(25)
FC = _font(17)

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
        int(panel_x + 8 + x * 0.73),
        int(84 + 18 + y * 0.92),
    )


def circle(draw: ImageDraw.ImageDraw, xy, r, fill, outline=None, width=1):
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=width)


def panel(draw: ImageDraw.ImageDraw, x0: int, title: str, bridged: bool):
    draw.rounded_rectangle((x0, 84, x0 + 520, 444), radius=26, fill=PANEL, outline=BORDER, width=2)
    draw.text((x0 + 30, 54), title, fill=TEXT, font=FT)

    for a, b in EDGES:
        draw.line((p(x0, a), p(x0, b)), fill=EDGE, width=3)

    if bridged:
        draw.line((p(x0, "G"), p(x0, "S")), fill=BRIDGE, width=5)
        draw.line((p(x0, "S"), p(x0, "J")), fill=BRIDGE, width=5)

    for key in ["A","B","C","D","E","F","G","H","I","J","K","L","M"]:
        if key == "A":
            circle(draw, p(x0, key), 12, START, START_STROKE, 3)
        else:
            circle(draw, p(x0, key), 10, NODE, NODE_STROKE, 2)

    if bridged:
        circle(draw, p(x0, "S"), 12, BRIDGE, BRIDGE_STROKE, 3)

    circle(draw, p(x0, "L"), 18, None, TARGET, 5)
    circle(draw, p(x0, "L"), 5, TARGET)

    if bridged:
        path = ["A", "B", "D", "E", "G", "S", "J", "L"]
    else:
        path = ["A", "B", "D", "E", "G"]
    draw.line([p(x0, k) for k in path], fill=PATH, width=8, joint="curve")
    for key in path[1:]:
        circle(draw, p(x0, key), 7, PATH)

    if bridged:
        circle(draw, p(x0, "L"), 28, None, TARGET, 5)
    else:
        circle(draw, p(x0, "G"), 22, None, PATH, 5)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    panel(draw, 40, "Connected graph, greedy path gets trapped", False)
    panel(draw, 640, "Same graph plus one bridge node", True)
    draw.text((406, 490), "orange = routing-only Steiner bridge", fill=SUBTLE, font=FC)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
