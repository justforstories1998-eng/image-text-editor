"""
═══════════════════════════════════════════════════════════════
  OCR Image Text Editor Pro — Backend API (v2.1 — patched)
  ─────────────────────────────────────────────────────────────
  Changelog v2.1:
    • FIXED:  Missing `import easyocr`
    • FIXED:  Retry logic — no longer shadows urllib permanently
    • FIXED:  Startup no longer crashes if model download fails
    • ADDED:  Lazy-load fallback — model loads on first request
    • ADDED:  /api/status endpoint shows exactly what's loaded
    • CLEANED: Removed unused imports
═══════════════════════════════════════════════════════════════
"""

# ─── SSL Fix (MUST be before any network imports) ─────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# ──────────────────────────────────────────────────────────

import io
import sys
import time
import hashlib
import logging
import base64
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

# ─── EasyOCR — guarded import ────────────────────────────
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as _imp_err:
    EASYOCR_AVAILABLE = False
    logging.warning(f"⚠️  easyocr not installed: {_imp_err}")

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import (
    Image, ImageDraw, ImageFont, ImageFilter,
    ImageEnhance, ImageOps,
)

# ─── Optional: compression ───────────────────────────────
try:
    from flask_compress import Compress
    HAS_COMPRESS = True
except ImportError:
    HAS_COMPRESS = False


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class Config:
    """Central config — override via environment variables."""
    PORT: int = int(os.getenv("PORT", 5000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    MAX_IMAGE_MB: int = int(os.getenv("MAX_IMAGE_MB", 25))
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", 50_000_000))
    OCR_LANGUAGES: List[str] = field(
        default_factory=lambda: os.getenv("OCR_LANGS", "en").split(",")
    )
    OCR_GPU: bool = os.getenv("OCR_GPU", "false").lower() == "true"
    OUTPUT_FORMAT: str = os.getenv("OUTPUT_FORMAT", "PNG")
    JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", 92))
    FONT_DIR: str = os.getenv(
        "FONT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"),
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    BG_SAMPLE_CLUSTERS: int = 3
    BG_SAMPLE_MARGIN: int = 8
    DEFAULT_FONT_SIZE: int = 20
    MIN_FONT_SIZE: int = 6
    MAX_FONT_SIZE: int = 600


CFG = Config()


# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, CFG.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ocr-editor")


# ═══════════════════════════════════════════════════════════
# FLASK APP SETUP
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CFG.MAX_IMAGE_MB * 1024 * 1024

CORS(app, resources={r"/api/*": {"origins": CFG.ALLOWED_ORIGINS}})

if HAS_COMPRESS:
    Compress(app)
    log.info("✅ Response compression enabled")


@app.before_request
def _start_timer():
    g.start_time = time.perf_counter()


@app.after_request
def _log_request(response):
    if hasattr(g, "start_time"):
        elapsed = (time.perf_counter() - g.start_time) * 1000
        log.info(
            f"{request.method} {request.path} → "
            f"{response.status_code} ({elapsed:.0f}ms)"
        )
    response.headers["X-Server"] = "OCR-Editor-Pro/2.1"
    return response


# ═══════════════════════════════════════════════════════════
# FONT MANAGER
# ═══════════════════════════════════════════════════════════

class FontManager:
    """Font discovery, caching, and fallback chains."""

    FONT_MAP = {
        "Arial":            ["arial", "Arial"],
        "Helvetica":        ["helvetica", "Helvetica", "arial"],
        "Verdana":          ["verdana", "Verdana"],
        "Tahoma":           ["tahoma", "Tahoma"],
        "Trebuchet MS":     ["trebuc", "Trebuchet"],
        "Georgia":          ["georgia", "Georgia"],
        "Times New Roman":  ["times", "Times"],
        "Palatino":         ["palatino", "Palatino"],
        "Garamond":         ["garamond", "Garamond"],
        "Courier New":      ["courier", "Courier"],
        "Lucida Console":   ["lucida", "Lucida"],
        "Impact":           ["impact", "Impact"],
        "Comic Sans MS":    ["comic", "Comic"],
    }

    SYSTEM_DIRS = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/System/Library/Fonts",
        "/Library/Fonts",
        "C:/Windows/Fonts",
    ]

    def __init__(self, custom_dir: str = ""):
        self._font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self._discovered: Dict[str, str] = {}
        self._discover_fonts(custom_dir)
        log.info(f"🔤 FontManager: discovered {len(self._discovered)} font files")

    def _discover_fonts(self, custom_dir: str):
        dirs = list(self.SYSTEM_DIRS)
        if custom_dir and os.path.isdir(custom_dir):
            dirs.insert(0, custom_dir)

        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
                    if ext in ("ttf", "otf", "ttc"):
                        name_key = f.lower().replace("-", "").replace("_", "")
                        self._discovered[name_key] = os.path.join(root, f)

    def _find_path(
        self, family: str, bold: bool = False, italic: bool = False
    ) -> Optional[str]:
        base = family.lower().replace(" ", "")
        variants = []

        if bold and italic:
            variants += [f"{base}bolditalic", f"{base}bi", f"{base}bold"]
        elif bold:
            variants += [f"{base}bold", f"{base}bd", f"{base}b", f"{base}700"]
        elif italic:
            variants += [f"{base}italic", f"{base}it", f"{base}i", f"{base}oblique"]

        variants.append(base)

        aliases = self.FONT_MAP.get(family, [family])
        for alias in aliases:
            a = alias.lower().replace(" ", "")
            if bold:
                variants += [f"{a}bold", f"{a}bd"]
            if italic:
                variants += [f"{a}italic", f"{a}it"]
            variants.append(a)

        for v in variants:
            for key, path in self._discovered.items():
                if v in key:
                    return path
        return None

    def get_font(
        self,
        family: str = "Arial",
        size: int = 20,
        bold: bool = False,
        italic: bool = False,
    ) -> ImageFont.FreeTypeFont:
        cache_key = f"{family}|{size}|{bold}|{italic}"
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        path = self._find_path(family, bold, italic)
        font = None

        if path:
            try:
                font = ImageFont.truetype(path, size)
            except Exception as e:
                log.warning(f"⚠️  Font load failed: {path} → {e}")

        if font is None:
            fallbacks = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
                if bold
                else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "C:/Windows/Fonts/arialbd.ttf"
                if bold
                else "C:/Windows/Fonts/arial.ttf",
            ]
            for fb in fallbacks:
                try:
                    font = ImageFont.truetype(fb, size)
                    break
                except (OSError, IOError):
                    continue

        if font is None:
            log.warning(f"⚠️  All fonts failed for '{family}' — bitmap fallback")
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def list_available(self) -> List[str]:
        available = set()
        for family in self.FONT_MAP:
            if self._find_path(family, False, False):
                available.add(family)
        return sorted(available)


FONTS = FontManager(CFG.FONT_DIR)


# ═══════════════════════════════════════════════════════════
# COLOR UTILITIES
# ═══════════════════════════════════════════════════════════

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return (0, 0, 0)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"


def hex_to_rgba(hex_color: str, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(hex_color)
    a = max(0, min(255, int(opacity * 255)))
    return (r, g, b, a)


# ═══════════════════════════════════════════════════════════
# BACKGROUND COLOR SAMPLING (K-MEANS)
# ═══════════════════════════════════════════════════════════

def sample_bg_color_kmeans(
    img: Image.Image,
    bbox: List[List[int]],
    margin: int = None,
    k: int = None,
) -> str:
    """
    Dominant background color via K-means clustering
    on border pixels around the bounding box.
    """
    margin = margin or CFG.BG_SAMPLE_MARGIN
    k = k or CFG.BG_SAMPLE_CLUSTERS

    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(img.width - 1, max(xs) + margin)
    y2 = min(img.height - 1, max(ys) + margin)

    pixels = []
    step_x = max(1, (x2 - x1) // 60)
    step_y = max(1, (y2 - y1) // 60)

    for x in range(x1, x2 + 1, step_x):
        for y_off in [y1, y1 + 1, y2 - 1, y2]:
            if 0 <= x < img.width and 0 <= y_off < img.height:
                px = img.getpixel((x, y_off))
                pixels.append(px[:3] if len(px) >= 3 else (px[0],) * 3)

    for y in range(y1, y2 + 1, step_y):
        for x_off in [x1, x1 + 1, x2 - 1, x2]:
            if 0 <= x_off < img.width and 0 <= y < img.height:
                px = img.getpixel((x_off, y))
                pixels.append(px[:3] if len(px) >= 3 else (px[0],) * 3)

    if not pixels:
        return "#ffffff"

    if len(pixels) < k * 2:
        most_common = Counter(pixels).most_common(1)[0][0]
        return rgb_to_hex(*most_common)

    # ── K-means (no sklearn needed) ───────────────────────
    arr = np.array(pixels, dtype=np.float32)

    rng = np.random.default_rng(42)          # deterministic
    centroids = [arr[rng.integers(len(arr))]]
    for _ in range(1, k):
        dists = np.min(
            [np.sum((arr - c) ** 2, axis=1) for c in centroids], axis=0
        )
        probs = dists / (dists.sum() + 1e-8)
        idx = rng.choice(len(arr), p=probs)
        centroids.append(arr[idx])
    centroids = np.array(centroids)

    for _ in range(15):
        dists = np.array(
            [np.sum((arr - c) ** 2, axis=1) for c in centroids]
        )
        labels = np.argmin(dists, axis=0)
        new_centroids = []
        for ci in range(k):
            members = arr[labels == ci]
            new_centroids.append(
                members.mean(axis=0) if len(members) > 0 else centroids[ci]
            )
        centroids = np.array(new_centroids)

    labels_final = np.argmin(
        [np.sum((arr - c) ** 2, axis=1) for c in centroids], axis=0
    )
    counts = np.bincount(labels_final, minlength=k)
    dominant = centroids[np.argmax(counts)]

    return rgb_to_hex(int(dominant[0]), int(dominant[1]), int(dominant[2]))


# ═══════════════════════════════════════════════════════════
# TEXT STYLE  &  RENDERER
# ═══════════════════════════════════════════════════════════

@dataclass
class TextStyle:
    """Every styling property for one text region."""
    text: str = ""
    font_family: str = "Arial"
    font_size: int = 20
    bold: bool = False
    italic: bool = False
    underline: bool = False
    font_color: str = "#000000"
    bg_color: str = "#ffffff"
    bg_opacity: float = 1.0
    alignment: str = "center"
    letter_spacing: int = 0
    line_height: int = 130
    border_width: int = 0
    border_color: str = "#e63946"
    border_radius: int = 0
    pad_top: int = 4
    pad_bottom: int = 4
    pad_left: int = 6
    pad_right: int = 6

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextStyle":
        return cls(
            text=str(d.get("newText", d.get("text", ""))),
            font_family=str(d.get("fontFamily", "Arial")),
            font_size=max(CFG.MIN_FONT_SIZE, min(CFG.MAX_FONT_SIZE, int(d.get("fontSize", 20)))),
            bold=bool(d.get("bold", False)),
            italic=bool(d.get("italic", False)),
            underline=bool(d.get("underline", False)),
            font_color=str(d.get("fontColor", "#000000")),
            bg_color=str(d.get("bgColor", "#ffffff")),
            bg_opacity=max(0.0, min(1.0, float(d.get("bgOpacity", 1.0)))),
            alignment=str(d.get("alignment", "center")),
            letter_spacing=max(0, int(d.get("letterSpacing", 0))),
            line_height=max(50, min(400, int(d.get("lineHeight", 130)))),
            border_width=max(0, min(50, int(d.get("borderWidth", 0)))),
            border_color=str(d.get("borderColor", "#e63946")),
            border_radius=max(0, min(200, int(d.get("borderRadius", 0)))),
            pad_top=max(0, int(d.get("padTop", 4))),
            pad_bottom=max(0, int(d.get("padBottom", 4))),
            pad_left=max(0, int(d.get("padLeft", 6))),
            pad_right=max(0, int(d.get("padRight", 6))),
        )


class TextRenderer:
    """
    Full text renderer matching the frontend canvas engine:
    wrapping, letter-spacing, line-height, all alignments,
    BG opacity, borders with radius, underline, padding.
    """

    def __init__(self, font_mgr: FontManager):
        self.fonts = font_mgr

    def render_region(
        self,
        base_img: Image.Image,
        bbox: List[List[int]],
        style: TextStyle,
        scale: float = 1.0,
    ) -> Image.Image:
        x1 = int(min(p[0] for p in bbox) * scale)
        y1 = int(min(p[1] for p in bbox) * scale)
        x2 = int(max(p[0] for p in bbox) * scale)
        y2 = int(max(p[1] for p in bbox) * scale)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return base_img

        fs = max(CFG.MIN_FONT_SIZE, int(style.font_size * scale))
        br = int(style.border_radius * scale)
        bdr_w = int(style.border_width * scale)
        pl = int(style.pad_left * scale)
        pr = int(style.pad_right * scale)
        pt = int(style.pad_top * scale)
        pb = int(style.pad_bottom * scale)
        ls = int(style.letter_spacing * scale)

        # ── RGBA overlay ─────────────────────────────────
        region = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        rdraw = ImageDraw.Draw(region)

        bg_rgba = hex_to_rgba(style.bg_color, style.bg_opacity)
        self._draw_rounded_rect(rdraw, 0, 0, bw, bh, br, fill=bg_rgba)

        if bdr_w > 0:
            bdr_color = hex_to_rgba(style.border_color, 1.0)
            self._draw_rounded_rect(
                rdraw, 0, 0, bw, bh, br, outline=bdr_color, width=bdr_w
            )

        # ── Text ──────────────────────────────────────────
        font = self.fonts.get_font(style.font_family, fs, style.bold, style.italic)
        taw = bw - pl - pr
        tah = bh - pt - pb

        if taw > 0 and tah > 0 and style.text:
            lines = self._wrap(rdraw, style.text, font, taw, ls)
            lh = fs * (style.line_height / 100.0)
            total_h = len(lines) * lh
            start_y = max(pt, pt + (tah - total_h) / 2.0)
            tc = hex_to_rgb(style.font_color) + (255,)

            for li, line in enumerate(lines):
                cy = start_y + li * lh
                if cy + lh < 0 or cy > bh:
                    continue
                lw = self._measure(rdraw, line, font, ls)

                if style.alignment == "left":
                    lx = pl
                elif style.alignment == "right":
                    lx = pl + taw - lw
                elif style.alignment == "justify" and li < len(lines) - 1 and len(lines) > 1:
                    self._draw_justified(
                        rdraw, line, pl, cy + fs * 0.12,
                        taw, font, tc, ls, fs, style.underline,
                    )
                    continue
                else:
                    lx = pl + (taw - lw) / 2.0

                ty = cy + fs * 0.12
                if ls > 0:
                    px = lx
                    for ch in line:
                        rdraw.text((px, ty), ch, fill=tc, font=font)
                        px += self._cw(rdraw, ch, font) + ls
                else:
                    rdraw.text((lx, ty), line, fill=tc, font=font)

                if style.underline:
                    ul_y = int(ty + fs * 0.92)
                    rdraw.line(
                        [(int(lx), ul_y), (int(lx + lw), ul_y)],
                        fill=tc, width=max(1, int(fs * 0.06)),
                    )

        # ── Composite ────────────────────────────────────
        if base_img.mode != "RGBA":
            base_img = base_img.convert("RGBA")

        paste_x = max(0, x1)
        paste_y = max(0, y1)
        crop_x = max(0, -x1)
        crop_y = max(0, -y1)
        pw = min(bw - crop_x, base_img.width - paste_x)
        ph = min(bh - crop_y, base_img.height - paste_y)

        if pw > 0 and ph > 0:
            cropped = region.crop((crop_x, crop_y, crop_x + pw, crop_y + ph))
            base_img.paste(cropped, (paste_x, paste_y), cropped)

        return base_img

    # ── Helpers ───────────────────────────────────────────

    def _cw(self, draw, char, font) -> float:
        bb = draw.textbbox((0, 0), char, font=font)
        return bb[2] - bb[0]

    def _measure(self, draw, text, font, ls=0) -> float:
        if not text:
            return 0
        if ls <= 0:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0]
        w = 0
        for i, c in enumerate(text):
            w += self._cw(draw, c, font)
            if i < len(text) - 1:
                w += ls
        return w

    def _wrap(self, draw, text, font, max_w, ls=0) -> List[str]:
        if not text:
            return [""]
        out = []
        for para in text.split("\n"):
            if not para:
                out.append("")
                continue
            words = para.split(" ")
            cur = ""
            for word in words:
                test = f"{cur} {word}" if cur else word
                if self._measure(draw, test, font, ls) > max_w and cur:
                    out.append(cur)
                    cur = word
                    if self._measure(draw, word, font, ls) > max_w:
                        broken = self._break_word(draw, word, font, max_w, ls)
                        if len(broken) > 1:
                            out.extend(broken[:-1])
                            cur = broken[-1]
                else:
                    cur = test
            out.append(cur)
        return out

    def _break_word(self, draw, word, font, max_w, ls) -> List[str]:
        lines, cur = [], ""
        for c in word:
            if self._measure(draw, cur + c, font, ls) > max_w and cur:
                lines.append(cur)
                cur = c
            else:
                cur += c
        if cur:
            lines.append(cur)
        return lines

    def _draw_justified(self, draw, text, x, y, max_w, font, color, ls, fs, underline):
        words = text.split()
        if len(words) <= 1:
            draw.text((x, y), text, fill=color, font=font)
            return
        total_w = sum(self._measure(draw, w, font, ls) for w in words)
        gap = (max_w - total_w) / (len(words) - 1)
        px = x
        for i, w in enumerate(words):
            if ls > 0:
                for c in w:
                    draw.text((px, y), c, fill=color, font=font)
                    px += self._cw(draw, c, font) + ls
                px -= ls
            else:
                draw.text((px, y), w, fill=color, font=font)
                px += self._measure(draw, w, font, ls)
            if i < len(words) - 1:
                px += gap
        if underline:
            draw.line(
                [(int(x), int(y + fs * 0.80)), (int(x + max_w), int(y + fs * 0.80))],
                fill=color, width=max(1, int(fs * 0.06)),
            )

    def _draw_rounded_rect(self, draw, x1, y1, x2, y2, r, fill=None, outline=None, width=1):
        try:
            draw.rounded_rectangle(
                [x1, y1, x2 - 1, y2 - 1],
                radius=r, fill=fill, outline=outline, width=width,
            )
        except AttributeError:
            # Pillow < 10 fallback
            r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
            if fill:
                draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
                draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)
                draw.pieslice([x1, y1, x1 + 2*r, y1 + 2*r], 180, 270, fill=fill)
                draw.pieslice([x2 - 2*r, y1, x2, y1 + 2*r], 270, 360, fill=fill)
                draw.pieslice([x1, y2 - 2*r, x1 + 2*r, y2], 90, 180, fill=fill)
                draw.pieslice([x2 - 2*r, y2 - 2*r, x2, y2], 0, 90, fill=fill)
            if outline:
                draw.arc([x1, y1, x1 + 2*r, y1 + 2*r], 180, 270, fill=outline, width=width)
                draw.arc([x2 - 2*r, y1, x2, y1 + 2*r], 270, 360, fill=outline, width=width)
                draw.arc([x1, y2 - 2*r, x1 + 2*r, y2], 90, 180, fill=outline, width=width)
                draw.arc([x2 - 2*r, y2 - 2*r, x2, y2], 0, 90, fill=outline, width=width)
                draw.line([(x1 + r, y1), (x2 - r, y1)], fill=outline, width=width)
                draw.line([(x1 + r, y2), (x2 - r, y2)], fill=outline, width=width)
                draw.line([(x1, y1 + r), (x1, y2 - r)], fill=outline, width=width)
                draw.line([(x2, y1 + r), (x2, y2 - r)], fill=outline, width=width)


RENDERER = TextRenderer(FONTS)


# ═══════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════

def validate_image(img: Image.Image) -> Tuple[bool, str]:
    w, h = img.size
    px = w * h
    if px > CFG.MAX_IMAGE_PIXELS:
        return False, f"Image too large: {w}×{h} = {px:,}px (max {CFG.MAX_IMAGE_PIXELS:,})"
    if w < 10 or h < 10:
        return False, f"Image too small: {w}×{h}"
    return True, ""


def encode_image(img: Image.Image, fmt: str = None, quality: int = None) -> str:
    fmt = fmt or CFG.OUTPUT_FORMAT
    quality = quality or CFG.JPEG_QUALITY
    buf = io.BytesIO()

    if fmt.upper() == "JPEG":
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        if img.mode not in ("RGBA", "RGB"):
            img = img.convert("RGB")
        img.save(buf, format="PNG", optimize=True)

    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════════════════════
# OCR ENGINE  (lazy-loading, safe retry)
# ═══════════════════════════════════════════════════════════

class OCREngine:
    """
    Wraps EasyOCR with:
      • Guarded import check
      • Lazy loading (loads on first request, not on import)
      • Safe SSL retry that doesn't permanently patch urllib
      • Multi-language hot-reload
    """

    def __init__(self):
        self.reader = None
        self.loaded_langs: List[str] = []
        self.load_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self.reader is not None

    def _ensure_loaded(self, languages: List[str] = None):
        langs = languages or CFG.OCR_LANGUAGES
        langs = [l.strip() for l in langs if l.strip()]

        if not EASYOCR_AVAILABLE:
            self.load_error = "easyocr package is not installed"
            raise RuntimeError(self.load_error)

        if self.reader is not None and self.loaded_langs == langs:
            return  # already loaded with same languages

        log.info(f"⏳ Loading EasyOCR model for: {langs}")
        t0 = time.perf_counter()
        self.load_error = None

        # ── Attempt 1: normal load ────────────────────────
        try:
            self.reader = easyocr.Reader(langs, gpu=CFG.OCR_GPU)
            self.loaded_langs = langs
            log.info(f"✅ EasyOCR loaded in {time.perf_counter() - t0:.1f}s")
            return
        except Exception as e1:
            log.warning(f"⚠️  EasyOCR load attempt 1 failed: {e1}")

        # ── Attempt 2: with SSL context disabled ──────────
        try:
            import urllib.request
            original_urlopen = urllib.request.urlopen

            def _ssl_urlopen(*args, **kwargs):
                kwargs.setdefault("context", ssl._create_unverified_context())
                return original_urlopen(*args, **kwargs)

            urllib.request.urlopen = _ssl_urlopen
            try:
                self.reader = easyocr.Reader(langs, gpu=CFG.OCR_GPU)
                self.loaded_langs = langs
                log.info(
                    f"✅ EasyOCR loaded (SSL bypass) in "
                    f"{time.perf_counter() - t0:.1f}s"
                )
            finally:
                # ALWAYS restore — never leave patched
                urllib.request.urlopen = original_urlopen

        except Exception as e2:
            self.load_error = str(e2)
            log.error(f"❌ EasyOCR load attempt 2 failed: {e2}")
            raise RuntimeError(
                f"Could not load OCR model: {e2}"
            ) from e2

    def detect(
        self,
        img: Image.Image,
        languages: List[str] = None,
        min_confidence: float = 0.1,
        paragraph: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run OCR on a PIL Image. Lazy-loads model if needed."""

        # ── Lazy load ─────────────────────────────────────
        self._ensure_loaded(languages)

        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)

        t0 = time.perf_counter()
        results = self.reader.readtext(
            img_np,
            paragraph=paragraph,
            min_size=10,
            text_threshold=0.6,
            low_text=0.3,
        )
        elapsed = time.perf_counter() - t0
        log.info(f"🔍 OCR: {len(results)} detections in {elapsed:.2f}s")

        detections = []
        for idx, (bbox, text, confidence) in enumerate(results):
            if confidence < min_confidence:
                continue

            bg = sample_bg_color_kmeans(img_rgb, bbox)
            font_color = self._estimate_font_color(img_rgb, bbox, bg)

            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            est_h = max(ys) - min(ys)

            detections.append({
                "id": idx,
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                "text": text.strip(),
                "confidence": round(float(confidence), 4),
                "bgColor": bg,
                "fontColor": font_color,
                "fontSize": max(CFG.MIN_FONT_SIZE, int(est_h * 0.75)),
                "estWidth": int(max(xs) - min(xs)),
                "estHeight": int(est_h),
            })

        # Sort in reading order (top→bottom, left→right)
        detections.sort(
            key=lambda d: (min(p[1] for p in d["bbox"]), min(p[0] for p in d["bbox"]))
        )
        for i, d in enumerate(detections):
            d["id"] = i

        return detections

    def _estimate_font_color(
        self, img: Image.Image, bbox: List, bg_hex: str
    ) -> str:
        try:
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)

            pixels = []
            sx = max(1, w // 20)
            sy = max(1, h // 10)
            for dx in range(-w // 4, w // 4, sx):
                for dy in range(-h // 4, h // 4, sy):
                    px_x = max(0, min(img.width - 1, cx + dx))
                    px_y = max(0, min(img.height - 1, cy + dy))
                    px = img.getpixel((px_x, px_y))
                    pixels.append(px[:3] if len(px) >= 3 else (px[0],) * 3)

            if not pixels:
                return "#000000"

            bg_rgb = np.array(hex_to_rgb(bg_hex), dtype=np.float32)
            arr = np.array(pixels, dtype=np.float32)
            dists = np.sqrt(np.sum((arr - bg_rgb) ** 2, axis=1))

            top_n = max(1, int(len(dists) * 0.3))
            top_idx = np.argsort(dists)[-top_n:]

            if len(top_idx) == 0:
                return "#000000"

            avg = arr[top_idx].mean(axis=0)
            return rgb_to_hex(int(avg[0]), int(avg[1]), int(avg[2]))

        except Exception:
            return "#000000"


OCR = OCREngine()

# ── Eager load (non-fatal) ────────────────────────────────
try:
    OCR._ensure_loaded()
except Exception as e:
    log.warning(
        f"⚠️  OCR not loaded at startup (will retry on first request): {e}"
    )


# ═══════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "OCR Image Text Editor Pro API",
        "version": "2.1.0",
        "status": "running",
        "ocr_ready": OCR.is_ready,
        "endpoints": {
            "POST /api/detect":    "Upload image → detect text regions",
            "POST /api/edit":      "Apply text edits → download result",
            "POST /api/enhance":   "One-tap image enhancement",
            "POST /api/remove-bg": "Basic background removal",
            "GET  /api/health":    "Health check",
            "GET  /api/fonts":     "List available fonts",
            "GET  /api/status":    "Detailed system status",
        },
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if OCR.is_ready else "degraded",
        "ocr_loaded": OCR.is_ready,
        "ocr_error": OCR.load_error,
        "ocr_languages": OCR.loaded_langs,
        "easyocr_installed": EASYOCR_AVAILABLE,
        "fonts_available": len(FONTS.list_available()),
        "max_image_mb": CFG.MAX_IMAGE_MB,
        "gpu_enabled": CFG.OCR_GPU,
    })


@app.route("/api/status", methods=["GET"])
def status():
    """Detailed system status for debugging."""
    return jsonify({
        "service": "OCR Image Text Editor Pro",
        "version": "2.1.0",
        "python": sys.version,
        "easyocr_installed": EASYOCR_AVAILABLE,
        "easyocr_version": getattr(easyocr, "__version__", "unknown") if EASYOCR_AVAILABLE else None,
        "ocr_loaded": OCR.is_ready,
        "ocr_languages": OCR.loaded_langs,
        "ocr_error": OCR.load_error,
        "gpu": CFG.OCR_GPU,
        "fonts_discovered": len(FONTS._discovered),
        "fonts_available": FONTS.list_available(),
        "config": {
            "max_image_mb": CFG.MAX_IMAGE_MB,
            "max_pixels": CFG.MAX_IMAGE_PIXELS,
            "output_format": CFG.OUTPUT_FORMAT,
            "jpeg_quality": CFG.JPEG_QUALITY,
            "bg_sample_clusters": CFG.BG_SAMPLE_CLUSTERS,
        },
        "compression": HAS_COMPRESS,
    })


@app.route("/api/fonts", methods=["GET"])
def list_fonts():
    return jsonify({
        "fonts": FONTS.list_available(),
        "total": len(FONTS.list_available()),
    })


@app.route("/api/detect", methods=["POST"])
def detect_text():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file in request. Send as multipart with key 'image'."}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty file"}), 400

        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception:
            return jsonify({"error": "Cannot decode image. Supported: PNG, JPEG, WebP, BMP, TIFF."}), 400

        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        img = img.convert("RGB")

        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        log.info(
            f"📸 Detect: {file.filename} — "
            f"{img.width}×{img.height} ({len(img_bytes)/1024:.0f} KB)"
        )

        langs = request.form.get("languages", "en").split(",")
        min_conf = float(request.form.get("min_confidence", 0.1))
        paragraph = request.form.get("paragraph", "false").lower() == "true"

        detections = OCR.detect(
            img, languages=langs,
            min_confidence=min_conf, paragraph=paragraph,
        )

        img_b64 = encode_image(img)

        return jsonify({
            "success": True,
            "detections": detections,
            "image": img_b64,
            "width": img.width,
            "height": img.height,
            "hash": hashlib.sha256(img_bytes).hexdigest()[:16],
            "ocr_languages": langs,
            "detection_count": len(detections),
        })

    except RuntimeError as e:
        log.error(f"❌ OCR runtime error: {e}")
        return jsonify({"error": f"OCR engine error: {e}"}), 503
    except Exception as e:
        log.exception("❌ /api/detect error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/edit", methods=["POST"])
def edit_image():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body. Send Content-Type: application/json."}), 400

        img_b64 = data.get("image")
        edits = data.get("edits", [])
        out_format = data.get("outputFormat", CFG.OUTPUT_FORMAT)
        quality = data.get("quality", CFG.JPEG_QUALITY)
        scale = max(0.1, min(4.0, float(data.get("scale", 1.0))))

        if not img_b64:
            return jsonify({"error": "Missing 'image' field (base64)"}), 400
        if not edits:
            return jsonify({"error": "Missing 'edits' array"}), 400

        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGBA")
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        if scale != 1.0:
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.Resampling.LANCZOS,
            )

        log.info(
            f"✏️  Edit: {len(edits)} regions, "
            f"{img.width}×{img.height}, scale={scale}"
        )

        errors = []
        applied = 0
        for i, edit in enumerate(edits):
            try:
                bbox = edit.get("bbox")
                if not bbox or len(bbox) < 4:
                    errors.append({"index": i, "error": "Invalid bbox (need 4 points)"})
                    continue
                style = TextStyle.from_dict(edit)
                img = RENDERER.render_region(img, bbox, style, scale=scale)
                applied += 1
            except Exception as e:
                log.warning(f"⚠️  Edit #{i} failed: {e}")
                errors.append({"index": i, "error": str(e)})

        result_b64 = encode_image(img, fmt=out_format, quality=quality)

        resp = {
            "success": True,
            "image": result_b64,
            "width": img.width,
            "height": img.height,
            "applied": applied,
            "total": len(edits),
        }
        if errors:
            resp["warnings"] = errors
        return jsonify(resp)

    except Exception as e:
        log.exception("❌ /api/edit error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/enhance", methods=["POST"])
def enhance_image():
    try:
        data = request.json
        if not data or not data.get("image"):
            return jsonify({"error": "No image data"}), 400

        img = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")

        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        brightness = max(0.5, min(2.0, float(data.get("brightness", 1.05))))
        contrast   = max(0.5, min(2.0, float(data.get("contrast", 1.10))))
        sharpness  = max(0.5, min(3.0, float(data.get("sharpness", 1.15))))
        saturation = max(0.0, min(2.0, float(data.get("saturation", 1.08))))
        denoise    = bool(data.get("denoise", False))

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        img = ImageEnhance.Color(img).enhance(saturation)

        if denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))

        return jsonify({
            "success": True,
            "image": encode_image(img),
            "width": img.width,
            "height": img.height,
            "enhancements": {
                "brightness": brightness, "contrast": contrast,
                "sharpness": sharpness, "saturation": saturation,
                "denoise": denoise,
            },
        })

    except Exception as e:
        log.exception("❌ /api/enhance error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/remove-bg", methods=["POST"])
def remove_background():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file"}), 400

        img = Image.open(request.files["image"].stream).convert("RGBA")

        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        threshold = max(0, min(255, int(request.form.get("threshold", 240))))

        arr = np.array(img)
        brightness = arr[:, :, :3].mean(axis=2)
        alpha = np.where(brightness > threshold, 0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha, mode="L").filter(
            ImageFilter.GaussianBlur(radius=1)
        )
        arr[:, :, 3] = np.array(alpha_img)
        result = Image.fromarray(arr, mode="RGBA")

        return jsonify({
            "success": True,
            "image": encode_image(result, fmt="PNG"),
            "width": result.width,
            "height": result.height,
            "note": "Basic threshold removal. Use a segmentation model for production.",
        })

    except Exception as e:
        log.exception("❌ /api/remove-bg error")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {CFG.MAX_IMAGE_MB} MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. GET / for available endpoints."}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("═" * 56)
    log.info("  OCR Image Text Editor Pro — API v2.1")
    log.info("═" * 56)
    log.info(f"  Port:          {CFG.PORT}")
    log.info(f"  OCR Languages: {CFG.OCR_LANGUAGES}")
    log.info(f"  OCR Ready:     {OCR.is_ready}")
    if OCR.load_error:
        log.info(f"  OCR Error:     {OCR.load_error}")
        log.info(f"  → Will retry on first /api/detect request")
    log.info(f"  GPU:           {CFG.OCR_GPU}")
    log.info(f"  Max Image:     {CFG.MAX_IMAGE_MB} MB")
    log.info(f"  Fonts Found:   {len(FONTS.list_available())}")
    log.info(f"  Debug:         {CFG.DEBUG}")
    log.info("═" * 56)

    app.run(host="0.0.0.0", port=CFG.PORT, debug=CFG.DEBUG)