"""
═══════════════════════════════════════════════════════════════
  OCR Image Text Editor Pro — Backend API (v2.2 — Memory Optimized)
  ─────────────────────────────────────────────────────────────
  Optimized for Render.com free tier (512MB RAM):
    • Lazy model loading (loads on first request, not startup)
    • Aggressive garbage collection after each request
    • Image downscaling before OCR (saves ~200MB)
    • Single-worker optimized
    • Memory monitoring endpoint
═══════════════════════════════════════════════════════════════
"""

# ─── SSL Fix ──────────────────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ─── Force low memory mode ────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

import io
import gc
import sys
import time
import hashlib
import logging
import base64
import resource
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import (
    Image, ImageDraw, ImageFont, ImageFilter,
    ImageEnhance, ImageOps,
)

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
    PORT: int = int(os.getenv("PORT", 5000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    MAX_IMAGE_MB: int = int(os.getenv("MAX_IMAGE_MB", 10))
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", 20_000_000))
    OCR_LANGUAGES: List[str] = field(
        default_factory=lambda: os.getenv("OCR_LANGS", "en").split(",")
    )
    OCR_GPU: bool = False
    OUTPUT_FORMAT: str = "PNG"
    JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", 85))
    FONT_DIR: str = os.getenv(
        "FONT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"),
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    BG_SAMPLE_CLUSTERS: int = 3
    BG_SAMPLE_MARGIN: int = 8
    MIN_FONT_SIZE: int = 6
    MAX_FONT_SIZE: int = 600
    # ── Memory optimization settings ──
    OCR_MAX_DIMENSION: int = int(os.getenv("OCR_MAX_DIMENSION", 1500))
    ENABLE_GC: bool = True


CFG = Config()

logging.basicConfig(
    level=getattr(logging, CFG.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ocr-editor")


# ═══════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CFG.MAX_IMAGE_MB * 1024 * 1024

CORS(app, resources={r"/api/*": {"origins": CFG.ALLOWED_ORIGINS}})

if HAS_COMPRESS:
    Compress(app)


def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return round(usage.ru_maxrss / 1024, 1)  # Linux: KB → MB
    except Exception:
        return 0


@app.before_request
def _start_timer():
    g.start_time = time.perf_counter()


@app.after_request
def _after(response):
    if hasattr(g, "start_time"):
        elapsed = (time.perf_counter() - g.start_time) * 1000
        mem = get_memory_mb()
        log.info(
            f"{request.method} {request.path} → "
            f"{response.status_code} ({elapsed:.0f}ms, {mem}MB)"
        )
    # Aggressive garbage collection after every request
    if CFG.ENABLE_GC:
        gc.collect()
    response.headers["X-Server"] = "OCR-Editor/2.2"
    response.headers["X-Memory-MB"] = str(get_memory_mb())
    return response


# ═══════════════════════════════════════════════════════════
# FONT MANAGER (lightweight)
# ═══════════════════════════════════════════════════════════

class FontManager:
    SYSTEM_DIRS = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "C:/Windows/Fonts",
    ]
    FONT_MAP = {
        "Arial": ["arial"], "Helvetica": ["helvetica", "arial"],
        "Verdana": ["verdana"], "Georgia": ["georgia"],
        "Times New Roman": ["times"], "Courier New": ["courier"],
        "Impact": ["impact"], "Tahoma": ["tahoma"],
        "Trebuchet MS": ["trebuc"], "Palatino": ["palatino"],
        "Garamond": ["garamond"], "Lucida Console": ["lucida"],
        "Comic Sans MS": ["comic"],
    }

    def __init__(self):
        self._cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self._discovered: Dict[str, str] = {}
        self._discover()

    def _discover(self):
        for d in self.SYSTEM_DIRS:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.ttf', '.otf', '.ttc')):
                        key = f.lower().replace("-", "").replace("_", "")
                        self._discovered[key] = os.path.join(root, f)
        log.info(f"🔤 Fonts: {len(self._discovered)} files found")

    def get_font(self, family="Arial", size=20, bold=False, italic=False):
        ck = f"{family}|{size}|{bold}|{italic}"
        if ck in self._cache:
            return self._cache[ck]

        base = family.lower().replace(" ", "")
        variants = []
        if bold:
            variants += [f"{base}bold", f"{base}bd", f"{base}b"]
        if italic:
            variants += [f"{base}italic", f"{base}it"]
        variants.append(base)

        aliases = self.FONT_MAP.get(family, [family.lower()])
        for a in aliases:
            a = a.lower().replace(" ", "")
            if bold: variants.append(f"{a}bold")
            variants.append(a)

        font = None
        for v in variants:
            for key, path in self._discovered.items():
                if v in key:
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        continue
            if font:
                break

        if not font:
            for fb in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]:
                try:
                    font = ImageFont.truetype(fb, size)
                    break
                except Exception:
                    continue

        if not font:
            font = ImageFont.load_default()

        # Limit cache size to save memory
        if len(self._cache) > 50:
            self._cache.clear()
        self._cache[ck] = font
        return font

    def list_available(self):
        avail = set()
        for family in self.FONT_MAP:
            base = family.lower().replace(" ", "")
            for key in self._discovered:
                if base in key:
                    avail.add(family)
                    break
        return sorted(avail)


FONTS = FontManager()


# ═══════════════════════════════════════════════════════════
# COLOR UTILITIES
# ═══════════════════════════════════════════════════════════

def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3: h = "".join(c * 2 for c in h)
    if len(h) != 6: return (0, 0, 0)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

def hex_to_rgba(h, opacity=1.0):
    r, g, b = hex_to_rgb(h)
    return (r, g, b, max(0, min(255, int(opacity * 255))))


# ═══════════════════════════════════════════════════════════
# BACKGROUND SAMPLING (simplified for memory)
# ═══════════════════════════════════════════════════════════

def sample_bg_color(img, bbox, margin=8):
    """Sample dominant border color — memory efficient version."""
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(img.width - 1, max(xs) + margin)
    y2 = min(img.height - 1, max(ys) + margin)

    pixels = []
    step = max(1, (x2 - x1) // 30)

    for x in range(x1, x2 + 1, step):
        for yo in [y1, y2]:
            if 0 <= x < img.width and 0 <= yo < img.height:
                px = img.getpixel((x, yo))
                pixels.append(px[:3] if len(px) >= 3 else (px[0],) * 3)

    for y in range(y1, y2 + 1, step):
        for xo in [x1, x2]:
            if 0 <= xo < img.width and 0 <= y < img.height:
                px = img.getpixel((xo, y))
                pixels.append(px[:3] if len(px) >= 3 else (px[0],) * 3)

    if not pixels:
        return "#ffffff"

    most_common = Counter(pixels).most_common(1)[0][0]
    return rgb_to_hex(*most_common)


# ═══════════════════════════════════════════════════════════
# TEXT STYLE & RENDERER
# ═══════════════════════════════════════════════════════════

@dataclass
class TextStyle:
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
    def from_dict(cls, d):
        return cls(
            text=str(d.get("newText", d.get("text", ""))),
            font_family=str(d.get("fontFamily", "Arial")),
            font_size=max(6, min(600, int(d.get("fontSize", 20)))),
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
    def __init__(self, font_mgr):
        self.fonts = font_mgr

    def render_region(self, base_img, bbox, style, scale=1.0):
        x1 = int(min(p[0] for p in bbox) * scale)
        y1 = int(min(p[1] for p in bbox) * scale)
        x2 = int(max(p[0] for p in bbox) * scale)
        y2 = int(max(p[1] for p in bbox) * scale)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return base_img

        fs = max(6, int(style.font_size * scale))
        br = int(style.border_radius * scale)
        bdr_w = int(style.border_width * scale)
        pl, pr = int(style.pad_left * scale), int(style.pad_right * scale)
        pt, pb = int(style.pad_top * scale), int(style.pad_bottom * scale)
        ls = int(style.letter_spacing * scale)

        region = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        rdraw = ImageDraw.Draw(region)

        bg_rgba = hex_to_rgba(style.bg_color, style.bg_opacity)
        try:
            rdraw.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=br, fill=bg_rgba)
        except AttributeError:
            rdraw.rectangle([0, 0, bw - 1, bh - 1], fill=bg_rgba)

        if bdr_w > 0:
            bdr_c = hex_to_rgba(style.border_color)
            try:
                rdraw.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=br, outline=bdr_c, width=bdr_w)
            except AttributeError:
                rdraw.rectangle([0, 0, bw - 1, bh - 1], outline=bdr_c, width=bdr_w)

        font = self.fonts.get_font(style.font_family, fs, style.bold, style.italic)
        taw, tah = bw - pl - pr, bh - pt - pb

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
                else:
                    lx = pl + (taw - lw) / 2.0

                ty = cy + fs * 0.12
                if ls > 0:
                    px = lx
                    for ch in line:
                        rdraw.text((px, ty), ch, fill=tc, font=font)
                        bb = rdraw.textbbox((0, 0), ch, font=font)
                        px += (bb[2] - bb[0]) + ls
                else:
                    rdraw.text((lx, ty), line, fill=tc, font=font)

                if style.underline:
                    ul_y = int(ty + fs * 0.92)
                    rdraw.line([(int(lx), ul_y), (int(lx + lw), ul_y)],
                              fill=tc, width=max(1, int(fs * 0.06)))

        if base_img.mode != "RGBA":
            base_img = base_img.convert("RGBA")

        px_x = max(0, x1)
        px_y = max(0, y1)
        cx = max(0, -x1)
        cy_off = max(0, -y1)
        pw = min(bw - cx, base_img.width - px_x)
        ph = min(bh - cy_off, base_img.height - px_y)

        if pw > 0 and ph > 0:
            cropped = region.crop((cx, cy_off, cx + pw, cy_off + ph))
            base_img.paste(cropped, (px_x, px_y), cropped)

        del region, rdraw
        return base_img

    def _measure(self, draw, text, font, ls=0):
        if not text: return 0
        if ls <= 0:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0]
        w = 0
        for i, c in enumerate(text):
            bb = draw.textbbox((0, 0), c, font=font)
            w += (bb[2] - bb[0])
            if i < len(text) - 1: w += ls
        return w

    def _wrap(self, draw, text, font, max_w, ls=0):
        if not text: return [""]
        out = []
        for para in text.split("\n"):
            if not para: out.append(""); continue
            words = para.split(" ")
            cur = ""
            for word in words:
                test = f"{cur} {word}" if cur else word
                if self._measure(draw, test, font, ls) > max_w and cur:
                    out.append(cur)
                    cur = word
                else:
                    cur = test
            out.append(cur)
        return out


RENDERER = TextRenderer(FONTS)


# ═══════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ═══════════════════════════════════════════════════════════

def validate_image(img):
    w, h = img.size
    if w * h > CFG.MAX_IMAGE_PIXELS:
        return False, f"Image too large: {w}×{h} ({w*h:,}px, max {CFG.MAX_IMAGE_PIXELS:,})"
    if w < 10 or h < 10:
        return False, f"Image too small: {w}×{h}"
    return True, ""


def downscale_for_ocr(img, max_dim):
    """Downscale image for OCR to save memory. Returns (scaled_img, scale_factor)."""
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img, 1.0
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    log.info(f"📐 Downscaled {w}×{h} → {new_w}×{new_h} for OCR (factor={scale:.2f})")
    return scaled, scale


def encode_image(img, fmt="PNG", quality=85):
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        if img.mode in ("RGBA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                bg.paste(img, mask=img.split()[3])
            else:
                bg.paste(img)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    result = base64.b64encode(buf.getvalue()).decode()
    del buf
    return result


# ═══════════════════════════════════════════════════════════
# OCR ENGINE — LAZY LOADING (critical for memory)
# ═══════════════════════════════════════════════════════════

class OCREngine:
    """
    Lazy-loading OCR engine.
    Model is NOT loaded at startup — only on first request.
    This saves ~300MB during startup and prevents OOM on deploy.
    """

    def __init__(self):
        self.reader = None
        self.loaded_langs = []
        self.load_error = None
        self._easyocr = None

    @property
    def is_ready(self):
        return self.reader is not None

    def _ensure_loaded(self, languages=None):
        langs = languages or CFG.OCR_LANGUAGES
        langs = [l.strip() for l in langs if l.strip()]

        if self.reader is not None and self.loaded_langs == langs:
            return

        # Lazy import — don't import easyocr until needed
        if self._easyocr is None:
            log.info("⏳ Importing easyocr (lazy)…")
            t0 = time.perf_counter()
            import easyocr
            self._easyocr = easyocr
            log.info(f"📦 easyocr imported in {time.perf_counter()-t0:.1f}s")

        log.info(f"⏳ Loading OCR model for: {langs} (mem: {get_memory_mb()}MB)")
        t0 = time.perf_counter()
        self.load_error = None

        try:
            self.reader = self._easyocr.Reader(
                langs, gpu=False,
                model_storage_directory='/tmp/easyocr_models',
                download_enabled=True
            )
            self.loaded_langs = langs
            gc.collect()
            log.info(
                f"✅ OCR loaded in {time.perf_counter()-t0:.1f}s "
                f"(mem: {get_memory_mb()}MB)"
            )
        except Exception as e1:
            log.warning(f"⚠️  Attempt 1 failed: {e1}")
            try:
                import urllib.request
                orig = urllib.request.urlopen
                def _patched(*a, **kw):
                    kw.setdefault("context", ssl._create_unverified_context())
                    return orig(*a, **kw)
                urllib.request.urlopen = _patched
                try:
                    self.reader = self._easyocr.Reader(
                        langs, gpu=False,
                        model_storage_directory='/tmp/easyocr_models'
                    )
                    self.loaded_langs = langs
                    gc.collect()
                    log.info(f"✅ OCR loaded (retry) in {time.perf_counter()-t0:.1f}s")
                finally:
                    urllib.request.urlopen = orig
            except Exception as e2:
                self.load_error = str(e2)
                raise RuntimeError(f"OCR load failed: {e2}") from e2

    def detect(self, img, languages=None, min_confidence=0.1):
        self._ensure_loaded(languages)

        # Downscale for OCR to save memory
        ocr_img, scale = downscale_for_ocr(img.convert("RGB"), CFG.OCR_MAX_DIMENSION)
        img_np = np.array(ocr_img)

        # Free the scaled image immediately
        del ocr_img
        gc.collect()

        t0 = time.perf_counter()
        results = self.reader.readtext(img_np, min_size=10)
        elapsed = time.perf_counter() - t0

        # Free numpy array
        del img_np
        gc.collect()

        log.info(f"🔍 OCR: {len(results)} results in {elapsed:.2f}s (mem: {get_memory_mb()}MB)")

        # Scale coordinates back to original image size
        inv_scale = 1.0 / scale if scale != 1.0 else 1.0

        detections = []
        for idx, (bbox, text, confidence) in enumerate(results):
            if confidence < min_confidence:
                continue

            # Scale bbox back to original coordinates
            scaled_bbox = [[int(p[0] * inv_scale), int(p[1] * inv_scale)] for p in bbox]

            bg = sample_bg_color(img, scaled_bbox)
            ys = [p[1] for p in scaled_bbox]
            est_h = max(ys) - min(ys)

            detections.append({
                "id": idx,
                "bbox": scaled_bbox,
                "text": text.strip(),
                "confidence": round(float(confidence), 4),
                "bgColor": bg,
                "fontSize": max(6, int(est_h * 0.75)),
                "estHeight": int(est_h),
            })

        detections.sort(key=lambda d: (
            min(p[1] for p in d["bbox"]),
            min(p[0] for p in d["bbox"])
        ))
        for i, d in enumerate(detections):
            d["id"] = i

        return detections


OCR = OCREngine()

# ── DO NOT load model at startup — lazy load on first request ──
log.info(f"🔋 Startup memory: {get_memory_mb()}MB (model will load on first request)")


# ═══════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "OCR Image Text Editor Pro API",
        "version": "2.2.0-mem-optimized",
        "status": "running",
        "ocr_ready": OCR.is_ready,
        "memory_mb": get_memory_mb(),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "ocr_loaded": OCR.is_ready,
        "ocr_error": OCR.load_error,
        "memory_mb": get_memory_mb(),
        "fonts_available": len(FONTS.list_available()),
        "max_image_mb": CFG.MAX_IMAGE_MB,
        "ocr_max_dimension": CFG.OCR_MAX_DIMENSION,
    })


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "version": "2.2.0",
        "python": sys.version.split()[0],
        "ocr_loaded": OCR.is_ready,
        "ocr_error": OCR.load_error,
        "memory_mb": get_memory_mb(),
        "fonts": FONTS.list_available(),
        "config": {
            "max_image_mb": CFG.MAX_IMAGE_MB,
            "max_pixels": CFG.MAX_IMAGE_PIXELS,
            "ocr_max_dimension": CFG.OCR_MAX_DIMENSION,
        },
    })


@app.route("/api/fonts", methods=["GET"])
def list_fonts():
    return jsonify({"fonts": FONTS.list_available()})


@app.route("/api/detect", methods=["POST"])
def detect_text():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": "Empty file"}), 400

        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception:
            return jsonify({"error": "Invalid image"}), 400

        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        img = img.convert("RGB")
        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        log.info(f"📸 Detect: {img.width}×{img.height} ({len(img_bytes)//1024}KB, mem: {get_memory_mb()}MB)")

        # Run OCR
        detections = OCR.detect(img, min_confidence=0.1)

        # Encode image (compress if large to save bandwidth)
        if img.width * img.height > 4_000_000:
            img_b64 = encode_image(img, fmt="JPEG", quality=85)
        else:
            img_b64 = encode_image(img)

        # Free original bytes
        del img_bytes
        gc.collect()

        return jsonify({
            "success": True,
            "detections": detections,
            "image": img_b64,
            "width": img.width,
            "height": img.height,
            "detection_count": len(detections),
            "memory_mb": get_memory_mb(),
        })

    except RuntimeError as e:
        return jsonify({"error": f"OCR engine: {e}"}), 503
    except Exception as e:
        log.exception("❌ /api/detect error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/edit", methods=["POST"])
def edit_image():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        img_b64 = data.get("image")
        edits = data.get("edits", [])
        scale = max(0.1, min(4.0, float(data.get("scale", 1.0))))

        if not img_b64:
            return jsonify({"error": "No image"}), 400
        if not edits:
            return jsonify({"error": "No edits"}), 400

        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGBA")
        except Exception:
            return jsonify({"error": "Invalid image data"}), 400

        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        if scale != 1.0:
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.Resampling.LANCZOS,
            )

        errors = []
        applied = 0
        for i, edit in enumerate(edits):
            try:
                bbox = edit.get("bbox")
                if not bbox or len(bbox) < 4:
                    errors.append({"index": i, "error": "Invalid bbox"})
                    continue
                style = TextStyle.from_dict(edit)
                img = RENDERER.render_region(img, bbox, style, scale=scale)
                applied += 1
            except Exception as e:
                errors.append({"index": i, "error": str(e)})

        result_b64 = encode_image(img)
        del img
        gc.collect()

        resp = {"success": True, "image": result_b64, "applied": applied, "total": len(edits)}
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
            return jsonify({"error": "No image"}), 400

        img = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")
        valid, msg = validate_image(img)
        if not valid:
            return jsonify({"error": msg}), 400

        b = max(0.5, min(2.0, float(data.get("brightness", 1.05))))
        c = max(0.5, min(2.0, float(data.get("contrast", 1.10))))
        s = max(0.5, min(3.0, float(data.get("sharpness", 1.15))))
        sat = max(0.0, min(2.0, float(data.get("saturation", 1.08))))

        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        img = ImageEnhance.Sharpness(img).enhance(s)
        img = ImageEnhance.Color(img).enhance(sat)

        result = encode_image(img)
        del img
        gc.collect()

        return jsonify({"success": True, "image": result})
    except Exception as e:
        log.exception("❌ /api/enhance error")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {CFG.MAX_IMAGE_MB}MB"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found. GET / for endpoints"}), 404

@app.errorhandler(500)
def server_error(e):
    gc.collect()
    return jsonify({"error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("═" * 50)
    log.info("  OCR Text Editor — v2.2 Memory Optimized")
    log.info("═" * 50)
    log.info(f"  Memory:     {get_memory_mb()} MB")
    log.info(f"  OCR:        lazy (loads on first request)")
    log.info(f"  Max image:  {CFG.OCR_MAX_DIMENSION}px for OCR")
    log.info(f"  Port:       {CFG.PORT}")
    log.info("═" * 50)
    app.run(host="0.0.0.0", port=CFG.PORT, debug=CFG.DEBUG)