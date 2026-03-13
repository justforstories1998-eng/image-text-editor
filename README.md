# 🔤 OCR Image Text Editor Pro

> Upload → Detect Text → Edit, Style & Reposition → Download

An AI-powered image text editor that detects text regions using EasyOCR
and lets you edit, restyle, and reposition every detected text block
with a full visual canvas editor.

![Version](https://img.shields.io/badge/version-2.1.0-red)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

### Core Text Editing
- 🔍 **OCR Detection** — EasyOCR-powered text detection with confidence scores
- ✏️ **Full Text Styling** — Font family, size, color, bold, italic, underline
- 📐 **Auto-Fit Engine** — Binary-search algorithm to fit text perfectly in boxes
- 🎨 **Background & Border** — Opacity, border radius, border color/width
- ↔️ **Spacing Controls** — Letter spacing, line height, padding (T/B/L/R)
- 📏 **Alignment** — Left, center, right, justify with word-wrapping

### Object Editing Suite
- 🎯 **Object Detection** — Region-based object identification
- 🖱️ **Drag & Reposition** — Move objects with smart snap guides
- 🔄 **Replace Objects** — Search and swap objects from the internet
- 🎨 **Recolor** — Hue, saturation, brightness adjustments
- 📋 **Duplicate & Delete** — Clone or remove any object

### Background Editing
- 🛡️ **Background Preservation** — Background stays intact by default
- 🌫️ **Selective Blur** — Blur behind selected objects or full background
- 🎨 **Color Overlay** — Tint background with adjustable opacity
- 🖼️ **Background Replace** — Search or upload replacement backgrounds
- 📐 **8 Gradient Presets** — Quick background styles

### In-App Image Search
- 🔍 **Integrated Search** — Unsplash / Picsum image search
- 🏷️ **Category Filters** — Objects, Nature, Textures, Backgrounds, etc.
- 🔄 **Context-Aware** — Pre-fills search based on selected object
- ⬇️ **One-Click Use** — Replace object or set as background instantly

### Power Features
- ↩️ **Undo / Redo** — 50-state history with visual timeline
- ✨ **One-Tap Enhance** — AI auto-adjusts brightness, contrast, color
- 📏 **Smart Snap Guides** — Alignment to center, thirds, other objects
- 🖱️ **Right-Click Menu** — 10 contextual actions
- 👁️ **Before/After** — Hold `C` to compare with original
- ⌨️ **12+ Keyboard Shortcuts** — See table below

---

## 🏗️ Architecture

```
┌─────────────┐     HTTP/JSON     ┌──────────────┐
│   Frontend   │ ◄──────────────► │   Backend    │
│  (Netlify)   │                  │  (Render)    │
│              │                  │              │
│  HTML/CSS/JS │  POST /api/detect│  Flask       │
│  Canvas API  │  POST /api/edit  │  EasyOCR     │
│  Editor UI   │  POST /api/enhance  Pillow     │
│              │  GET  /api/health│  NumPy       │
└─────────────┘                  └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/image-text-editor.git
cd image-text-editor
```

### 2. Set up the backend
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
python app.py
```

The API will start at `http://localhost:5000`

### 3. Open the frontend
```bash
cd frontend
# Simply open index.html in your browser
# Or use a local server:
python -m http.server 8080
```

Then visit `http://localhost:8080`

---

## 🌐 Deployment

### Backend → Render.com (Free)
1. Push to GitHub
2. Go to [render.com](https://render.com)
3. New → Web Service → Connect your repo
4. Root directory: `backend`
5. It auto-detects `render.yaml`
6. Deploy!

### Frontend → Netlify (Free)
1. Go to [netlify.com](https://netlify.com)
2. New site → Import from Git
3. Base directory: `frontend`
4. Build command: _(leave empty)_
5. Publish directory: `frontend`
6. Deploy!

### ⚠️ Update API URL
After deploying the backend, update `index.html`:
```javascript
// Change this line:
const API = "http://localhost:5000";
// To your Render URL:
const API = "https://your-app.onrender.com";
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` / `2` / `3` | Switch mode (Text / Object / BG) |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+S` | Download PNG |
| `Ctrl+F` | Open image search |
| `H` | Toggle history panel |
| `E` | Toggle enhance panel |
| `C` (hold) | Before/after compare |
| `D` | Duplicate object |
| `Delete` | Remove selected |
| `Space` | Toggle bounding boxes |
| `Arrows` | Nudge 1px (Shift = 10px) |
| `+` / `-` | Zoom in/out |
| `Esc` | Deselect / close modals |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/fonts` | List available fonts |
| `GET` | `/api/status` | Detailed system status |
| `POST` | `/api/detect` | Upload image → OCR detection |
| `POST` | `/api/edit` | Apply text edits → rendered image |
| `POST` | `/api/enhance` | One-tap image enhancement |
| `POST` | `/api/remove-bg` | Basic background removal |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **OCR Engine** | EasyOCR |
| **Backend** | Flask, Pillow, NumPy |
| **Frontend** | Vanilla JS, Canvas API |
| **Text Rendering** | Custom engine (wrap, fit, align, spacing) |
| **BG Sampling** | K-means clustering |
| **Hosting** | Render (API) + Netlify (UI) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using EasyOCR + Flask + Canvas
</p>