"""Procedural Artist Engine v1.6 - brief-driven colour-field painter with a physical stroke model.

What changed from v1.5
  * No reference pixels reach the canvas. A reference (optional) is reduced to a small JSON
    "brief" of ~40 numbers (canvas box, field box, palettes, border width). Without a reference
    the brief is sampled from a colour-field grammar, so the engine can plan on its own.
  * Every mark is a real stroke: a bristle profile, a paint load that depletes along the stroke,
    frayed bristle ends, brush pickup of underlying colour, and dry-brush transfer that only
    catches the peaks of the surface tooth. Strokes leave their own ridges in the tooth, so
    later dry passes catch earlier brushwork.
  * Height is stored in millimetres with an explicit encoding scale chosen from the actual peak,
    and normals use the physical pixel pitch. The preview exaggeration is a separate setting.
  * Portable CLI with no hard-coded paths.

Usage
  python engine_v16.py --out out_dir --seed 44315 --reference IMG_0443.jpeg
  python engine_v16.py --out out_dir --seed 7                      # autonomous plan
Requires numpy, scipy, opencv-python, pillow.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

VERSION = "1.7-print"
REF_W_PX = 850.0
# Substrates: thread pitch (mm), weave relief after priming (mm), slub (thread irregularity),
# and the paint thickness (mm) over which the weave is levelled out (e-folding).
SUBSTRATES = {
    "fine_linen":  dict(pitch_mm=0.45, amp_mm=0.010, slub=0.12, level_mm=0.030),
    "linen":       dict(pitch_mm=0.70, amp_mm=0.014, slub=0.20, level_mm=0.040),
    "cotton_duck": dict(pitch_mm=0.90, amp_mm=0.018, slub=0.08, level_mm=0.050),
    "jute":        dict(pitch_mm=1.80, amp_mm=0.035, slub=0.35, level_mm=0.080),
    "panel":       dict(pitch_mm=0.70, amp_mm=0.000, slub=0.00, level_mm=0.040),
}
BIG = 4_000_000  # arrays above this many pixels use cropped / chunked evaluation
MATERIALS = {"gesso": 0, "imprimatura": 1, "border": 2, "ground": 3, "scumble": 4,
             "field_under": 5, "field": 6, "dry_fringe": 7, "edge_accent": 8, "glaze": 9}


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(rng, h, w, cell):
    """Smooth zero-mean, unit-std noise with feature size ~cell pixels."""
    gh, gw = max(2, int(math.ceil(h / cell)) + 2), max(2, int(math.ceil(w / cell)) + 2)
    a = rng.normal(size=(gh, gw)).astype(np.float32)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
    return (a - a.mean()) / (a.std() + 1e-6)


class Noise:
    """value_noise that can be evaluated on a crop without building the full canvas.
    Consumes the RNG exactly like value_noise, so small canvases render identically."""

    def __init__(self, rng, h, w, cell):
        self.h, self.w = h, w
        gh, gw = max(2, int(math.ceil(h / cell)) + 2), max(2, int(math.ceil(w / cell)) + 2)
        self.a = rng.normal(size=(gh, gw)).astype(np.float32)
        self._full = None
        if h * w <= BIG:
            full = cv2.resize(self.a, (w, h), interpolation=cv2.INTER_CUBIC)
            self.mu, self.sd = float(full.mean()), float(full.std()) + 1e-6
            self._full = (full - self.mu) / self.sd
        else:
            sw = 2048; sh = max(2, int(round(h * sw / w)))
            small = cv2.resize(self.a, (sw, sh), interpolation=cv2.INTER_CUBIC)
            self.mu, self.sd = float(small.mean()), float(small.std()) + 1e-6

    def crop(self, ya, yb, xa, xb):
        if self._full is not None:
            return self._full[ya:yb, xa:xb]
        gh, gw = self.a.shape
        mx = ((np.arange(xa, xb, dtype=np.float32) + 0.5) * (gw / self.w) - 0.5)
        my = ((np.arange(ya, yb, dtype=np.float32) + 0.5) * (gh / self.h) - 0.5)
        MX, MY = np.meshgrid(mx, my)
        out = cv2.remap(self.a, MX, MY, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return (out - self.mu) / self.sd

    def full(self):
        return self.crop(0, self.h, 0, self.w)

    def at(self, y, x):
        y = int(np.clip(y, 0, self.h - 1)); x = int(np.clip(x, 0, self.w - 1))
        return float(self.crop(y, y + 1, x, x + 1)[0, 0])


def row_chunks(h, w, budget=BIG):
    step = max(1, budget // max(1, w))
    for y in range(0, h, step):
        yield y, min(h, y + step)


def jitter_color(rng, c, amount):
    return np.clip(np.asarray(c, np.float32) + rng.normal(0, amount, 3), 0, 255)


def lerp_palette(stops, t):
    """stops: list of RGB; t in [0,1] (scalar)."""
    stops = np.asarray(stops, np.float32)
    pos = np.linspace(0, 1, len(stops))
    return np.array([np.interp(t, pos, stops[:, k]) for k in range(3)], np.float32)


# --------------------------------------------------------------------------------------------
# Brief: from a reference (summary statistics only) or sampled from a grammar
# --------------------------------------------------------------------------------------------
def _kmeans(pixels, k, seed=0):
    pixels = np.float32(pixels)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(seed)
    _, lab, cen = cv2.kmeans(pixels, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lab.ravel(), minlength=k)
    order = np.argsort(-counts)
    return cen[order], counts[order] / counts.sum()


def detect_canvas_box(rgb):
    """Find the painting inside the photo using the dark frame line; fallback to whole image."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = g < 70
    h, w = g.shape
    rows = np.where(dark.sum(1) > 0.25 * w)[0]
    cols = np.where(dark.sum(0) > 0.25 * h)[0]
    if len(rows) < 2 or len(cols) < 2:
        return [0, 0, w, h]
    inset = 6
    return [int(cols.min() + inset), int(rows.min() + inset), int(cols.max() - inset), int(rows.max() - inset)]


def wall_gain(rgb, box, target=236.0):
    """Per-channel gain that maps the (assumed white, neutral) wall around the canvas to `target`.
    Corrects photo exposure and colour cast before any colour is summarised."""
    x0, y0, x1, y1 = box; h, w = rgb.shape[:2]; m = 40
    ring = np.zeros((h, w), bool)
    ring[max(0, y0 - 3 * m): max(0, y0 - m), :] = True; ring[min(h, y1 + m): min(h, y1 + 3 * m), :] = True
    ring[:, max(0, x0 - 3 * m): max(0, x0 - m)] = True; ring[:, min(w, x1 + m): min(w, x1 + 3 * m)] = True
    px = rgb[ring].astype(np.float32)
    if len(px) < 500:
        return np.ones(3, np.float32), None
    lum = px.mean(1); sat = px.max(1) - px.min(1)
    sel = (lum > np.percentile(lum, 40)) & (sat < 25)
    if sel.sum() < 200:
        return np.ones(3, np.float32), None
    wall = np.median(px[sel], axis=0)
    return np.clip(target / wall, 0.85, 1.35).astype(np.float32), wall


def brief_from_reference(path, canvas_box=None, calibrate=True):
    rgb = np.array(Image.open(path).convert("RGB"))
    box = canvas_box or detect_canvas_box(rgb)
    x0, y0, x1, y1 = box
    gain, wall = wall_gain(rgb, box) if calibrate else (np.ones(3, np.float32), None)
    c = np.clip(rgb[y0:y1, x0:x1].astype(np.float32) * gain, 0, 255)
    H, W = c.shape[:2]
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    # Dominant saturated field (largest warm component)
    hsv = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    field = ((sat > 140) & (r > 120) & (r > b * 1.6)).astype(np.uint8)
    field = cv2.morphologyEx(field, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(field, 8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fm = lab == idx
    ys, xs = np.nonzero(fm)
    fx0, fx1 = np.percentile(xs, [1, 99]); fy0, fy1 = np.percentile(ys, [1, 99])
    fbox = [fx0 / W, fy0 / H, fx1 / W, fy1 / H]
    # Field colour stops: top / middle / bottom thirds; plus the light core
    field_stops = []
    for a, bnd in [(0.0, 0.12), (0.3, 0.7), (0.85, 1.0)]:
        yy0 = int(fy0 + a * (fy1 - fy0)); yy1 = int(fy0 + bnd * (fy1 - fy0)) + 1
        sel = fm[yy0:yy1]
        field_stops.append(np.median(c[yy0:yy1][sel], axis=0).tolist())
    # Border band: luminance profile from each edge inward
    L = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    mid_rows = slice(int(0.55 * H), int(0.9 * H))
    prof_l = L[mid_rows, : W // 5].mean(0)
    prof_r = L[mid_rows, -(W // 5):].mean(0)[::-1]
    prof_t = L[: H // 6, int(0.1 * W): int(0.9 * W)].mean(1)
    prof_b = L[-(H // 6):, int(0.1 * W): int(0.9 * W)].mean(1)[::-1]

    def band_width(p):
        p = gaussian_filter(p, 2)
        inner = np.median(p[len(p) // 2:]); outer = np.median(p[: max(3, len(p) // 12)])
        thr = (inner + outer) / 2
        above = np.where(p > thr)[0]
        return int(above[0]) if len(above) else len(p) // 4

    bw = [band_width(prof_l) / W, band_width(prof_t) / H, band_width(prof_r) / W, band_width(prof_b) / H]
    bmask = np.zeros((H, W), bool)
    bmask[:, : int(bw[0] * W * 0.8)] = True; bmask[: int(bw[1] * H * 0.8)] = True
    bmask[:, W - int(bw[2] * W * 0.8):] = True; bmask[H - int(bw[3] * H * 0.8):] = True
    border_cols, border_w = _kmeans(c[bmask].reshape(-1, 3)[::7], 3)
    # Ground: inside border, outside field + fringe; colour by vertical band and a k-means palette
    gmask = ~bmask.copy()
    gmask[:, : int(bw[0] * W * 1.3)] = False; gmask[:, W - int(bw[2] * W * 1.3):] = False
    gmask[: int(bw[1] * H * 1.3)] = False; gmask[H - int(bw[3] * H * 1.3):] = False
    fy1p = int(fy1 + 0.12 * H)
    gmask[int(fy0) - 8: fy1p, int(fx0) - 8: int(fx1) + 8] = False
    bands = []
    for k in range(6):
        a0, a1 = int(k * H / 6), int((k + 1) * H / 6)
        sel = gmask[a0:a1]
        bands.append(np.median(c[a0:a1][sel], axis=0).tolist() if sel.any() else None)
    bands = [b_ if b_ is not None else bands[i - 1] for i, b_ in enumerate(bands)]
    ground_cols, ground_w = _kmeans(c[gmask].reshape(-1, 3)[::11], 5)
    # Dry fringe below the field: pixels clearly darker / cooler than the local ground
    strip = c[int(fy1): int(fy1 + 0.06 * H), int(fx0): int(fx1)]
    sl = cv2.cvtColor(strip.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    ssat = strip.max(2) - strip.min(2)
    neutral = ssat < np.percentile(ssat, 50)                 # exclude orange pulled down past the edge
    dk = neutral & (sl < np.percentile(sl[neutral], 12))
    fringe = np.median(strip[dk], axis=0).tolist()
    fringe_depth = 0.0
    col_profile = sl.mean(1)
    base = np.median(L[int(fy1 + 0.1 * H): int(fy1 + 0.14 * H), int(fx0): int(fx1)])
    below = np.where(col_profile > base - 3)[0]
    fringe_depth = (below[0] if len(below) else len(col_profile)) / H
    # Top edge accent: thin brighter, yellower line above the field?
    top = c[max(0, int(fy0) - 4): int(fy0) + 6, int(fx0): int(fx1)]
    top_col = np.median(top.reshape(-1, 3), axis=0).tolist()
    return {
        "source": "reference-summary",
        "reference": Path(path).name,
        "reference_canvas_box_xyxy": [int(v) for v in box],
        "photo_calibration": {"wall_rgb": None if wall is None else [round(float(v), 1) for v in wall],
                              "gain_rgb": [round(float(v), 3) for v in gain]},
        "aspect_w_over_h": W / H,
        "field_box_norm": [round(float(v), 4) for v in fbox],
        "field_stops_top_mid_bottom": [[round(float(x), 1) for x in s] for s in field_stops],
        "field_top_accent": [round(float(x), 1) for x in top_col],
        "border_width_norm_ltrb": [round(float(v), 4) for v in bw],
        "border_palette": [[round(float(x), 1) for x in s] for s in border_cols.tolist()],
        "ground_bands_top_to_bottom": [[round(float(x), 1) for x in s] for s in bands],
        "ground_palette": [[round(float(x), 1) for x in s] for s in ground_cols.tolist()],
        "ground_palette_weights": [round(float(v), 3) for v in ground_w],
        "fringe_color": [round(float(x), 1) for x in fringe],
        "fringe_depth_norm": round(float(max(0.02, min(0.08, fringe_depth))), 4),
    }


HARMONIES = [  # (field top, mid, bottom, accent), ground light, ground warm, border, fringe
    dict(field=[(214, 132, 18), (205, 114, 22), (196, 84, 20)], accent=(218, 170, 40),
         ground=[(228, 222, 204), (222, 212, 190), (214, 196, 172)], border=[(150, 140, 118), (126, 117, 98)], fringe=(138, 112, 112)),
    dict(field=[(130, 40, 38), (150, 48, 40), (104, 34, 36)], accent=(180, 84, 50),
         ground=[(206, 196, 178), (188, 176, 160), (170, 154, 140)], border=[(78, 70, 66), (96, 86, 78)], fringe=(84, 62, 70)),
    dict(field=[(48, 86, 110), (38, 74, 98), (30, 58, 80)], accent=(120, 160, 170),
         ground=[(222, 218, 206), (210, 206, 194), (196, 190, 178)], border=[(120, 120, 112), (100, 102, 98)], fringe=(94, 104, 118)),
    dict(field=[(196, 160, 60), (184, 146, 50), (160, 120, 40)], accent=(226, 200, 90),
         ground=[(54, 58, 62), (66, 66, 68), (48, 50, 54)], border=[(28, 30, 32), (40, 40, 42)], fringe=(110, 96, 70)),
]


def brief_from_grammar(rng, aspect=0.8):
    h = HARMONIES[int(rng.integers(len(HARMONIES)))]
    fw = rng.uniform(0.42, 0.66); fh = rng.uniform(0.18, 0.32)
    cx = 0.5 + rng.normal(0, 0.03); top = rng.uniform(0.16, 0.30)
    bw = float(rng.uniform(0.03, 0.075))
    g = [np.array(c, np.float32) for c in h["ground"]]
    bands = [lerp_palette(g, t).tolist() for t in np.linspace(0, 1, 6)]
    return {
        "source": "grammar",
        "aspect_w_over_h": aspect,
        "field_box_norm": [cx - fw / 2, top, cx + fw / 2, top + fh],
        "field_stops_top_mid_bottom": [list(c) for c in h["field"]],
        "field_top_accent": list(h["accent"]),
        "border_width_norm_ltrb": [bw, bw * aspect, bw, bw * aspect * 1.1],
        "border_palette": [list(c) for c in h["border"]],
        "ground_bands_top_to_bottom": bands,
        "ground_palette": [list(c) for c in h["ground"]],
        "ground_palette_weights": [0.5, 0.3, 0.2],
        "fringe_color": list(h["fringe"]),
        "fringe_depth_norm": float(rng.uniform(0.025, 0.06)),
    }


# --------------------------------------------------------------------------------------------
# Canvas + stroke model
# --------------------------------------------------------------------------------------------
class Canvas:
    def __init__(self, w, h, width_mm, rng, weave_pitch_mm=0.7, substrate=None):
        self.w, self.h, self.rng = w, h, rng
        self.substrate = substrate
        self.px_mm = width_mm / w
        self.s = w / REF_W_PX  # pixels per reference pixel; marks are sized in reference px * s
        sig = 0.7 * self.s     # tooth grain: same physical size at every resolution
        if sig <= 1.5:
            fine = gaussian_filter(rng.normal(size=(h, w)).astype(np.float32), sig)
            fine /= fine.std() + 1e-6
        else:  # smooth noise built on a coarse grid (identical spectrum, far less memory)
            f = sig / 1.5
            g = gaussian_filter(rng.normal(size=(int(h / f) + 3, int(w / f) + 3)).astype(np.float32), 1.5)
            fine = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); del g
            fine -= fine.mean(); fine /= fine.std() + 1e-6
        mid = Noise(rng, h, w, 9 * self.s)
        self.tooth = np.empty((h, w), np.float32)
        for a, b in row_chunks(h, w):
            self.tooth[a:b] = np.clip(0.5 + 0.18 * fine[a:b] + 0.12 * mid.crop(a, b, 0, w), 0, 1)
        del fine
        # Linen weave, only once the thread pitch is resolvable (>= ~3 px). Below that it would alias,
        # so low-resolution renders keep the purely stochastic tooth (and stay byte-identical to v1.6).
        self.weave_pitch_px = weave_pitch_mm / self.px_mm
        self.weave_gate = float(smoothstep(5.0, 7.5, self.weave_pitch_px))  # needs >=5 px per thread or it aliases
        if substrate is not None:
            self.weave_gate = 0.0   # the substrate model below replaces the legacy weave
        self.weave_amp_mm = 0.012 * self.weave_gate
        if self.weave_gate > 0:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from weave import Weave
            wv_gen = Weave(np.random.default_rng(int(rng.integers(1 << 30))), h, w, self.weave_pitch_px)
            for a, b in row_chunks(h, w):
                wv = wv_gen.rows(a, b)
                self.tooth[a:b] = np.clip(self.tooth[a:b] * (1 - 0.35 * self.weave_gate) +
                                          0.35 * self.weave_gate * (0.5 + 0.5 * wv), 0, 1)
        self.height = np.empty((h, w), np.float32)
        for a, b in row_chunks(h, w):
            self.height[a:b] = 0.02 + 0.015 * (self.tooth[a:b] - 0.5) + self.weave_amp_mm * (self.tooth[a:b] - 0.5) * 2
        self.weave = None
        if substrate is not None:
            self._init_substrate(SUBSTRATES[substrate] if isinstance(substrate, str) else substrate, fine_seed=rng)
        self.color = np.empty((h, w, 3), np.float32); self.color[:] = (238, 234, 224)
        self.rough = np.full((h, w), 0.85, np.float32)
        self.max_tile = 2_000_000
        self.material = np.zeros((h, w), np.uint8)
        self.log = []
        self.counts = {}

    def _init_substrate(self, sub, fine_seed):
        """Primed woven substrate. Weave depth is stored separately so paint can level it."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from weave import Weave, weave_rows_supersampled
        h, w = self.h, self.w
        self.sub = dict(sub)
        pitch_px = sub["pitch_mm"] / self.px_mm
        self.weave_pitch_px = pitch_px
        self.weave_seed = int(fine_seed.integers(1 << 30))
        f = 1 if pitch_px >= 8 else int(math.ceil(8 / pitch_px))     # supersample small threads
        self.weave_supersample = f
        gen = Weave(np.random.default_rng(self.weave_seed), h * f, w * f, pitch_px * f, slub=sub["slub"])
        self.weave = np.empty((h, w), np.float16)        # -1..1 thread relief
        self.weave_shown = np.ones((h, w), np.float16)   # fraction of weave depth not yet levelled by paint
        self.sub_base = self.height.copy()                # primed surface without the weave
        amp = sub["amp_mm"]
        for a, b in row_chunks(h, w, BIG // max(1, f * f)):
            wv = weave_rows_supersampled(gen, f, a, b, w) if f > 1 else gen.rows(a, b)
            self.weave[a:b] = wv
            self.height[a:b] += amp * wv
            if amp > 0:  # dry paint catches on the threads; gesso noise stays as a secondary grain
                self.tooth[a:b] = np.clip(0.55 * self.tooth[a:b] + 0.45 * (0.5 + 0.5 * wv), 0, 1)

    def level(self):
        """Paint levels the weave: visible weave depth falls as exp(-film / level_mm)."""
        if self.weave is None or self.sub["amp_mm"] <= 0:
            return
        amp, lev = self.sub["amp_mm"], self.sub["level_mm"]
        for a, b in row_chunks(self.h, self.w):
            wv = self.weave[a:b].astype(np.float32); shown = self.weave_shown[a:b].astype(np.float32)
            film = np.maximum(0, self.height[a:b] - self.sub_base[a:b] - amp * wv * shown)
            target = np.exp(-film / lev)
            d = target - shown
            self.height[a:b] += amp * wv * d
            self.tooth[a:b] = np.clip(self.tooth[a:b] + 0.45 * 0.5 * wv * d, 0, 1)
            self.weave_shown[a:b] = target

    def stroke(self, p0, p1, width, color, *, load=1.0, thickness=0.03, decay=0.6, dry=0.45,
               pickup=0.15, opacity=1.0, bend=0.0, rough=0.6, material="ground", clip=None,
               ridge=0.4, fray=0.25, streak=0.5, endpile=0.0, groove=1.0):
        """Lay one bristle-brush stroke from p0 to p1.

        load      initial paint load (>1 = loaded brush, <0.5 = dry brush)
        decay     fraction of stroke length after which load falls to 1/e
        dry       how strongly transfer depends on surface tooth (0 = covers flat, 1 = peaks only)
        pickup    how much underlying wet colour the brush drags along by the stroke's end
        fray      fraction of length over which individual bristles lift off at the end
        streak    strength of along-stroke bristle streaking
        clip      optional (h,w) float mask multiplying coverage (edge discipline)
        """
        rng = self.rng
        self._rel = (endpile, groove)
        build = getattr(self, "build", None)
        if build is not None:   # regional paint build-up: scrubbed-thin vs built-up passages
            mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
            thickness = thickness * float(build(my, mx))
        p0 = np.asarray(p0, np.float32); p1 = np.asarray(p1, np.float32)
        d = p1 - p0; L = float(np.hypot(*d))
        if L < 2 or width < 1:
            return
        d /= L; nrm = np.array([-d[1], d[0]], np.float32)
        half = width / 2.0
        pad = half + abs(bend) + 2
        xa = int(max(0, math.floor(min(p0[0], p1[0]) - pad))); xb = int(min(self.w, math.ceil(max(p0[0], p1[0]) + pad)))
        ya = int(max(0, math.floor(min(p0[1], p1[1]) - pad))); yb = int(min(self.h, math.ceil(max(p0[1], p1[1]) + pad)))
        if xb <= xa or yb <= ya:
            return
        s = self.s
        # Bristle bundle: per-bristle density, start and lift-off positions, along-stroke texture.
        # Counts use physical size (px / s) so the brush looks the same at every resolution.
        nb = int(np.clip(width / s * 0.9, 10, 160)); nu = int(np.clip(L / s / 3, 8, 256))
        geom = (p0, d, nrm, L, half, bend)
        st = {}
        # Tile very large strokes so per-stroke temporaries stay bounded at print resolution
        rows = max(1, self.max_tile // max(1, xb - xa))
        any_paint = False
        for ta in range(ya, yb, rows):
            tb = min(yb, ta + rows)
            any_paint |= self._stroke_tile(ta, tb, xa, xb, geom, nb, nu, st, color, load, thickness, decay, dry,
                                           pickup, opacity, rough, material, clip, ridge, fray, streak)
        if any_paint:
            self.counts[material] = self.counts.get(material, 0) + 1

    def _bundle(self, st, nb, nu, fray, streak):
        if st:
            return st
        rng = self.rng
        dens = np.clip(rng.lognormal(0, 0.35, nb), 0.15, 2.2)
        dens[rng.random(nb) < 0.08] *= 0.15  # missing / splayed bristles -> clean gaps
        dens = gaussian_filter(dens, 0.6)
        start = rng.uniform(0, 0.04, nb)
        lift = 1.0 - fray * rng.random(nb) ** 1.6
        tex = rng.normal(size=(nb, nu)).astype(np.float32)
        tex = gaussian_filter(tex, (0.5, max(1.0, nu / 10)))
        tex = 1 + streak * tex / (tex.std() + 1e-6) * 0.5
        st.update(dens=dens, start=start, lift=lift, tex=tex)
        return st

    def _stroke_tile(self, ya, yb, xa, xb, geom, nb, nu, st, color, load, thickness, decay, dry,
                     pickup, opacity, rough, material, clip, ridge, fray, streak):
        p0, d, nrm, L, half, bend = geom
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        rx, ry = xx - p0[0], yy - p0[1]
        u = (rx * d[0] + ry * d[1]) / L
        v = rx * nrm[0] + ry * nrm[1]
        uc = np.clip(u, 0, 1)
        v = v - bend * 4 * uc * (1 - uc)
        vn = v / half
        inside = (np.abs(vn) < 1.0) & (u > -0.02) & (u < 1.0)
        if not inside.any():
            return False
        b_ = self._bundle(st, nb, nu, fray, streak)
        dens, start, lift, tex = b_["dens"], b_["start"], b_["lift"], b_["tex"]
        bi = np.clip((vn + 1) / 2 * (nb - 1), 0, nb - 1)
        ui = np.clip(u, 0, 1) * (nu - 1)
        T = map_coordinates(tex, [bi.ravel(), ui.ravel()], order=1, mode="nearest").reshape(u.shape)
        bint = np.rint(bi).astype(int)
        if self.s > 1.5:
            # print resolution: anti-aliased bristle touch-down/lift-off (about one pixel of ramp)
            # bristle start/lift vary smoothly across the brush (a bristle is < 1 px wide at print scale)
            bx = np.arange(nb, dtype=np.float32)
            st_i = np.interp(bi, bx, gaussian_filter(start, 1.5))
            lf_i = np.interp(bi, bx, gaussian_filter(lift, 1.5))
            e = 3.0 * self.s / L          # paint thins out over ~0.5 mm where a bristle lifts
            on = inside * smoothstep(st_i - e, st_i + e, u) * (1 - smoothstep(lf_i - e, lf_i + e, u))
            lift_u = lf_i
        else:
            on = inside & (u >= start[bint]) & (u <= lift[bint])
            lift_u = lift[bint]
        # Paint load depletes along the stroke; profile falls off at the brush shoulders
        load_u = load * np.exp(-np.clip(u, 0, 1) / max(decay, 1e-3))
        # Soft shoulders and a touch-down ramp: no square-cut stroke footprints
        shoulder = smoothstep(1.0, 0.72, np.abs(vn))
        touch = smoothstep(0.0, min(0.5, 6.0 * self.s / L), u)
        dn = np.interp(bi, np.arange(nb, dtype=np.float32), dens) if self.s > 1.5 else dens[bint]
        amount = load_u * dn * T * shoulder * (0.35 + 0.65 * touch)
        if self.s > 1.5:
            # print scale: no hard cut where the brush lands or lifts (avoids stepped height cliffs)
            e_px = 4.0 * self.s / L
            amount = amount * smoothstep(-0.02, -0.02 + 2 * e_px, u) * (1 - smoothstep(1.0 - 2 * e_px, 1.0, u))
        # Dry-brush transfer: paint reaches the valleys only when the load exceeds their depth
        tooth = self.tooth[ya:yb, xa:xb]
        if self.weave is not None:
            # a loaded brush pushes paint down into the thread valleys; only a starved brush skips them
            dry = dry * float(np.clip(1.5 - load, 0.15, 1.0))
        need = dry * (1.0 - tooth) * 1.6
        cov = smoothstep(need, need + 0.35, amount) * on
        if clip is not None:
            cov = cov * (clip.get(ya, yb, xa, xb) if hasattr(clip, "get") else clip[ya:yb, xa:xb])
        a = np.clip(cov * opacity, 0, 1)
        if a.max() <= 0.003:
            return False
        # Brush pickup: colour becomes dirtier toward the end of the stroke
        old = self.color[ya:yb, xa:xb]
        k = np.clip(pickup * np.clip(u, 0, 1), 0, 0.9)[..., None]
        pig = np.asarray(color, np.float32)[None, None, :] * (1 - k) + old * k
        self.color[ya:yb, xa:xb] = old * (1 - a[..., None]) + pig * a[..., None]
        # Relief: thickness follows load; loaded strokes push ridges to their shoulders
        prof = 1.0 + ridge * np.clip(load - 0.8, 0, 1) * (np.abs(vn) ** 3)
        endpile, groove = self._rel
        dz = cov * thickness * np.clip(load_u, 0, 1.6) * prof * (1.0 + groove * 0.25 * (T - 1.0))
        if endpile > 0:
            # paint pushed ahead of the brush piles up where each bristle lifts off, and a little at touch-down
            wpile = max(0.02, 6.0 * self.s / L)
            pile = np.exp(-((u - lift_u + 0.5 * wpile) / wpile) ** 2) + 0.35 * np.exp(-(u / (0.7 * wpile)) ** 2)
            dz = dz + inside * shoulder * pile * thickness * endpile * np.clip(load, 0, 1.5)
        self.height[ya:yb, xa:xb] += dz
        # Paint leaves its own tooth (bristle ridges along the stroke) for later dry passes
        self.tooth[ya:yb, xa:xb] = np.clip(tooth * (1 - 0.45 * a) + 0.45 * a * np.clip(0.5 + 0.35 * (T - 1) / max(streak, 1e-3), 0, 1), 0, 1)
        self.rough[ya:yb, xa:xb] = self.rough[ya:yb, xa:xb] * (1 - a) + rough * a
        sub = self.material[ya:yb, xa:xb]
        sub[a > 0.45] = MATERIALS[material]
        return True


class FieldMask:
    """Soft field mask evaluated per tile (no full-canvas arrays)."""

    def __init__(self, painter, grow=0.0, ragged=1.6, grow_top=None, fray_bottom=0.0, grow_bottom=None, invert=False):
        r = painter.rng; w, h, s = painter.w, painter.h, painter.s
        self.s, self.invert = s, invert
        self.x0, self.y0, self.x1, self.y1 = painter._field_px()
        self.grow, self.ragged = grow, ragged
        self.gt = grow if grow_top is None else grow_top
        self.gb = grow if grow_bottom is None else grow_bottom
        self.nl, self.nr, self.nt = Noise(r, h, w, 60 * s), Noise(r, h, w, 70 * s), Noise(r, h, w, 55 * s)
        self.fray = fray_bottom
        if fray_bottom <= 0:
            self.nb = Noise(r, h, w, 45 * s)
        else:
            col = gaussian_filter(r.normal(size=w).astype(np.float32), 0.5 * s)
            col = col / (col.std() + 1e-6)
            slow = value_noise(r, 1, w, 40 * s)[0]
            self.reach = (self.y1 - 3 * s + grow + fray_bottom * s * np.maximum(0, 0.45 + 0.7 * col + 0.4 * slow))

    def get(self, ya, yb, xa, xb):
        s, rg = self.s, self.ragged * self.s
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        dl = xx - (self.x0 - self.grow + self.nl.crop(ya, yb, xa, xb) * rg)
        dr = (self.x1 + self.grow + self.nr.crop(ya, yb, xa, xb) * rg) - xx
        dt = yy - (self.y0 - self.gt + self.nt.crop(ya, yb, xa, xb) * rg)
        side = smoothstep(-1.5 * s, 2.5 * s, np.minimum.reduce([dl, dr, dt]))
        if self.fray <= 0:
            db = (self.y1 + self.gb + self.nb.crop(ya, yb, xa, xb) * rg) - yy
            m = side * smoothstep(-1.5 * s, 2.5 * s, db)
        else:
            m = side * smoothstep(-1.5 * s, 3.0 * s, self.reach[None, xa:xb] - yy)
        return 1.0 - m if self.invert else m


class BorderBand:
    def __init__(self, painter, sides):
        r = painter.rng; self.w, self.h, s = painter.w, painter.h, painter.s
        self.sides = [(side, bwid, Noise(r, self.h, self.w, 90 * s)) for side, bwid in sides]

    def get(self, ya, yb, xa, xb):
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        dist = {"l": xx, "r": self.w - 1 - xx, "t": yy, "b": self.h - 1 - yy}
        band = np.zeros(yy.shape, np.float32)
        for side, bwid, nz in self.sides:
            edge = bwid * (1.0 + 0.08 * nz.crop(ya, yb, xa, xb))
            band = np.maximum(band, smoothstep(edge + 0.25 * bwid, edge - 0.15 * bwid, dist[side]))
        return band


# --------------------------------------------------------------------------------------------
# Painter: the plan for a colour-field picture, executed as passes of strokes
# --------------------------------------------------------------------------------------------
class ColourFieldPainter:
    def __init__(self, brief, seed, width_px=850, width_mm=760.0, height_px=None, substrate=None):
        self.brief, self.seed = brief, seed
        self.rng = np.random.default_rng(seed)
        w = int(width_px); h = int(height_px) if height_px else int(round(w / brief["aspect_w_over_h"]))
        self.cv = Canvas(w, h, width_mm, self.rng, substrate=substrate)
        self.w, self.h, self.s = w, h, self.cv.s
        # stroke counts follow physical canvas area relative to the reference canvas (not pixel count)
        # Marks are sized relative to canvas width (s = w / 850), so stroke counts follow the canvas
        # area in reference pixels: identical to v1.6 at 850 px, and constant as resolution rises.
        self.k = (w / self.s) * (h / self.s) / (REF_W_PX * 1065)
        self.passes = []
        # Broad light/dark zones the painter works toward (not copied: a seeded low-frequency field)
        self._d1, self._d2 = Noise(self.rng, h, w, 260 * self.s), Noise(self.rng, h, w, 110 * self.s)

    def drift_at(self, y, x):
        return np.float32(self._d1.at(y, x) * 7 + self._d2.at(y, x) * 4)

    # pass bookkeeping ------------------------------------------------------------------------
    def _pass(self, name, fn):
        t = time.time(); before = dict(self.cv.counts)
        fn()
        self.cv.level()
        n = sum(self.cv.counts.values()) - sum(before.values())
        self.passes.append({"pass": name, "strokes": n, "seconds": round(time.time() - t, 2)})

    def _ground_color(self, y):
        bands = self.brief["ground_bands_top_to_bottom"]
        return lerp_palette(bands, np.clip(y / self.h, 0, 1))

    def _field_px(self):
        fx0, fy0, fx1, fy1 = self.brief["field_box_norm"]
        return fx0 * self.w, fy0 * self.h, fx1 * self.w, fy1 * self.h

    def _field_mask(self, grow=0.0, ragged=1.6, grow_top=None, fray_bottom=0.0, grow_bottom=None):
        """Soft field mask with hand-cut, slightly wandering edges (no vector rectangle)."""
        return FieldMask(self, grow, ragged, grow_top, fray_bottom, grow_bottom)
        r = self.rng; w, h, s = self.w, self.h, self.s
        x0, y0, x1, y1 = self._field_px()
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        wob = lambda n, cell: value_noise(r, h, w, cell * s) * ragged * s
        dl = xx - (x0 - grow + wob(0, 60)); dr = (x1 + grow + wob(0, 70)) - xx
        gt = grow if grow_top is None else grow_top
        dt = yy - (y0 - gt + wob(0, 55))
        side = smoothstep(-1.5 * s, 2.5 * s, np.minimum.reduce([dl, dr, dt]))
        if fray_bottom <= 0:
            db = (y1 + (grow if grow_bottom is None else grow_bottom) + wob(0, 45)) - yy
            return side * smoothstep(-1.5 * s, 2.5 * s, db)
        # Frayed lower boundary: each bristle-width column stops at its own depth, so the edge
        # is a comb of streaks of varying length rather than a cut line.
        col = gaussian_filter(r.normal(size=w).astype(np.float32), 0.5 * s)
        col = col / (col.std() + 1e-6)
        slow = value_noise(r, 1, w, 40 * s)[0]
        reach = (y1 - 3 * s + grow + fray_bottom * s * np.maximum(0, 0.45 + 0.7 * col + 0.4 * slow))[None, :]
        db = reach - yy
        return side * smoothstep(-1.5 * s, 3.0 * s, db)

    # passes -----------------------------------------------------------------------------------
    def imprimatura(self):
        r, w, h, s = self.rng, self.w, self.h, self.s
        base = np.mean(self.brief["ground_bands_top_to_bottom"], axis=0)
        tint = np.clip(base * 0.92 + np.array([6, 2, -6]), 0, 255)
        # Toned ground: a thin, rubbed-in wash over the whole canvas before any brushwork
        m1, m2 = Noise(r, h, w, 120 * s), Noise(r, h, w, 25 * s)
        for a, b in row_chunks(h, w):
            mott = m1.crop(a, b, 0, w) * 4 + m2.crop(a, b, 0, w) * 2
            self.cv.color[a:b] = np.clip(tint[None, None, :] + mott[..., None], 0, 255)
        self.cv.height += 0.003
        for _ in range(int(70)):
            x = r.uniform(-0.1, 1.1) * w; y = r.uniform(-0.05, 1.05) * h
            ang = r.normal(0.0, 0.25) + (math.pi / 2 if r.random() < 0.5 else 0)
            L = r.uniform(250, 600) * s
            self.cv.stroke((x, y), (x + L * math.cos(ang), y + L * math.sin(ang)), r.uniform(90, 180) * s,
                           jitter_color(r, tint, 4), load=1.5, thickness=0.004, decay=1.5, dry=0.0,
                           pickup=0.05, opacity=0.85, bend=r.normal(0, 12) * s, material="imprimatura", streak=0.3)

    def border(self):
        r, w, h, s = self.rng, self.w, self.h, self.s
        bl, bt, br, bb = self.brief["border_width_norm_ltrb"]
        pal = self.brief["border_palette"]
        sides = [("l", bl * w), ("r", br * w), ("t", bt * h), ("b", bb * h)]
        # Soft, wandering inner boundary: the band is painted up to a hand-held line, then feathered
        band = BorderBand(self, sides)
        for layer in range(3):
            for side, bwid in sides:
                n = int((h if side in "lr" else w) / (40 * s)) * (2 if layer == 0 else 1)
                for _ in range(n):
                    width = r.uniform(0.45, 0.95) * bwid
                    off = r.uniform(-0.1, 1.0) * bwid           # reaches the canvas edge
                    t = np.clip(off / bwid, 0, 1)                 # outer = darker, inner = closer to ground
                    along = r.uniform(-0.1, 1.05); L = r.uniform(120, 380) * s
                    if side in "lr":
                        x = off if side == "l" else w - off; y = along * h
                        p0, p1 = (x, y), (x + r.normal(0, 2 * s), y + L)
                        gc = self._ground_color(y + L / 2)
                    else:
                        y = off if side == "t" else h - off; x = along * w
                        p0, p1 = (x, y), (x + L, y + r.normal(0, 2 * s))
                        gc = self._ground_color(y)
                    bc = np.asarray(pal[int(r.integers(len(pal)))], np.float32)
                    col = jitter_color(r, bc * (1 - 0.4 * t) + gc * 0.4 * t, 5)
                    self.cv.stroke(p0, p1, width, col, load=1.15 - 0.3 * layer, thickness=0.02,
                                   decay=0.9, dry=0.3 + 0.2 * layer, pickup=0.3, opacity=0.75 - 0.15 * layer,
                                   bend=r.normal(0, 3) * s, rough=0.65, material="border", clip=band)

    def ground(self):
        r, w, h, s = self.rng, self.w, self.h, self.s
        bl, bt, br, bb = self.brief["border_width_norm_ltrb"]
        pal = self.brief["ground_palette"]; wts = np.asarray(self.brief["ground_palette_weights"], float)
        wts = wts / wts.sum()
        X0, X1 = bl * w * 0.75, w - br * w * 0.75
        Y0, Y1 = bt * h * 0.75, h - bb * h * 0.75
        # Broad body strokes: band colour (vertical tonal drift) nudged toward a palette colour
        for i in range(int(900 * self.k)):
            y = r.uniform(Y0, Y1); x = r.uniform(X0, X1)
            vertical = r.random() < 0.62
            L = r.uniform(80, 340) * s
            ang = (math.pi / 2 if vertical else 0) + r.normal(0, 0.12)
            if r.random() < 0.5:
                ang += math.pi
            p1 = (x + L * math.cos(ang), y + L * math.sin(ang))
            c = self._ground_color(y) * 0.85 + np.asarray(pal[r.choice(len(pal), p=wts)], np.float32) * 0.15
            c = c + self.drift_at(y, x) * np.array([1.0, 0.95, 0.85])
            self.cv.stroke((x, y), p1, r.uniform(30, 110) * s, jitter_color(r, c, 3),
                           load=r.uniform(0.75, 1.25), thickness=0.018, decay=r.uniform(0.4, 0.9),
                           dry=0.45, pickup=0.3, opacity=r.uniform(0.45, 0.85),
                           bend=r.normal(0, 8) * s, rough=0.55, material="ground")

    def scumble(self):
        """Dry, broken light passes that catch the tooth and earlier ridges; plus faint pentimenti."""
        r, w, h, s = self.rng, self.w, self.h, self.s
        bl, bt, br, bb = self.brief["border_width_norm_ltrb"]
        light = np.max(np.asarray(self.brief["ground_palette"], np.float32), axis=0)
        avoid = FieldMask(self, grow=8 * s, grow_bottom=self.brief["fringe_depth_norm"] * h * 1.2, invert=True)
        for i in range(int(420 * self.k)):
            x = r.uniform(bl * w * 0.6, w - br * w * 0.6); y = r.uniform(bt * h * 0.6, h - bb * h * 0.6)
            vertical = r.random() < 0.55
            L = r.uniform(60, 260) * s
            ang = (math.pi / 2 if vertical else 0) + r.normal(0, 0.1)
            roll = r.random()
            if roll < 0.62:
                c = jitter_color(r, 0.45 * light + 0.55 * self._ground_color(y) + 4, 3); op = r.uniform(0.18, 0.4)
            elif roll < 0.85:
                c = jitter_color(r, self._ground_color(y) * 0.9 + np.array([8, -4, -4]), 6); op = r.uniform(0.15, 0.35)  # pink-grey
            else:
                c = jitter_color(r, self._ground_color(y) * 0.82, 5); op = r.uniform(0.12, 0.28)  # shadowy pentimento
            self.cv.stroke((x, y), (x + L * math.cos(ang), y + L * math.sin(ang)), r.uniform(18, 80) * s, c,
                           load=r.uniform(0.35, 0.7), thickness=0.012, decay=0.5, dry=0.85, pickup=0.1,
                           opacity=op, bend=r.normal(0, 6) * s, rough=0.7, material="scumble", streak=0.8,
                           clip=avoid)

    def field(self):
        r, w, h, s = self.rng, self.w, self.h, self.s
        x0, y0, x1, y1 = self._field_px()
        top, mid, bot = [np.asarray(c, np.float32) for c in self.brief["field_stops_top_mid_bottom"]]
        accent = np.asarray(self.brief["field_top_accent"], np.float32)
        under = self._field_mask(grow=-2 * s, grow_top=4 * s, grow_bottom=-14 * s)
        main = self._field_mask(grow=0.0, fray_bottom=16.0)
        fw, fh = x1 - x0, y1 - y0
        # 1. Warm yellow underlayer, slightly larger at the top so it glints at the edge
        under_col = np.clip(accent * 0.55 + top * 0.45, 0, 255)
        for _ in range(int(140)):
            x = r.uniform(x0, x1); y = r.uniform(y0 - 6 * s, y1)
            L = r.uniform(0.25, 0.7) * fw
            self.cv.stroke((x - L / 2, y), (x + L / 2, y + r.normal(0, 3) * s), r.uniform(20, 55) * s,
                           jitter_color(r, under_col, 6), load=1.2, thickness=0.015, decay=1.0, dry=0.2,
                           pickup=0.05, opacity=0.9, clip=under, material="field_under")

        def fcol(y, x):
            t = np.clip((y - y0) / fh, 0, 1)
            c = lerp_palette([top, mid, bot], t)
            # lighter, yellower core; redder flanks (the field glows from its centre)
            core = math.exp(-(((x - (x0 + 0.5 * fw)) / (0.35 * fw)) ** 2 + ((y - (y0 + 0.4 * fh)) / (0.45 * fh)) ** 2))
            return np.clip(c + core * np.array([6, 14, 0]) - (1 - core) * np.array([0, 6, 2]), 0, 255)

        # 2. Body: mostly vertical, overlapping, wet-into-wet strokes with brush pickup
        for _ in range(int(800)):
            x = r.uniform(x0 - 4 * s, x1 + 4 * s); y = r.uniform(y0 - 10 * s, y1)
            L = r.uniform(0.2, 0.75) * fh
            if r.random() < 0.78:
                p0, p1 = (x, y), (x + r.normal(0, 3) * s, y + L)
                wd = r.uniform(12, 40) * s
            else:
                p0, p1 = (x - L, y), (x + L, y + r.normal(0, 3) * s)
                wd = r.uniform(25, 70) * s
            self.cv.stroke(p0, p1, wd, jitter_color(r, fcol(y + L / 2, x), 4), load=r.uniform(0.8, 1.3),
                           thickness=0.022, decay=r.uniform(0.5, 1.2), dry=0.35, pickup=0.35,
                           opacity=r.uniform(0.3, 0.6), bend=r.normal(0, 3) * s, rough=0.45,
                           clip=main, material="field", ridge=0.6)
        # 3. Thin vertical drags: lighter and darker pulls that read as bristle streaks
        for _ in range(int(160)):
            x = r.uniform(x0, x1); y = r.uniform(y0, y1 - 0.2 * fh)
            L = r.uniform(0.15, 0.6) * fh
            c = fcol(y, x) + (np.array([16, 22, 6]) if r.random() < 0.5 else np.array([-30, -22, -8]))
            self.cv.stroke((x, y), (x + r.normal(0, 2) * s, y + L), r.uniform(4, 14) * s, jitter_color(r, c, 4),
                           load=r.uniform(0.4, 0.75), thickness=0.015, decay=0.6, dry=0.75, pickup=0.2,
                           opacity=r.uniform(0.25, 0.55), clip=main, material="field", streak=0.9)

    def edges(self):
        """Dry-brush fringe dragged down out of the field's lower edge, and a thin top accent."""
        r, w, h, s = self.rng, self.w, self.h, self.s
        x0, y0, x1, y1 = self._field_px()
        fw = x1 - x0
        fringe = np.asarray(self.brief["fringe_color"], np.float32)
        bot = np.asarray(self.brief["field_stops_top_mid_bottom"][2], np.float32)
        depth = self.brief["fringe_depth_norm"] * h
        edge_y = lambda x: y1 + r.normal(0, 3.0 * s)
        # Violet-grey scrubbed shadow below the edge (dirty brush, near-dry)
        for _ in range(int(420)):
            x = r.uniform(x0 - 6 * s, x1 + 4 * s); y = edge_y(x) + r.uniform(-2, 0.4 * depth / s) * s
            L = r.uniform(0.3, 1.6) * depth
            vertical = r.random() < 0.8
            p1 = (x + r.normal(0, 2) * s, y + L) if vertical else (x + L * 1.6, y + r.normal(0, 2) * s)
            mixg = r.uniform(0.25, 0.7)
            c = (1 - mixg) * fringe + mixg * self._ground_color(y) + np.array([0, -5, 3])
            near = math.exp(-max(0.0, y - y1) / (0.5 * depth))
            self.cv.stroke((x, y), p1, r.uniform(6, 30) * s, jitter_color(r, c, 8),
                           load=r.uniform(0.55, 0.9), thickness=0.008, decay=0.6, dry=0.7, pickup=0.15,
                           opacity=r.uniform(0.25, 0.5) + 0.3 * near, fray=0.8, streak=1.0, material="dry_fringe")
        # Orange that the brush pulled down past the edge, depleting fast
        for _ in range(int(240)):
            x = r.uniform(x0 + 2 * s, x1 - 2 * s); y = edge_y(x) - r.uniform(2, 12) * s
            L = r.uniform(0.2, 1.0) * depth * r.choice([0.7, 1.0, 1.4])
            self.cv.stroke((x, y), (x + r.normal(0, 2) * s, y + L), r.uniform(6, 22) * s, jitter_color(r, bot, 6),
                           load=r.uniform(0.7, 1.15), thickness=0.012, decay=0.4, dry=0.7, pickup=0.25,
                           opacity=r.uniform(0.35, 0.75), fray=0.7, streak=1.0, material="dry_fringe")
        # Dark bristle streaks rising into the bottom of the field (dirty brush dragged upward)
        dark = (0.55 * bot + 0.45 * fringe) * 0.72
        for _ in range(int(170)):
            x = r.uniform(x0 + 2 * s, x1 - 2 * s); y = edge_y(x) + r.uniform(0, 10) * s
            L = r.uniform(0.25, 1.0) * depth * 1.2
            self.cv.stroke((x, y), (x + r.normal(0, 1.5) * s, y - L), r.uniform(4, 16) * s, jitter_color(r, dark, 6),
                           load=r.uniform(0.6, 0.95), thickness=0.008, decay=0.45, dry=0.75, pickup=0.1,
                           opacity=r.uniform(0.4, 0.75), fray=0.8, streak=1.0, material="dry_fringe")
        # A few stray drops of fringe colour well below, as in a scrubbed passage
        for _ in range(int(25)):
            x = r.uniform(x0, x0 + 0.5 * fw); y = y1 + r.uniform(0.8, 2.2) * depth
            self.cv.stroke((x, y), (x + r.normal(0, 2) * s, y + r.uniform(8, 30) * s), r.uniform(3, 9) * s,
                           jitter_color(r, fringe, 8), load=0.45, thickness=0.005, dry=0.9, opacity=0.35,
                           material="dry_fringe")
        # Top accent: thin, broken, yellower line riding the upper edge
        acc = np.asarray(self.brief["field_top_accent"], np.float32)
        top = np.asarray(self.brief["field_stops_top_mid_bottom"][0], np.float32)
        acc = np.clip(0.5 * acc + 0.5 * top + np.array([12, 18, 0]), 0, 255)
        for _ in range(int(45)):
            x = r.uniform(x0, x1 - 40 * s); L = r.uniform(40, 200) * s
            y = y0 + r.uniform(-1.5, 3.5) * s
            self.cv.stroke((x, y), (min(x1, x + L), y + r.normal(0, 1) * s), r.uniform(3, 8) * s, jitter_color(r, acc, 6),
                           load=r.uniform(0.4, 0.75), thickness=0.012, decay=0.6, dry=0.85, opacity=r.uniform(0.25, 0.5),
                           material="edge_accent", streak=1.0)

    def glaze(self):
        """Very thin unifying veils over the ground; nearly no relief."""
        r, w, h, s = self.rng, self.w, self.h, self.s
        for _ in range(26):
            x = r.uniform(0.1, 0.9) * w; y = r.uniform(0.35, 0.95) * h
            L = r.uniform(150, 400) * s
            c = self._ground_color(y) * np.array([1.0, 0.95, 0.9]) + np.array([4, 0, 0])
            self.cv.stroke((x, y), (x + r.normal(0, 30) * s, y + L), r.uniform(80, 200) * s, jitter_color(r, c, 4),
                           load=1.5, thickness=0.002, decay=2.0, dry=0.05, pickup=0.0, opacity=0.08,
                           material="glaze", streak=0.2)

    def run(self):
        for name, fn in [("imprimatura", self.imprimatura), ("ground", self.ground), ("border", self.border),
                         ("field", self.field), ("edges", self.edges), ("scumble", self.scumble),
                         ("glaze", self.glaze)]:
            self._pass(name, fn)
        return self

    # export ---------------------------------------------------------------------------------
    def export(self, out, preview_exaggeration=3.0, dpi=None):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        cv = self.cv; w, h = self.w, self.h
        dpi = dpi or round(25.4 / cv.px_mm, 2)
        tags = {"dpi": (dpi, dpi)}
        albedo = np.empty((h, w, 3), np.uint8)
        for a, b in row_chunks(h, w):
            albedo[a:b] = np.clip(np.rint(cv.color[a:b]), 0, 255)
        Image.fromarray(albedo, "RGB").save(out / "albedo.png", **tags)
        peak = float(cv.height.max())
        scale_mm = math.ceil(peak * 1.25 * 100) / 100  # encode with headroom, not a fixed 3 mm
        h16 = np.empty((h, w), np.uint16)
        for a, b in row_chunks(h, w):
            h16[a:b] = np.rint(np.clip(cv.height[a:b] / scale_mm, 0, 1) * 65535)
        cv2.imwrite(str(out / "height_16bit.png"), h16); del h16
        Image.fromarray(np.uint8(np.clip(cv.rough * 255, 0, 255)), "L").save(out / "roughness.png", **tags)
        Image.fromarray(cv.material, "L").save(out / "material_id.png", **tags)
        # Physical normals: slope = dz(mm) / dx(mm). Smoothing radius is physical (0.6 reference px).
        z = gaussian_filter(cv.height, 0.6 if self.s <= 1.5 else 0.6 * min(self.s, 3.0))
        n16 = np.empty((h, w, 3), np.uint16); n8 = np.empty((h, w, 3), np.uint8)
        lit_out = np.empty((h, w, 3), np.uint8)
        light = np.array([-0.45, -0.55, 0.70], np.float32); light /= np.linalg.norm(light)
        flat = light[2]
        for a, b in row_chunks(h, w, BIG // 2):
            a0, b0 = max(0, a - 1), min(h, b + 1)
            gy, gx = np.gradient(z[a0:b0], cv.px_mm)
            gy, gx = gy[a - a0: a - a0 + (b - a)], gx[a - a0: a - a0 + (b - a)]
            n = np.stack([-gx, -gy, np.ones_like(gx)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
            n16[a:b] = np.rint((n * 0.5 + 0.5) * 65535)
            ns = np.stack([-gx * preview_exaggeration, -gy * preview_exaggeration, np.ones_like(gx)], 2)
            ns /= np.linalg.norm(ns, axis=2, keepdims=True)
            n8[a:b] = np.clip(np.rint((ns * 0.5 + 0.5) * 255), 0, 255)
            shade = np.clip((ns * light).sum(2), 0, 1)
            lit_out[a:b] = np.clip(albedo[a:b].astype(np.float32) * (1.0 + 0.45 * (shade - flat))[..., None], 0, 255)
        del z
        cv2.imwrite(str(out / "normal_16bit_physical.png"), n16[..., ::-1]); del n16
        Image.fromarray(n8, "RGB").save(out / "normal.png", **tags); del n8
        Image.fromarray(lit_out, "RGB").save(out / "artwork_preview.png", **tags); del lit_out
        meta = {
            "version": VERSION, "seed": self.seed, "canvas_px": [self.w, self.h],
            "canvas_mm": [round(self.w * cv.px_mm, 1), round(self.h * cv.px_mm, 1)],
            "dpi": dpi,
            "canvas_in": [round(self.w * cv.px_mm / 25.4, 2), round(self.h * cv.px_mm / 25.4, 2)],
            "linen_weave": {"pitch_mm": 0.7, "pitch_px": round(cv.weave_pitch_px, 2), "visible": cv.weave_gate > 0},
            "substrate": None if cv.weave is None else {**cv.sub, "name": cv.substrate if isinstance(cv.substrate, str) else "custom",
                          "weave_seed": cv.weave_seed, "pitch_px": round(cv.weave_pitch_px, 3),
                          "supersample": cv.weave_supersample},
            "pixel_pitch_mm": round(cv.px_mm, 4),
            "height_units": "mm above bare primed canvas",
            "height_encoding": {"file": "height_16bit.png", "dtype": "uint16", "scale_mm": scale_mm,
                                "decode": "height_mm = value / 65535 * scale_mm",
                                "peak_mm": round(peak, 4), "mean_mm": round(float(cv.height.mean()), 4),
                                "fraction_of_range_used": round(peak / scale_mm, 3)},
            "normal_maps": {"normal_16bit_physical.png": "true slope (strength 1.0), 16-bit",
                            "normal.png": f"8-bit, slope x {preview_exaggeration} (same strength as preview)",
                            "convention": "X right, Y down (DirectX-style); flip G for OpenGL"},
            "preview_relief_exaggeration": preview_exaggeration,
            "materials": MATERIALS,
            "stroke_counts": self.cv.counts,
            "passes": self.passes,
            "brief": self.brief,
            "pixel_transfer_from_reference": False,
            "limitations": [
                "Colour-field grammar only: one dominant field, ground, border band.",
                "Mixing is RGB alpha blending, not Kubelka-Munk pigment mixing.",
                "Wet-into-wet is approximated by brush pickup; no drying-time or flow simulation.",
                "Relief values are plausible orders of magnitude, not calibrated measurements.",
            ],
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=44315)
    ap.add_argument("--reference", help="optional reference photo; only summary statistics are used")
    ap.add_argument("--canvas-box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="painting bounds inside the reference photo (auto-detected if omitted)")
    ap.add_argument("--brief", help="load a brief JSON instead of analysing a reference")
    ap.add_argument("--width", type=int, default=850, help="output width in px")
    ap.add_argument("--size-in", type=float, nargs=2, metavar=("W", "H"),
                    help="physical canvas size in inches, e.g. 30 40 (overrides --width/--width-mm)")
    ap.add_argument("--dpi", type=float, default=300.0, help="with --size-in: output resolution")
    ap.add_argument("--width-mm", type=float, default=760.0, help="physical canvas width")
    ap.add_argument("--preview-exaggeration", type=float, default=3.0)
    ap.add_argument("--no-calibrate", action="store_true", help="skip wall-based photo white balance")
    a = ap.parse_args()
    if a.brief:
        brief = json.loads(Path(a.brief).read_text())
    elif a.reference:
        brief = brief_from_reference(a.reference, a.canvas_box, calibrate=not a.no_calibrate)
    else:
        brief = brief_from_grammar(np.random.default_rng(a.seed + 1))
    t = time.time()
    if a.size_in:
        wpx, hpx = int(round(a.size_in[0] * a.dpi)), int(round(a.size_in[1] * a.dpi))
        p = ColourFieldPainter(brief, a.seed, wpx, a.size_in[0] * 25.4, height_px=hpx).run()
        meta = p.export(a.out, a.preview_exaggeration, dpi=a.dpi)
    else:
        p = ColourFieldPainter(brief, a.seed, a.width, a.width_mm).run()
        meta = p.export(a.out, a.preview_exaggeration)
    print(f"v{VERSION} -> {a.out}  {p.w}x{p.h}px  strokes={sum(meta['stroke_counts'].values())}  "
          f"peak={meta['height_encoding']['peak_mm']}mm  {time.time() - t:.1f}s  "
          f"peak RAM {__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


if __name__ == "__main__":
    main()
