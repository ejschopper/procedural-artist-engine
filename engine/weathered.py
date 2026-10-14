"""Stage 1: weathered field painter (v1.8-alpha), built on the resolution-independent print engine.

A composition is 2-4 zones (horizontal bands, or vertical panels) with a chosen boundary type
between each pair: dry-brush fringe, ragged scumble with drips, soft fog, hard line, or scored
line. The process is physical and ordered:

  1. toned ground in a contrasting under-colour            (snapshot S0 - deepest reveal)
  2. first coat per zone, often a neighbour's colour       (snapshot S1 - shallow reveal)
  3. body coats per zone, clipped to soft zone masks
  4. boundary treatment
  5. scrape-back: knife drags and worn patches remove top paint from the tooth peaks and
     reveal S1 or S0 below (a pass-level layer stack, after RC1)
  6. dry scumbles, edge darkening, sparse marks (scratches, sgraffito, tally, X, bar, drips, daubs)
  7. thin glaze

Nothing is copied from the reference library: palettes are curated schemes, every composition
is sampled from the seed. Usage:
  python weathered.py --seed 3 --out out --size-in 30 40 --dpi 150     # print-scale native
  python weathered.py --seed 3 --out out --width 850                     # quick preview
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from engine_print import (BIG, ColourFieldPainter, Noise, jitter_color, lerp_palette, row_chunks,
                          smoothstep)

VERSION = "1.8-alpha weathered-field"

# Curated schemes: zone colours (top->bottom or left->right pool), under-colour revealed by
# scraping, accent (stripes, fringes), mark colour. Hues are typical of the weathered-field family.
SCHEMES = {
    "teal_cream_rust": dict(zones=[(212, 208, 192), (63, 128, 132), (24, 78, 92)], under=(156, 88, 52),
                            accent=(176, 104, 62), mark=(44, 40, 38), light=(232, 228, 214)),
    "mustard_orange": dict(zones=[(196, 158, 62), (214, 150, 48), (226, 124, 34)], under=(52, 140, 178),
                           accent=(60, 160, 200), mark=(40, 36, 30), light=(232, 205, 120)),
    "teal_red_yellow": dict(zones=[(44, 146, 136), (184, 82, 70), (172, 74, 64)], under=(70, 50, 40),
                            accent=(232, 182, 72), mark=(36, 30, 28), light=(226, 160, 120)),
    "olive_aubergine": dict(zones=[(196, 182, 138), (176, 162, 118), (48, 38, 48)], under=(146, 84, 62),
                            accent=(150, 92, 70), mark=(30, 26, 30), light=(222, 212, 176)),
    "navy_cream": dict(zones=[(34, 52, 76), (28, 40, 60), (226, 214, 188)], under=(18, 18, 22),
                       accent=(96, 110, 128), mark=(16, 16, 20), light=(240, 232, 214)),
    "yellow_black": dict(zones=[(222, 162, 34), (28, 25, 20)], under=(168, 52, 48),
                         accent=(196, 70, 60), mark=(236, 228, 210), light=(240, 200, 90)),
    "rust_umber": dict(zones=[(216, 194, 150), (154, 88, 42), (206, 176, 128), (84, 62, 46)],
                       under=(60, 46, 36), accent=(170, 96, 46), mark=(40, 32, 26), light=(238, 226, 196)),
    "blue_grey_orange": dict(zones=[(206, 192, 166), (98, 116, 132), (102, 122, 112)], under=(222, 94, 42),
                             accent=(230, 108, 50), mark=(30, 30, 32), light=(232, 222, 200)),
    "greygreen_cream": dict(zones=[(92, 98, 88), (206, 196, 182)], under=(60, 58, 52),
                            accent=(70, 70, 66), mark=(34, 34, 34), light=(232, 226, 214)),
}
BOUNDARY_TYPES = ["fringe", "ragged", "fog", "hard", "scored"]


def aniso_noise(rng, h, w, cell_x, cell_y):
    """Noise with different feature sizes along x and y (e.g. stretched along a band)."""
    n = Noise.__new__(Noise)
    n.h, n.w = h, w
    n.a = rng.normal(size=(max(2, int(math.ceil(h / cell_y)) + 2), max(2, int(math.ceil(w / cell_x)) + 2))).astype(np.float32)
    n._full = None
    sw = min(w, 1024); sh = max(2, int(round(h * sw / w)))
    small = cv2.resize(n.a, (sw, sh), interpolation=cv2.INTER_CUBIC)
    n.mu, n.sd = float(small.mean()), float(small.std()) + 1e-6
    return n


def smooth_tooth(cv, a, b, sigma=1.2):
    """Tooth with sub-weave detail removed, for thresholds that must not alias into pixel blocks."""
    a0, b0 = max(0, a - 4), min(cv.h, b + 4)
    t = cv2.GaussianBlur(cv.tooth[a0:b0], (0, 0), sigma)
    return t[a - a0: a - a0 + (b - a)]


# --------------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------------
def make_plan(rng, scheme=None, orientation=None):
    name = scheme or list(SCHEMES)[int(rng.integers(len(SCHEMES)))]
    sc = SCHEMES[name]
    orient = orientation or ("horizontal" if rng.random() < 0.8 else "vertical")
    n = min(int(rng.choice([2, 2, 3, 3, 3, 4])), len(sc["zones"]))
    # zone colours: keep scheme order but allow a light/dark swap for variety
    pool = list(sc["zones"])
    def lum_gap(cs):
        L = [0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2] for c in cs]
        dist = [np.abs(np.subtract(cs[i], cs[i + 1])).sum() for i in range(len(cs) - 1)]
        return min(dist) if dist else 0
    best = None
    for _ in range(30):   # adjacent zones must read as different fields
        idx = sorted(rng.choice(len(pool), n, replace=False).tolist())
        cand = [pool[i] for i in idx]
        if best is None or lum_gap(cand) > lum_gap(best):
            best = cand
        if lum_gap(best) > 120:
            break
    cols = best
    if rng.random() < 0.3:
        cols = cols[::-1]
    # sizes: asymmetric partition; a 3- or 4-zone plan may get one thin band
    w = rng.dirichlet(np.ones(n) * 2.2)
    if n >= 3 and rng.random() < 0.5:
        k = int(rng.integers(1, n - 1)) if n > 2 else 1
        w[k] = rng.uniform(0.06, 0.12)
    w = np.clip(w, 0.06, None); w = w / w.sum()
    cuts = np.cumsum(w)[:-1].tolist()
    soft_scheme = name in ("navy_cream", "rust_umber")
    bts = []
    for i in range(n - 1):
        choices = ["fringe", "ragged", "fog", "hard", "scored"]
        p = np.array([0.25, 0.3, 0.2, 0.12, 0.13]) if not soft_scheme else np.array([0.15, 0.15, 0.55, 0.05, 0.1])
        bts.append(str(rng.choice(choices, p=p / p.sum())))
    accent_stripe = (int(rng.integers(n - 1)) if (n >= 2 and rng.random() < 0.45) else None)
    marks = dict(
        scratches=int(rng.integers(6, 26)),
        sgraffito=int(rng.integers(0, 2)),
        tally=int(rng.random() < 0.35),
        x_mark=int(rng.random() < 0.25),
        bar=int(rng.random() < 0.3),
        drips=int(rng.integers(0, 14)) if rng.random() < 0.6 else 0,
        daubs=int(rng.integers(0, 6)),
    )
    return dict(source="weathered_field grammar", scheme=name, orientation=orient, zones=len(cols),
                zone_colors=[list(map(int, c)) for c in cols], cuts=[round(c, 4) for c in cuts],
                boundaries=bts, accent_stripe=accent_stripe,
                edge_darkening=round(float(rng.uniform(0.0, 0.32)), 3),
                scrape_strength=round(float(rng.uniform(0.35, 1.0)), 3),
                marks=marks, aspect_w_over_h=0.75)


# --------------------------------------------------------------------------------------------
# Geometry helpers: boundaries are 1-D curves along the "u" axis, zones lie between them on "t"
# --------------------------------------------------------------------------------------------
class Zones:
    def __init__(self, painter):
        p = painter; r = p.rng; s = p.s
        self.vertical = p.plan["orientation"] == "vertical"
        self.U = p.h if self.vertical else p.w      # length along boundaries
        self.T = p.w if self.vertical else p.h      # split axis
        self.n = p.plan["zones"]
        self.b = []        # boundary position along t for each u (float32 arrays)
        self.soft = []     # transition half-width (px)
        u = np.arange(self.U, dtype=np.float32)
        for k, (cut, bt) in enumerate(zip(p.plan["cuts"], p.plan["boundaries"])):
            slow = Noise(r, 1, self.U, 260 * s).full()[0] * 6 * s
            mid = Noise(r, 1, self.U, 40 * s).full()[0]
            fine = Noise(r, 1, self.U, 11 * s).full()[0]
            pos = cut * self.T + slow
            if bt == "ragged":
                pos = pos + (6 * mid + 4 * fine) * s
                soft = 2.0 * s
            elif bt == "fringe":
                pos = pos + (1.5 * mid) * s
                soft = 2.5 * s
            elif bt == "fog":
                soft = 26 * s
            elif bt == "hard":
                pos = cut * self.T + slow * 0.25 + 0.6 * s * fine
                soft = 1.0 * s
            else:  # scored
                pos = cut * self.T + slow * 0.3 + 0.8 * s * fine
                soft = 1.4 * s
            self.b.append(pos.astype(np.float32)); self.soft.append(float(soft))

    def tu(self, ya, yb, xa, xb):
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        if self.vertical:
            return xx, yy.astype(np.int32)
        return yy, xx.astype(np.int32)

    def mask(self, k, grow=0.0):
        return ZoneMask(self, k, grow)


class ZoneMask:
    def __init__(self, zones, k, grow=0.0):
        self.z, self.k, self.grow = zones, k, grow

    def get(self, ya, yb, xa, xb):
        t, ui = self.z.tu(ya, yb, xa, xb)
        m = np.ones(t.shape, np.float32)
        if self.k > 0:
            b, sft = self.z.b[self.k - 1][ui], self.z.soft[self.k - 1]
            m *= smoothstep(-sft, sft, t - (b - self.grow))
        if self.k < self.z.n - 1:
            b, sft = self.z.b[self.k][ui], self.z.soft[self.k]
            m *= 1 - smoothstep(-sft, sft, t - (b + self.grow))
        return m


class BandMask:
    """Soft band of half-width hw (px) centred on boundary k (with optional ragged noise)."""

    def __init__(self, zones, k, hw, offset=0.0, ragged=None):
        self.z, self.k, self.hw, self.off, self.rag = zones, k, hw, offset, ragged

    def get(self, ya, yb, xa, xb):
        t, ui = self.z.tu(ya, yb, xa, xb)
        d = t - (self.z.b[self.k][ui] + self.off)
        hw = self.hw if self.rag is None else self.hw * (1 + self.rag[ui])
        return 1 - smoothstep(0.6 * hw, hw, np.abs(d))


# --------------------------------------------------------------------------------------------
# Painter
# --------------------------------------------------------------------------------------------
class WeatheredPainter(ColourFieldPainter):
    def __init__(self, plan, seed, width_px=850, width_mm=762.0, height_px=None):
        plan = dict(plan)
        super().__init__(plan, seed, width_px, width_mm, height_px)
        self.plan = plan
        self.sc = SCHEMES[plan["scheme"]]
        self.Z = Zones(self)
        self.S0 = self.S1 = None

    # direction helpers -----------------------------------------------------------------------
    def _along(self):
        return (math.pi / 2) if self.Z.vertical else 0.0

    def _pt(self, u, t):
        return (t, u) if self.Z.vertical else (u, t)

    def _zone_range(self, k):
        T = self.Z.T
        lo = 0 if k == 0 else float(np.median(self.Z.b[k - 1]))
        hi = T if k == self.Z.n - 1 else float(np.median(self.Z.b[k]))
        return lo, hi

    def _snapshot(self):
        snap = np.empty((self.h, self.w, 3), np.uint8)
        for a, b in row_chunks(self.h, self.w):
            snap[a:b] = np.clip(np.rint(self.cv.color[a:b]), 0, 255)
        return snap

    # passes ------------------------------------------------------------------------------------
    def ground(self):
        r, w, h, s = self.rng, self.w, self.h, self.s
        under = np.asarray(self.sc["under"], np.float32)
        m1, m2 = Noise(r, h, w, 140 * s), Noise(r, h, w, 30 * s)
        for a, b in row_chunks(h, w):
            mott = m1.crop(a, b, 0, w) * 9 + m2.crop(a, b, 0, w) * 4
            self.cv.color[a:b] = np.clip(under[None, None, :] + mott[..., None] * np.array([1, .9, .8]), 0, 255)
        self.cv.height += 0.004
        # rough, broken first scrub so the ground is never a flat fill
        for _ in range(60):
            x, y = r.uniform(-0.1, 1.1) * w, r.uniform(-0.05, 1.05) * h
            ang = r.uniform(0, math.pi); L = r.uniform(200, 520) * s
            self.cv.stroke((x, y), (x + L * math.cos(ang), y + L * math.sin(ang)), r.uniform(60, 160) * s,
                           jitter_color(r, under * r.uniform(0.85, 1.12), 6), load=1.2, thickness=0.006,
                           decay=1.2, dry=0.2, opacity=0.5, material="imprimatura", streak=0.6)
        self.S0 = self._snapshot()

    def first_coat(self):
        """Each zone is first blocked in with a neighbour-tinted colour (the shallow reveal layer)."""
        r, s = self.rng, self.s
        cols = [np.asarray(c, np.float32) for c in self.plan["zone_colors"]]
        for k in range(self.Z.n):
            nb = cols[k + 1] if k + 1 < len(cols) else cols[k - 1]
            c = cols[k] * 0.45 + nb * 0.35 + np.asarray(self.sc["accent"], np.float32) * 0.20
            self._fill_zone(k, c, count=260, opacity=(0.55, 0.85), dry=0.25, thickness=0.02, jit=10,
                            material="field_under", grow=10 * s)
        self.S1 = self._snapshot()

    def _fill_zone(self, k, color, count, opacity, dry, thickness, jit, material, grow=0.0,
                   width=(40, 130), length=(120, 420), cross=0.25, load=(0.8, 1.25), streak=0.5):
        r, s = self.rng, self.s
        lo, hi = self._zone_range(k)
        frac = max(0.05, (hi - lo) / self.Z.T)
        n = int(count * frac * self.k * 1.6)
        clip = self.Z.mask(k, grow)
        along = self._along()
        for _ in range(n):
            u = r.uniform(-0.05, 1.05) * self.Z.U
            t = r.uniform(lo - grow, hi + grow)
            L = r.uniform(*length) * s
            ang = along + (math.pi / 2 if r.random() < cross else 0) + r.normal(0, 0.06)
            if r.random() < 0.5:
                ang += math.pi
            x, y = self._pt(u, t)
            cc = np.asarray(color, np.float32) + self.drift_at(y, x) * np.array([1, .95, .85])
            self.cv.stroke((x, y), (x + L * math.cos(ang), y + L * math.sin(ang)), r.uniform(*width) * s,
                           jitter_color(r, cc, jit), load=r.uniform(*load), thickness=thickness,
                           decay=r.uniform(0.4, 1.0), dry=dry, pickup=0.3, opacity=r.uniform(*opacity),
                           bend=r.normal(0, 6) * s, rough=0.55, material=material, clip=clip, streak=streak)

    def body(self):
        cols = [np.asarray(c, np.float32) for c in self.plan["zone_colors"]]
        for k in range(self.Z.n):
            c = cols[k]
            self._fill_zone(k, c, count=520, opacity=(0.4, 0.8), dry=0.45, thickness=0.02, jit=7,
                            material="field")
            # tonal mottling: broad patches of darker / lighter / shifted versions of the zone colour
            for var in (c * 0.72, c * 1.15 + 10, c * 0.85 + np.asarray(self.sc["under"], np.float32) * 0.15):
                self._fill_zone(k, np.clip(var, 0, 255), count=130, opacity=(0.25, 0.5), dry=0.5,
                                thickness=0.01, jit=8, material="field", width=(60, 180), length=(80, 260),
                                cross=0.4, streak=0.7)
            # second, drier coat lets the first coat and ground flicker through
            self._fill_zone(k, c * 1.04 + 3, count=260, opacity=(0.25, 0.55), dry=0.75, thickness=0.012,
                            jit=6, material="field", width=(30, 90), streak=0.9)

    def mottle(self):
        """Uneven absorption / patchy coverage at 1-4 cm: multiplies the paint, not the ground."""
        r, s, w, h = self.rng, self.s, self.w, self.h
        v = self.Z.vertical
        mk = lambda along, across: aniso_noise(r, h, w, (across if v else along) * s, (along if v else across) * s)
        n1, n2, n3 = mk(120, 30), mk(40, 10), mk(12, 4)
        for a, b in row_chunks(h, w):
            m = 1 + 0.05 * n1.crop(a, b, 0, w) + 0.04 * n2.crop(a, b, 0, w) + 0.025 * n3.crop(a, b, 0, w)
            self.cv.color[a:b] = np.clip(self.cv.color[a:b] * m[..., None], 0, 255)

    def boundaries(self):
        r, s = self.rng, self.s
        cols = [np.asarray(c, np.float32) for c in self.plan["zone_colors"]]
        along = self._along()
        sgn = 1.0
        for k, bt in enumerate(self.plan["boundaries"]):
            a_col, b_col = cols[k], cols[k + 1]
            b = self.Z.b[k]
            if self.plan["accent_stripe"] == k:
                rag = Noise(r, 1, self.Z.U, 30 * s).full()[0] * 0.5
                hw = r.uniform(5, 13) * s
                band = BandMask(self.Z, k, hw, offset=r.uniform(-4, 4) * s, ragged=rag)
                acc = np.asarray(self.sc["accent"], np.float32)
                for _ in range(int(110 * self.k)):
                    u = r.uniform(-0.05, 1.0) * self.Z.U; L = r.uniform(80, 300) * s
                    t = float(b[int(np.clip(u, 0, self.Z.U - 1))])
                    x, y = self._pt(u, t)
                    self.cv.stroke((x, y), (x + L * math.cos(along), y + L * math.sin(along)), hw * 2.6,
                                   jitter_color(r, acc, 10), load=r.uniform(0.8, 1.3), thickness=0.03, dry=0.5,
                                   opacity=r.uniform(0.6, 0.95), material="edge_accent", clip=band, streak=1.0)
            if bt in ("fringe", "ragged"):
                # dry drags across the boundary in both directions (upper colour pulled down, etc.)
                n = int((260 if bt == "fringe" else 180) * self.k)
                depth = r.uniform(12, 30) * s
                for _ in range(n):
                    u = r.uniform(0, 1) * self.Z.U; ui = int(np.clip(u, 0, self.Z.U - 1))
                    down = r.random() < 0.6
                    col, d0 = (a_col, -r.uniform(0, 8) * s) if down else (b_col, r.uniform(0, 8) * s)
                    L = r.uniform(0.3, 1.4) * depth
                    t0 = float(b[ui]) + d0; t1 = t0 + (L if down else -L)
                    p0, p1 = self._pt(u, t0), self._pt(u + r.normal(0, 1.5) * s, t1)
                    self.cv.stroke(p0, p1, r.uniform(4, 18) * s, jitter_color(r, col, 8), load=r.uniform(0.5, 0.95),
                                   thickness=0.01, decay=0.45, dry=0.75, pickup=0.2, opacity=r.uniform(0.35, 0.8),
                                   fray=0.75, streak=1.0, material="dry_fringe")
            if bt == "fog":
                # wide translucent veils of both colours across the boundary
                for _ in range(int(70 * self.k)):
                    u = r.uniform(-0.1, 1.0) * self.Z.U; ui = int(np.clip(u, 0, self.Z.U - 1))
                    t = float(b[ui]) + r.normal(0, 22) * s
                    L = r.uniform(200, 500) * s
                    col = a_col if r.random() < 0.5 else b_col
                    x, y = self._pt(u, t)
                    fa = along + r.normal(0, 0.035)
                    self.cv.stroke((x, y), (x + L * math.cos(fa), y + L * math.sin(fa)), r.uniform(30, 70) * s,
                                   jitter_color(r, col * 0.5 + (a_col + b_col) * 0.25, 6), load=1.4, thickness=0.004,
                                   decay=2.0, dry=0.15, opacity=r.uniform(0.08, 0.2), material="glaze", streak=0.2)
            if bt == "scored":
                self._line_along(k, offset=r.uniform(-2, 2) * s, width=1.2 * s, kind="score")
            if bt == "hard" and r.random() < 0.5:
                self._line_along(k, offset=0, width=1.6 * s, kind="paint", col=self.sc["mark"])
            if bt == "ragged":
                self._drips_from(k, int(r.integers(4, 16)), a_col)

    def _line_along(self, k, offset, width, kind, col=None):
        """A single drawn/incised line that follows boundary k."""
        r = self.rng
        pts = []
        b = self.Z.b[k]
        step = max(4, int(self.Z.U / 400))
        for u in range(0, self.Z.U, step):
            t = float(b[u]) + offset + r.normal(0, 0.3) * self.s
            pts.append(self._pt(u, t))
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.polylines(m, [np.int32(np.rint(pts))], False, 255, max(1, int(round(width))), cv2.LINE_AA)
        # broken along its length, like a scored or drawn line catching the tooth
        self._apply_line_mask(m, kind, col)

    def _apply_line_mask(self, m, kind, col=None, strength=1.0):
        cv = self.cv
        brk = Noise(self.rng, self.h, self.w, 25 * self.s)
        col = np.asarray(col if col is not None else self.sc["mark"], np.float32)
        for a, b in row_chunks(self.h, self.w):
            mm = m[a:b].astype(np.float32) / 255 * strength
            if not mm.any():
                continue
            mm *= np.clip(0.75 + 0.5 * brk.crop(a, b, 0, self.w), 0, 1)
            if kind == "score":   # incised: removes paint, reveals the shallow layer, cuts height
                rev = self.S1[a:b].astype(np.float32) * 0.6 + self.S0[a:b].astype(np.float32) * 0.4
                cv.color[a:b] = cv.color[a:b] * (1 - mm[..., None]) + rev * mm[..., None]
                cv.height[a:b] = np.maximum(0.004, cv.height[a:b] - 0.03 * mm)
                cv.material[a:b][mm > 0.4] = 4
            elif kind == "light":  # scratch exposing pale under-layer
                light = np.asarray(self.sc["light"], np.float32)
                rev = self.S0[a:b].astype(np.float32) * 0.5 + light * 0.5
                cv.color[a:b] = cv.color[a:b] * (1 - mm[..., None]) + rev * mm[..., None]
                cv.height[a:b] = np.maximum(0.004, cv.height[a:b] - 0.02 * mm)
            else:                  # drawn (charcoal/paint) line
                tooth = cv.tooth[a:b]
                mm = mm * smoothstep(0.25, 0.6, tooth)
                cv.color[a:b] = cv.color[a:b] * (1 - 0.85 * mm[..., None]) + col * 0.85 * mm[..., None]
                cv.height[a:b] += 0.004 * mm

    def _drips_from(self, k, n, col):
        r, s = self.rng, self.s
        if self.Z.vertical:
            return  # drips run with gravity; skip on vertical splits
        b = self.Z.b[k]
        m = np.zeros((self.h, self.w), np.uint8)
        for _ in range(n):
            x = r.uniform(0.03, 0.97) * self.w; y = float(b[int(x)]) - r.uniform(0, 4) * s
            L = r.uniform(25, 140) * s; wd = r.uniform(1.5, 4.5) * s
            ys = np.linspace(y, y + L, 24); xs = x + np.cumsum(r.normal(0, 0.25, 24)) * s
            for i in range(23):
                th = max(1, int(round(wd * (1 - 0.6 * i / 23))))
                cv2.line(m, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), 255, th, cv2.LINE_AA)
            cv2.circle(m, (int(xs[-1]), int(ys[-1])), max(1, int(wd * 0.7)), 255, -1, cv2.LINE_AA)
        cv = self.cv; col = np.asarray(col, np.float32)
        for a, bb in row_chunks(self.h, self.w):
            mm = m[a:bb].astype(np.float32) / 255
            if mm.any():
                cv.color[a:bb] = cv.color[a:bb] * (1 - 0.9 * mm[..., None]) + col * 0.9 * mm[..., None]
                cv.height[a:bb] += 0.03 * mm
                cv.material[a:bb][mm > 0.5] = 7

    def scrape_back(self):
        """Knife drags + worn patches. Removes top paint from the tooth peaks and reveals S1/S0."""
        r, s, w, h = self.rng, self.s, self.w, self.h
        strength = self.plan["scrape_strength"]
        M = np.zeros((h, w), np.uint8)
        along = self._along()
        for _ in range(int(32 * strength * self.k)):
            u = r.uniform(-0.1, 1.0) * self.Z.U
            k = int(r.integers(self.Z.n - 1)) if (self.Z.n > 1 and r.random() < 0.55) else None
            t = (float(self.Z.b[k][int(np.clip(u, 0, self.Z.U - 1))]) + r.normal(0, 30) * s) if k is not None \
                else r.uniform(0.03, 0.97) * self.Z.T
            L = r.uniform(60, 420) * s
            x0, y0 = self._pt(u, t)
            ang = along + r.normal(0, 0.05)
            x1, y1 = x0 + L * math.cos(ang), y0 + L * math.sin(ang)
            # a knife smear tapers in at the start and lifts off at the end
            # one smooth tapered polygon (continuous width), sub-pixel accurate
            wmax = r.uniform(10, 36) * s; val = int(r.uniform(110, 230))
            f = np.linspace(0, 1, 48)
            env = np.sin(np.pi * np.clip(f * 1.05 + 0.02, 0, 1)) ** 0.6
            wob = np.cumsum(r.normal(0, 0.15, 48)) * s
            cx_, cy_ = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            nx_, ny_ = -math.sin(ang), math.cos(ang)
            hw = wmax * env / 2
            top_ = np.stack([cx_ + nx_ * (hw + wob), cy_ + ny_ * (hw + wob)], 1)
            bot_ = np.stack([cx_ - nx_ * (hw - wob), cy_ - ny_ * (hw - wob)], 1)
            poly = np.concatenate([top_, bot_[::-1]])
            cv2.fillPoly(M, [np.int32(np.rint(poly * 16))], val, cv2.LINE_AA, shift=4)
        M = cv2.GaussianBlur(M, (0, 0), max(1.0, 3.0 * s))
        # streaks along the drag (paint skips where the blade rides over ridges)
        cs, cl = max(2, int(h / (1.5 * s))), max(2, int(w / (60 * s)))
        st = r.normal(size=(cs, cl) if not self.Z.vertical else (cl, cs)).astype(np.float32)
        streak_g = st
        low = Noise(r, h, w, 220 * s); patch = Noise(r, h, w, 28 * s); fl2 = Noise(r, h, w, 8 * s)
        fl3 = Noise(r, h, w, 3 * s); grit = Noise(r, h, w, 2.5 * s); depthn = Noise(r, h, w, 90 * s)
        pit = Noise(r, h, w, 1.1 * s)
        edge_bias = 0.6 + 1.2 * self.plan["edge_darkening"]
        cv = self.cv
        for a, b in row_chunks(h, w):
            yy, xx = np.mgrid[a:b, 0:w].astype(np.float32)
            dedge = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / (0.06 * w)
            t, ui = self.Z.tu(a, b, 0, w)
            dbound = np.full(t.shape, 1e9, np.float32)
            for bk in self.Z.b:
                dbound = np.minimum(dbound, np.abs(t - bk[ui]))
            near = np.exp(-dbound / (40 * s))
            # where wear happens: broad zones, the canvas edges and the zone boundaries
            A = smoothstep(0.3, 1.6, 0.8 * low.crop(a, b, 0, w) + edge_bias * np.exp(-dedge) + 0.7 * near
                           + (strength - 0.6))
            g = grit.crop(a, b, 0, w)
            tz = smooth_tooth(cv, a, b, max(1.0, 0.35 * s))
            peaks = smoothstep(0.35, 0.85, tz + 0.12 * g)
            cluster = smoothstep(-0.2, 0.9, patch.crop(a, b, 0, w))     # abrasion comes in patches
            sanded = A * peaks * cluster * 0.6
            fn = 0.65 * patch.crop(a, b, 0, w) + 0.25 * fl2.crop(a, b, 0, w) + 0.1 * fl3.crop(a, b, 0, w)
            fv = fn + 1.1 * A + 0.06 * fl3.crop(a, b, 0, w)
            flakes = smoothstep(1.47, 1.485, fv) * smoothstep(0.05, 0.25, A)
            # torn paint lip: thin raised, slightly lighter rim just outside each flake
            rim = np.clip(smoothstep(1.40, 1.465, fv) - flakes, 0, 1) * smoothstep(0.05, 0.25, A)
            gh, gw = streak_g.shape
            mx = (np.arange(w, dtype=np.float32) + 0.5) * gw / w - 0.5
            my = (np.arange(a, b, dtype=np.float32) + 0.5) * gh / h - 0.5
            MX, MY = np.meshgrid(mx, my)
            stk = cv2.remap(streak_g, MX, MY, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            knife = (M[a:b].astype(np.float32) / 255 * smoothstep(0.2, 0.7, tz + 0.1 * g)
                     * smoothstep(-0.4, 0.8, stk))
            # granular speck erosion (pits down to the ground), densest where wear is strongest
            # pits: coarse clusters gate fine, crisp, irregular holes (not round droplets)
            fine_pit = 0.6 * pit.crop(a, b, 0, w) + 0.4 * g
            spk = (smoothstep(1.06, 1.12, fine_pit + 1.0 * A - 0.55)
                   * smoothstep(-0.3, 0.5, fl3.crop(a, b, 0, w) + A))
            amt = np.clip(np.maximum.reduce([sanded, flakes, knife, spk]) * min(1.0, 0.5 + strength), 0, 1)
            deep = smoothstep(0.9, 1.6, depthn.crop(a, b, 0, w) + 0.6 * flakes + 0.4 * knife) * 0.8
            rev = self.S1[a:b].astype(np.float32) * (1 - deep[..., None]) + self.S0[a:b].astype(np.float32) * deep[..., None]
            cv.color[a:b] = cv.color[a:b] * (1 - amt[..., None]) + rev * amt[..., None]
            cv.height[a:b] = np.maximum(0.006, cv.height[a:b] - amt * (0.012 + 0.02 * deep) - flakes * 0.03)
            cv.height[a:b] += rim * 0.012
            cv.color[a:b] = np.clip(cv.color[a:b] * (1 + 0.06 * rim[..., None]), 0, 255)
            # shadow on the exposed side of a flake (paint wall), lit from top-left
            cv.color[a:b] *= (1 - 0.18 * flakes * (1 - smoothstep(0.0, 0.6, flakes)) - 0.05 * flakes)[..., None]
            cv.rough[a:b] = cv.rough[a:b] * (1 - amt) + 0.8 * amt
            cv.material[a:b][amt > 0.5] = 4
        del M

    def scumble(self):
        r, s = self.rng, self.s
        cols = [np.asarray(c, np.float32) for c in self.plan["zone_colors"]]
        light = np.asarray(self.sc["light"], np.float32)
        for k in range(self.Z.n):
            c = cols[k] * 0.6 + light * 0.4 if r.random() < 0.6 else cols[k] * 0.9
            self._fill_zone(k, c, count=150, opacity=(0.12, 0.32), dry=0.9, thickness=0.008, jit=6,
                            material="scumble", width=(18, 70), length=(60, 260), cross=0.4,
                            load=(0.35, 0.7), streak=0.9)

    def edge_darkening(self):
        amt = self.plan["edge_darkening"]
        if amt <= 0.01:
            return
        r, s, w, h = self.rng, self.s, self.w, self.h
        nz = Noise(r, h, w, 50 * s); nz2 = Noise(r, h, w, 30 * s); self._grit = Noise(r, h, w, 2.5 * s)
        dark = np.asarray(self.sc["mark"], np.float32) * 0.6 + np.asarray(self.sc["under"], np.float32) * 0.4
        cv = self.cv
        for a, b in row_chunks(h, w):
            yy, xx = np.mgrid[a:b, 0:w].astype(np.float32)
            d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
            e = np.exp(-d / (np.clip(0.035 + 0.012 * nz.crop(a, b, 0, w), 0.012, None) * w)) * (0.7 + 0.3 * nz2.crop(a, b, 0, w))
            gr = self._grit.crop(a, b, 0, w)
            e = np.clip(e * amt * 2.4, 0, 0.9) * smoothstep(-0.4, 0.6, gr + 1.4 * e - 0.5)
            cv.color[a:b] = cv.color[a:b] * (1 - e[..., None]) + dark * e[..., None]

    def marks(self):
        r, s, w, h = self.rng, self.s, self.w, self.h
        mk = self.plan["marks"]
        # scratches: long thin, slightly curved, mostly near-vertical or along the bands
        m = np.zeros((h, w), np.uint8)
        for _ in range(mk["scratches"]):
            x0, y0 = r.uniform(0.04, 0.96) * w, r.uniform(0.04, 0.96) * h
            ang = (math.pi / 2 if r.random() < 0.55 else 0) + r.normal(0, 0.18)
            L = r.uniform(40, 260) * s
            pts = [(x0 + L * t * math.cos(ang) + r.normal(0, 0.8) * s * t * 10,
                    y0 + L * t * math.sin(ang)) for t in np.linspace(0, 1, 12)]
            cv2.polylines(m, [np.int32(pts)], False, int(r.uniform(140, 255)), max(1, int(0.9 * s)), cv2.LINE_AA)
        # sgraffito cluster: a scribbled zigzag scratched through the paint
        for _ in range(mk["sgraffito"]):
            cx, cy = r.uniform(0.15, 0.85) * w, r.uniform(0.1, 0.5) * h
            wd, ht = r.uniform(60, 140) * s, r.uniform(40, 110) * s
            for line in range(int(r.integers(2, 4))):
                xs = np.linspace(cx, cx + wd, int(r.integers(5, 9)))
                ys = [cy + (ht if i % 2 == 0 else 0) + r.normal(0, 8) * s + line * 6 * s for i in range(len(xs))]
                cv2.polylines(m, [np.int32(np.stack([xs + r.normal(0, 3, len(xs)) * s, ys], 1))], False, 230,
                              max(1, int(1.1 * s)), cv2.LINE_AA)
        self._apply_line_mask(m, "light")
        # drawn marks: tally, X, a short dark bar, daubs (dry charcoal / paint)
        dark = self.sc["mark"]
        if mk["tally"]:
            x0, y0 = r.uniform(0.6, 0.88) * w, r.uniform(0.15, 0.85) * h
            for i in range(int(r.integers(4, 7))):
                x = x0 + i * r.uniform(7, 11) * s
                self.cv.stroke((x, y0), (x + r.normal(0, 1) * s, y0 + r.uniform(22, 34) * s), r.uniform(2.5, 4) * s,
                               jitter_color(r, self.sc["accent"], 10), load=0.9, thickness=0.02, dry=0.6,
                               opacity=0.9, material="edge_accent", streak=1.0)
        if mk["x_mark"]:
            cx, cy = r.uniform(0.15, 0.4) * w, r.uniform(0.06, 0.2) * h; d = r.uniform(18, 30) * s
            for sg in (1, -1):
                self.cv.stroke((cx - d, cy - sg * d), (cx + d, cy + sg * d), r.uniform(9, 13) * s, jitter_color(r, dark, 4),
                               load=1.0, thickness=0.02, dry=0.55, opacity=0.95, material="edge_accent", streak=1.0)
        if mk["bar"]:
            x, y = r.uniform(0.25, 0.7) * w, r.uniform(0.2, 0.55) * h
            for _ in range(2):
                self.cv.stroke((x, y), (x + r.normal(0, 3) * s, y + r.uniform(90, 160) * s), r.uniform(7, 12) * s,
                               jitter_color(r, dark, 6), load=r.uniform(0.75, 1.0), thickness=0.025, dry=0.6,
                               opacity=0.85, fray=0.5, material="edge_accent", streak=0.9)
        for _ in range(mk["daubs"]):
            x, y = r.uniform(0.08, 0.92) * w, r.uniform(0.08, 0.92) * h
            col = self.sc["accent"] if r.random() < 0.6 else self.sc["under"]
            L = r.uniform(6, 20) * s
            self.cv.stroke((x, y), (x + L, y + r.normal(0, 2) * s), r.uniform(6, 16) * s, jitter_color(r, col, 10),
                           load=1.1, thickness=0.04, dry=0.5, opacity=0.85, material="edge_accent")
        if mk["drips"] and not self.Z.vertical:
            k = int(r.integers(self.Z.n - 1)) if self.Z.n > 1 else 0
            if self.Z.n > 1:
                self._drips_from(k, mk["drips"], np.asarray(self.plan["zone_colors"][k], np.float32) * 0.9)

    def glaze_pass(self):
        r, s = self.rng, self.s
        for _ in range(18):
            x, y = r.uniform(0.05, 0.95) * self.w, r.uniform(0.05, 0.95) * self.h
            L = r.uniform(200, 500) * s
            ang = self._along() + r.normal(0, 0.1)
            col = np.asarray(self.sc["under"], np.float32) * 0.3 + np.asarray(self.sc["light"], np.float32) * 0.7
            self.cv.stroke((x, y), (x + L * math.cos(ang), y + L * math.sin(ang)), r.uniform(80, 200) * s,
                           jitter_color(r, col, 5), load=1.5, thickness=0.002, decay=2.0, dry=0.05,
                           opacity=0.05, material="glaze", streak=0.2)

    def run(self):
        for name, fn in [("ground", self.ground), ("first_coat", self.first_coat), ("body", self.body),
                         ("mottle", self.mottle), ("boundaries", self.boundaries), ("scrape_back", self.scrape_back),
                         ("scumble", self.scumble), ("edge_darkening", self.edge_darkening),
                         ("marks", self.marks), ("glaze", self.glaze_pass)]:
            self._pass(name, fn)
        self.S0 = self.S1 = None
        return self


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out_weathered")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scheme", choices=list(SCHEMES))
    ap.add_argument("--orientation", choices=["horizontal", "vertical"])
    ap.add_argument("--width", type=int, default=850)
    ap.add_argument("--size-in", type=float, nargs=2, metavar=("W", "H"))
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--preview-exaggeration", type=float, default=3.0)
    a = ap.parse_args()
    plan = make_plan(np.random.default_rng(a.seed + 7919), a.scheme, a.orientation)
    t = time.time()
    if a.size_in:
        wpx, hpx = int(round(a.size_in[0] * a.dpi)), int(round(a.size_in[1] * a.dpi))
        plan["aspect_w_over_h"] = a.size_in[0] / a.size_in[1]
        p = WeatheredPainter(plan, a.seed, wpx, a.size_in[0] * 25.4, height_px=hpx).run()
        meta = p.export(a.out, a.preview_exaggeration, dpi=a.dpi)
    else:
        p = WeatheredPainter(plan, a.seed, a.width, 762.0, height_px=int(round(a.width / 0.75))).run()
        meta = p.export(a.out, a.preview_exaggeration)
    meta["version"] = VERSION
    meta["plan"] = plan
    meta["limitations"] = ["Pass-level reveal layers (S0 ground, S1 first coat), not per-stroke strata.",
                           "RGB mixing; relief values are plausible, not calibrated.",
                           "Palettes are curated schemes, not sampled from the reference library."]
    Path(a.out, "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    import resource
    print(f"{VERSION} seed={a.seed} scheme={plan['scheme']} {plan['orientation']} zones={plan['zones']} "
          f"boundaries={plan['boundaries']} -> {a.out} {p.w}x{p.h} {time.time() - t:.1f}s "
          f"RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB")


if __name__ == "__main__":
    main()
