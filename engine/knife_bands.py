"""Knife-dragged band process (calibrated against 'Structures 8', oil on cotton duck, 88 x 88 cm).

A wide palette knife or scraper is pulled sideways, one band at a time, top to bottom. Each pass:
  * leaves a thin, streaky film: the blade drags a lighter, drier paint over a dark under-layer,
    so the dark shows through in long fine streaks running with the drag;
  * pushes surplus paint to the band's lower edge, where it piles into a raised lip (0.1-0.35 mm);
  * the next band starts just below that lip, so a narrow gap (an undercut slit) stays between
    them, reaching down toward the dark under-layer: the darkest values in the painting.
Bands wander gently, vary in height (10-22 mm) and the lip breaks up along its length.

paint_knife_bands(h, w, px_mm, rng, region=None, palette=None) -> (albedo float32 0..255, height mm, roughness)
Works on any region of the canvas; all sizes are physical (mm), so it is resolution independent.
"""
import math

import cv2
import numpy as np


def _noise1d(rng, n, cell_px):
    k = max(4, int(n / max(cell_px, 1)) + 4)
    g = rng.standard_normal(k).astype(np.float32)
    x = np.linspace(1, k - 3, n).astype(np.float32)
    out = np.interp(x, np.arange(k, dtype=np.float32), g)
    return cv2.GaussianBlur(out.reshape(1, -1), (0, 0), max(1.0, cell_px / 3)).ravel()


def _aniso(rng, h, w, len_px, wid_px):
    """Streak noise: long along x, thin across y (drag marks)."""
    gh, gw = max(4, int(h / max(wid_px, 0.5)) + 4), max(4, int(w / max(len_px, 1)) + 4)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    out = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return (out - out.mean()) / (out.std() + 1e-6)


DEFAULT_PALETTE = dict(under=(56, 62, 64), paint=(102, 110, 113), drag=(158, 164, 164), gap=(38, 36, 34))


def paint_knife_bands(h, w, px_mm, rng, palette=None, band_mm=(9, 24), lip_mm=(0.12, 0.55),
                      wander_mm=1.6, film_mm=0.06, under_mm=0.08):
    pal = {k: np.asarray(v, np.float32) for k, v in (palette or DEFAULT_PALETTE).items()}
    mm = 1.0 / px_mm                                      # pixels per mm
    yy = np.arange(h, dtype=np.float32)[:, None]
    # under-layer: dark, slightly mottled, fully covering the canvas
    mott = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 6 * mm)
    mott /= mott.std() + 1e-6
    albedo = np.empty((h, w, 3), np.float32); albedo[:] = pal["under"]
    albedo += (mott * 4)[..., None]
    height = np.full((h, w), under_mm, np.float32) + 0.01 * mott
    rough = np.full((h, w), 0.45, np.float32)
    # band boundaries (top to bottom): wavy lines y_k(x)
    bounds = []
    y = -rng.uniform(0, band_mm[1]) * mm
    while y < h + band_mm[1] * mm:
        y_line = (y + _noise1d(rng, w, 90 * mm) * wander_mm * mm + _noise1d(rng, w, 25 * mm) * 0.6 * wander_mm * mm
                  + _noise1d(rng, w, 8 * mm) * 0.25 * wander_mm * mm)
        bounds.append(y_line)
        y += rng.uniform(*band_mm) * mm
    fine_streak = _aniso(rng, h, w, 5 * mm, 0.15 * mm)      # short broken drag marks
    coarse_streak = _aniso(rng, h, w, 25 * mm, 1.2 * mm)    # broader drag variation
    dry = _aniso(rng, h, w, 9 * mm, 0.3 * mm)
    blotch = _aniso(rng, h, w, 14 * mm, 5 * mm)             # cloudy patches inside a band
    cloud = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 12 * mm); cloud /= cloud.std() + 1e-6
    for k in range(len(bounds) - 1):
        top, bot = bounds[k][None, :], bounds[k + 1][None, :]
        # lips come and go along the band: segments of thick ridge, then almost nothing
        seg = np.clip(0.35 + 0.9 * _noise1d(rng, w, 30 * mm) + 0.3 * _noise1d(rng, w, 9 * mm), 0.0, 1.6)
        lip_h = rng.uniform(*lip_mm) * seg[None, :]
        lip_w = rng.uniform(0.8, 2.0) * mm
        gap_w = (rng.uniform(0.1, 1.4) * mm * np.clip(seg - 0.15, 0, 1.6) ** 1.5
                 * np.clip(0.5 + 0.7 * _noise1d(rng, w, 14 * mm), 0, 2))[None, :]
        inside = ((yy >= top + gap_w) & (yy < bot)).astype(np.float32)
        if not inside.any():
            continue
        t = np.clip((yy - top) / np.maximum(bot - top, 1), 0, 1)          # 0 at band top, 1 at lip
        # film: thin, thickening slightly toward the lip; streaky along the drag
        s = 0.25 * fine_streak + 0.35 * coarse_streak + 0.5 * blotch
        film = film_mm * (0.55 + 0.45 * t) * np.clip(0.8 + 0.25 * s, 0.2, 1.6)
        d_edge = bot - yy                                              # px above the lower edge
        lip = lip_h * np.clip(1 - d_edge / (2.5 * lip_w), 0, 1) ** 1.5 * (1 - np.exp(-np.maximum(d_edge, 0) / (0.25 * lip_w)))
        band_h = (under_mm + film + lip) * inside
        height = np.where(inside > 0, band_h, height)
        # colour: paint coverage follows the streaks (blade starves in the valleys, under-layer shows)
        cover = np.clip(0.62 + 0.22 * s + 0.15 * t, 0, 1)
        col = pal["under"] * (1 - cover[..., None]) + pal["paint"] * cover[..., None]
        col = col * (1 + 0.06 * cloud)[..., None]
        drag_amt = np.clip((0.6 * dry + 0.5 * blotch - 0.8) * 0.45, 0, 0.5) * (0.5 + 0.5 * t)    # dry light paint
        col = col * (1 - drag_amt[..., None]) + pal["drag"] * drag_amt[..., None]
        col = col + (lip / max(lip_mm[1], 1e-3) * 25)[..., None]           # lip catches thicker, lighter paint
        albedo = np.where(inside[..., None] > 0, col, albedo)
        rough = np.where(inside > 0, 0.35 + 0.2 * (1 - cover), rough)
        # the undercut gap between this band's top and the previous lip
        gap = ((yy >= top) & (yy < top + gap_w)).astype(np.float32)
        gap = cv2.GaussianBlur(gap, (0, 0), 0.35 * mm)
        height = height * (1 - gap) + (under_mm * 0.4) * gap
        albedo = albedo * (1 - 0.8 * gap[..., None]) + pal["gap"] * 0.8 * gap[..., None]
    # pigment grain and grit: fine isotropic speckle, plus short marks across the drag
    grain = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.06 * mm)
    grain /= grain.std() + 1e-6
    cross = _aniso(rng, h, w, 0.12 * mm, 0.6 * mm)            # tiny marks perpendicular to the drag
    albedo *= (1 + 0.05 * grain + 0.015 * cross)[..., None]
    height += 0.004 * grain
    # dust / grit specks catching light
    sp = (rng.random((h, w)) < 0.0007).astype(np.uint8)
    sp = cv2.dilate(sp, np.ones((2, 2), np.uint8)).astype(bool)          # flecks 2-3 px (0.2 mm)
    albedo[sp] = albedo[sp] * 0.35 + 205 * 0.65
    height[sp] += 0.02
    return np.clip(albedo, 0, 255), height, rough
