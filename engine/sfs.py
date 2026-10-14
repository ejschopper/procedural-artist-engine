"""Single-light shape-from-shading for low-relief paint, with a known light direction.

Model (low relief, grazing-ish light, Lambertian):  I = A * (c0 + c1 * dh/dl)
where l is the image direction pointing toward the light (here: light from the bottom -> +y), A is the
albedo (slowly varying + paint colour), and dh/dl is the surface slope toward the light.

1. Shading signal  s = I / lowpass(I) - 1        (removes albedo at scales > ~3 mm)
2. Slope           dh/dl = k * s                 (k from the light elevation)
3. Height          Poisson-integrate the slope (only the component toward the light is observed;
                   the perpendicular slope is set to 0 and the solution regularised), giving relative
                   height; scaled to mm with `scale_mm` (peak-to-peak), later from a measured reference.
4. De-lit albedo   A = I / (1 + s_shading)       (shading removed, colour kept)

Labelled ESTIMATED: shading and dark paint are not fully separable from one photo; the 4-light
captures replace this with measured data.
"""
import math

import cv2
import numpy as np


def estimate(rgb, px_mm, light_dir=(0.0, 1.0), elevation_deg=35.0, albedo_scale_mm=3.0, max_relief_mm=0.5):
    img = rgb.astype(np.float32) / 255.0
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0 + 1e-3
    sig = albedo_scale_mm / px_mm
    low = cv2.GaussianBlur(L, (0, 0), sig)
    s = L / (low + 1e-4) - 1.0
    s = np.clip(s, -0.8, 0.8)
    # slope toward the light: shading ~ tan(elevation)^-1 * slope for small slopes
    k = math.tan(math.radians(elevation_deg))
    # brighter = facing the light = height falls toward the light (image y grows toward a bottom light)
    slope = -s * k                                  # dh/dl (dimensionless, mm/mm)
    lx, ly = light_dir
    nrm = math.hypot(lx, ly); lx, ly = lx / nrm, ly / nrm
    gx, gy = slope * lx, slope * ly                 # observed gradient components (mm per mm)
    gx, gy = gx * px_mm, gy * px_mm                 # mm per pixel
    # Frankot-Chellappa integration with a small regulariser (suppresses the unobserved direction drift)
    H, W = s.shape
    pad = 64
    gxp = np.pad(gx, pad, mode="reflect"); gyp = np.pad(gy, pad, mode="reflect")
    Hh, Ww = gxp.shape
    wx = np.fft.fftfreq(Ww) * 2 * np.pi; wy = np.fft.fftfreq(Hh) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX ** 2 + WY ** 2 + 1e-3 * (2 * np.pi / (40 / px_mm)) ** 2   # damp features > ~40 mm
    Z = (-1j * WX * np.fft.fft2(gxp) - 1j * WY * np.fft.fft2(gyp)) / denom
    Z[0, 0] = 0
    h = np.real(np.fft.ifft2(Z))[pad:-pad, pad:-pad].astype(np.float32)
    h -= np.percentile(h, 1)
    h = np.clip(h, 0, None)
    p99 = np.percentile(h, 99.5)
    if p99 > max_relief_mm:                         # keep within plausible paint relief
        h *= max_relief_mm / p99
    # de-lit albedo: divide out only the shading that the recovered height explains, so paint detail
    # (grit, drags, colour variation) stays in the albedo
    gy2, gx2 = np.gradient(cv2.GaussianBlur(h, (0, 0), 0.6), px_mm)
    n = np.stack([-gx2, -gy2, np.ones_like(gx2)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
    el = math.radians(elevation_deg)
    Lv = np.array([lx * math.cos(el), ly * math.cos(el), math.sin(el)], np.float32)
    shade = np.clip((n * Lv).sum(2) / Lv[2], 0.3, 2.0)
    albedo = np.clip(img / (0.35 + 0.65 * shade)[..., None], 0, 1)
    return (albedo * 255).astype(np.uint8), h, s


def relight(albedo, height_mm, px_mm, light_dir=(0.0, 1.0), elevation_deg=35.0):
    """Lambertian re-render under a given light (for the check against the photo and for previews)."""
    gy, gx = np.gradient(cv2.GaussianBlur(height_mm, (0, 0), 0.6), px_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
    lx, ly = light_dir; nrm = math.hypot(lx, ly); lx, ly = lx / nrm, ly / nrm
    el = math.radians(elevation_deg)
    Lv = np.array([lx * math.cos(el), ly * math.cos(el), math.sin(el)], np.float32)
    shade = np.clip((n * Lv).sum(2), 0, None) / Lv[2]
    return np.uint8(np.clip(albedo.astype(np.float32) * (0.35 + 0.65 * shade)[..., None], 0, 255)), n


def lip_structure(rgb, px_mm, light_from_below=True, lip_height_mm=0.35):
    """Detect knife lips from their cast-shadow slits and rebuild them as relief.

    With light from below, a raised lip casts a dark slit ABOVE itself and shows a bright face
    BELOW. Slits = dark, thin, horizontally elongated features. The lip ridge is placed just
    below each slit, its height scaled by slit darkness and width (a longer shadow = taller lip).
    Returns (ridge_height_mm, slit_mask) ; slit_mask marks cast shadow (for de-lighting)."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    mm = 1.0 / px_mm
    local = cv2.GaussianBlur(L, (0, 0), 2.5 * mm)
    dark = np.clip((local - L - 6) / 22, 0, 1)                      # darker than surroundings
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(4 * mm) | 1, 1))
    slit = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)               # keep horizontally elongated darks
    slit = cv2.GaussianBlur(slit, (0, 0), 0.6)
    # shadow length along y ~ lip height / tan(elevation); use slit thickness as the height cue
    thick = cv2.GaussianBlur(slit, (0, 0), 0.8 * mm) * 2.5
    shift = int(round(0.9 * mm)) * (1 if light_from_below else -1)
    ridge_src = np.roll(np.clip(thick, 0, 1), shift, axis=0)
    ridge = cv2.GaussianBlur(ridge_src, (0, 0), 0.55 * mm)
    # ridge falls off gently on the band side (paint pushed into the lip), sharply on the slit side
    ridge = np.maximum(ridge, np.roll(ridge, shift, axis=0) * 0.6)
    return ridge / max(ridge.max(), 1e-6) * lip_height_mm, slit


def estimate_structured(rgb, px_mm, elevation_deg=35.0, **kw):
    alb, h_fine, s = estimate(rgb, px_mm, elevation_deg=elevation_deg, **kw)
    ridge, slit = lip_structure(rgb, px_mm)
    # fine detail from shading (high-passed), structure from lips, plus a low band film
    hp = h_fine - cv2.GaussianBlur(h_fine, (0, 0), 1.5 / px_mm)
    h = 0.08 + 0.6 * np.clip(hp, -0.15, 0.15) + ridge
    h = np.clip(h, 0, None).astype(np.float32)
    # de-light cast shadows: lift slit darkness toward the surrounding paint colour (keep 30% - real gap)
    surround = cv2.GaussianBlur(alb.astype(np.float32), (0, 0), 1.5 / px_mm)
    a = np.clip(slit * 0.7, 0, 1)[..., None]
    alb2 = np.uint8(np.clip(alb * (1 - a) + surround * a, 0, 255))
    return alb2, h
