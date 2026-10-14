"""Capture lighting: render a render's maps the way a camera sees a real painting.

relight(albedo_u8, height_mm, rough01, px_mm, **opts) -> uint8 RGB
  azimuth_deg / elevation_deg  main light direction (default upper-left, 38 deg elevation)
  relief                      height exaggeration for shading (1 = physical)
  ambient                     share of soft fill light
  gloss                       highlight strength (0 matte .. 1 varnished oil)
  shadows / occlusion         cast shadows and contact shading on/off
  falloff                     broad left-to-right light falloff across the canvas
Works on any crop; for big canvases call it per tile with a margin (cast shadows reach ~30 px).
"""
import math

import cv2
import numpy as np


def relight(albedo, height_mm, rough, px_mm, azimuth_deg=135.0, elevation_deg=38.0, relief=2.5,
            ambient=0.45, gloss=0.35, shadows=True, occlusion=True, falloff=0.06, origin=(0, 0), canvas=None):
    alb = albedo.astype(np.float32) / 255.0
    H, W = height_mm.shape
    hz = cv2.GaussianBlur(height_mm.astype(np.float32), (0, 0), 0.7) * relief
    gy, gx = np.gradient(hz, px_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], 2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    # image x right, y down; azimuth 135 = light from upper-left
    L = np.array([math.cos(az) * math.cos(el), -math.sin(az) * math.cos(el), math.sin(el)], np.float32)
    ndl = np.clip((n * L).sum(2), 0, 1) / L[2]          # 1.0 on flat paint
    sh = np.ones((H, W), np.float32)
    if shadows:
        horiz = math.hypot(L[0], L[1]); dx, dy = L[0] / horiz, L[1] / horiz; tan_el = L[2] / horiz
        for step in (1, 2, 3, 5, 8, 12, 18, 26):
            M = np.float32([[1, 0, dx * step], [0, 1, dy * step]])
            ahead = cv2.warpAffine(hz, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rise = ahead - hz - step * px_mm * tan_el
            sh = np.minimum(sh, 1 - np.clip(rise / (0.004 * relief), 0, 1))
        sh = cv2.GaussianBlur(sh, (0, 0), 0.8)
    ao = np.ones((H, W), np.float32)
    if occlusion:
        for s, w in ((2, 0.45), (6, 0.35), (16, 0.2)):
            ao -= w * np.clip((cv2.GaussianBlur(hz, (0, 0), s) - hz) / (0.02 * relief), 0, 1)
        ao = np.clip(ao, 0.4, 1)
    Hv = L + np.array([0, 0, 1], np.float32); Hv /= np.linalg.norm(Hv)
    r = rough.astype(np.float32)
    shin = 8 + 120 * (1 - r) ** 2
    spec = (np.clip((n * Hv).sum(2), 0, 1) ** shin) * (1 - r) * gloss
    if canvas is None:
        canvas = (W, H)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fall = 1.0 + falloff * (0.5 - (xx + origin[0]) / canvas[0]) - 0.5 * falloff * ((yy + origin[1]) / canvas[1] - 0.5)
    light = ambient * ao + (1 - ambient) * ndl * sh
    lit = alb * light[..., None] + spec[..., None] * np.array([1.0, 0.98, 0.95], np.float32)
    return np.uint8(np.clip(lit * fall[..., None], 0, 1) * 255)
