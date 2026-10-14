"""Capture reconstruction and checks (classical, no AI).

photometric_stereo(images, light_dirs) -> albedo, normals
    Lambertian photometric stereo: for each pixel, solve I_k = rho * (n . l_k) over the K lights by least
    squares. Pixels in cast shadow (very dark in one light) are down-weighted so they don't bias the fit.
integrate_normals(normals, px_mm) -> height (mm, relative; zero at the 1st percentile)
    Frankot-Chellappa (Fourier) integration of the surface gradients.
light_vector(direction, elevation_deg) -> unit vector; direction in image coords: 'top','bottom','left','right'
check_relight(albedo, height_mm, px_mm, images, light_dirs) -> per-light correlation of detail
    Re-render the reconstruction under each capture light and compare with the photo (the standing check).
"""
import math

import cv2
import numpy as np

DIRS = {"top": (0.0, -1.0), "bottom": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def light_vector(direction, elevation_deg):
    dx, dy = DIRS[direction] if isinstance(direction, str) else direction
    n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
    el = math.radians(elevation_deg)
    return np.array([dx * math.cos(el), dy * math.cos(el), math.sin(el)], np.float32)


def photometric_stereo(images, light_dirs, shadow_frac=0.15):
    """images: list of HxWx3 uint8 (linear-ish); light_dirs: list of unit 3-vectors."""
    I = np.stack([im.astype(np.float32).mean(2) / 255.0 for im in images], 0)       # K,H,W luminance
    Lm = np.stack(light_dirs, 0).astype(np.float32)                                 # K,3
    K, H, W = I.shape
    # weights: ignore samples that look like cast shadow (much darker than this pixel's brightest view)
    w = (I > shadow_frac * I.max(0, keepdims=True)).astype(np.float32) + 1e-3
    # weighted least squares per pixel: g = rho*n = (L^T W L)^-1 L^T W I
    A = np.einsum("khw,ki,kj->hwij", w, Lm, Lm)                                     # H,W,3,3
    b = np.einsum("khw,ki,khw->hwi", w, Lm, I)                                      # H,W,3
    A += np.eye(3, dtype=np.float32) * 1e-6
    g = np.linalg.solve(A, b[..., None])[..., 0]                                    # H,W,3
    rho = np.linalg.norm(g, axis=2) + 1e-8
    n = g / rho[..., None]
    n[..., 2] = np.clip(n[..., 2], 0.05, None); n /= np.linalg.norm(n, axis=2, keepdims=True)
    # colour albedo: per channel, the same fit scaled by each channel's share of luminance
    lum = np.stack([im.astype(np.float32).mean(2) for im in images], 0).sum(0) + 1e-3
    col = np.stack([im.astype(np.float32) for im in images], 0).sum(0) / lum[..., None]
    albedo = np.clip(rho[..., None] * col * 255.0, 0, 255).astype(np.uint8)
    return albedo, n


def integrate_normals(n, px_mm, long_mm=60.0):
    gx = -n[..., 0] / n[..., 2] * px_mm             # dz/dx in mm per pixel
    gy = -n[..., 1] / n[..., 2] * px_mm
    pad = 64
    gxp = np.pad(gx, pad, mode="reflect"); gyp = np.pad(gy, pad, mode="reflect")
    Hh, Ww = gxp.shape
    wx = np.fft.fftfreq(Ww) * 2 * np.pi; wy = np.fft.fftfreq(Hh) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    reg = (2 * np.pi / (long_mm / px_mm)) ** 2 * 1e-2
    Z = (-1j * WX * np.fft.fft2(gxp) - 1j * WY * np.fft.fft2(gyp)) / (WX ** 2 + WY ** 2 + reg)
    Z[0, 0] = 0
    h = np.real(np.fft.ifft2(Z))[pad:-pad, pad:-pad].astype(np.float32)
    return h - np.percentile(h, 1)


def render(albedo, height_mm, px_mm, light, ambient=0.0, shadows=True):
    """Lambertian render with cast shadows (used to make synthetic captures and for the relight check)."""
    gy, gx = np.gradient(height_mm, px_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
    sh = np.clip((n * light).sum(2), 0, None)
    if shadows:
        horiz = math.hypot(light[0], light[1]); dx, dy = light[0] / horiz, light[1] / horiz
        tan_el = light[2] / horiz
        vis = np.ones_like(sh); H, W = sh.shape
        for step in (1, 2, 3, 5, 8, 12, 18, 26, 40):
            M = np.float32([[1, 0, -dx * step], [0, 1, -dy * step]])
            ahead = cv2.warpAffine(height_mm, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vis = np.minimum(vis, 1 - np.clip((ahead - height_mm - step * px_mm * tan_el) / 0.005, 0, 1))
        sh *= vis
    return np.uint8(np.clip(albedo.astype(np.float32) * (ambient + (1 - ambient) * sh)[..., None], 0, 255))


def check_relight(albedo, height_mm, px_mm, images, lights):
    L = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 12)
    out = []
    for im, l in zip(images, lights):
        re = render(albedo, height_mm, px_mm, l)
        out.append(float(np.corrcoef(hp(L(im)).ravel(), hp(L(re)).ravel())[0, 1]))
    return out
