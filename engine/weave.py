"""Plain-weave linen height pattern (shared by the engine and the upscaler).

Each cell of the thread grid shows whichever thread is on top there (warp and weft alternate in a
checkerboard). The top thread is a rounded capsule running along its own direction, so the surface
reads as a basket of short vertical and horizontal bumps, never as diagonal hatching.
"""
import math

import cv2
import numpy as np


class Weave:
    def __init__(self, rng, h, w, pitch_px, slub=0.18):
        self.h, self.w, self.p = h, w, pitch_px
        n_u = int(w / pitch_px) + 4; n_v = int(h / pitch_px) + 4
        # slubs: per-thread thickness variation (fine linen ~0.1, jute ~0.35)
        self.thick_u = 1 + slub * rng.standard_normal(n_u).astype(np.float32)
        self.thick_v = 1 + slub * rng.standard_normal(n_v).astype(np.float32)
        cell = 60 * pitch_px                                                     # slow thread wander
        self.gx = rng.standard_normal((int(h / cell) + 4, int(w / cell) + 4)).astype(np.float32)
        self.gy = rng.standard_normal((int(h / cell) + 4, int(w / cell) + 4)).astype(np.float32)

    def _wander(self, g, ya, yb):
        gh, gw = g.shape
        mx = (np.arange(self.w, dtype=np.float32) + 0.5) * (gw - 3) / self.w + 1
        my = (np.arange(ya, yb, dtype=np.float32) + 0.5) * (gh - 3) / self.h + 1
        MX, MY = np.meshgrid(mx, my)
        return cv2.remap(g, MX, MY, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def rows(self, ya, yb):
        """Weave height in [-1, 1] for rows ya:yb (full width)."""
        pad = 2
        a0, b0 = max(0, ya - pad), min(self.h, yb + pad)
        yy, xx = np.mgrid[a0:b0, 0:self.w].astype(np.float32)
        u = (xx + 0.8 * self.p * self._wander(self.gx, a0, b0)) / self.p
        v = (yy + 0.8 * self.p * self._wander(self.gy, a0, b0)) / self.p
        iu, iv = np.floor(u), np.floor(v)
        fu, fv = u - iu, v - iv
        tu = self.thick_u[np.clip(iu.astype(int), 0, len(self.thick_u) - 1)]
        tv = self.thick_v[np.clip(iv.astype(int), 0, len(self.thick_v) - 1)]
        warp_on_top = ((iu + iv) % 2) == 0
        warp = np.sin(math.pi * fu) * (0.55 + 0.45 * np.sin(math.pi * fv)) * tu
        weft = np.sin(math.pi * fv) * (0.55 + 0.45 * np.sin(math.pi * fu)) * tv
        hgt = np.where(warp_on_top, warp, weft)
        hgt = cv2.GaussianBlur(hgt, (0, 0), max(0.5, 0.07 * self.p))           # anti-alias thread edges
        hgt = hgt[ya - a0: ya - a0 + (yb - ya)]
        return np.clip(hgt * 2 - 1, -1, 1).astype(np.float32)


def weave_rows_supersampled(gen, f, a, b, out_w):
    """Rows a:b of a weave generated at f x resolution and area-averaged down (no aliasing)."""
    import cv2
    hi = gen.rows(a * f, b * f)
    return cv2.resize(hi, (out_w, b - a), interpolation=cv2.INTER_AREA)
