"""Mark extraction and composition (classical image processing, no AI).

A *mark* is one recorded gesture with all its maps: albedo, height (mm, relative to the surface it sat on),
gloss, an alpha mask with a soft edge, and descriptors (type, size mm, orientation, mean colour, source).

extract_marks(area, ...) -> list[Mark]
    1. base surface = large-scale median of height (the ground the gesture was laid on)
    2. relief = height - base; marks = connected regions where relief exceeds a threshold, grown by a margin
       so each keeps its own shoulders and shadow side
    3. each mark is cut with a feathered alpha; descriptors from its principal axis and statistics
    The remaining low-relief surface is kept as *fill* patches (for areas between marks).

compose(canvas_h, canvas_w, px_mm, placements, marks, fill=None) -> dict of maps
    Places marks (position, rotation, scale ~1, colour shift) and composites them physically:
    a later mark's paint sits ON the surface below: height = below + alpha * mark_relief,
    albedo/gloss blend by alpha. Order = paint order.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Mark:
    albedo: np.ndarray        # h,w,3 uint8
    relief: np.ndarray        # h,w float32 mm above its base
    alpha: np.ndarray         # h,w float32 0..1
    gloss: np.ndarray         # h,w float32
    kind: str
    size_mm: tuple
    angle_deg: float
    mean_rgb: tuple
    source: str = ""
    meta: dict = field(default_factory=dict)


def extract_marks(area, source="", min_relief_mm=0.04, min_size_mm=3.0, margin_mm=1.2, max_marks=400):
    px = area["px_mm"]; mm = 1.0 / px
    h = area["height_mm"].astype(np.float32)
    alb = area["albedo"]
    gloss = area.get("gloss", np.full(h.shape, 0.4, np.float32))
    from scipy.ndimage import median_filter
    base = median_filter(cv2.resize(h, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), size=int(4 * mm / 4) | 1)
    base = cv2.resize(cv2.GaussianBlur(base, (0, 0), 3), (h.shape[1], h.shape[0]))
    relief = h - base
    m = (relief > min_relief_mm).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    marks = []
    order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
    pad = int(margin_mm * mm)
    for i in order[:max_marks]:
        x, y, w, hh, area_px = st[i]
        if max(w, hh) < min_size_mm * mm:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(h.shape[1], x + w + pad), min(h.shape[0], y + hh + pad)
        core = (lab[y0:y1, x0:x1] == i).astype(np.float32)
        rl = np.clip(relief[y0:y1, x0:x1] / max(min_relief_mm, 1e-6), 0, 1)       # soft, follows the relief
        grown = cv2.GaussianBlur(cv2.dilate(core, np.ones((pad | 1, pad | 1), np.uint8)), (0, 0), max(1.0, 0.5 * pad))
        a = np.clip(np.maximum(core, rl) * 1.2, 0, 1) * np.clip(grown * 1.3, 0, 1)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)
        # a mark cut by the edge of the captured area would end square; taper it over ~4 mm instead
        yy_, xx_ = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d_edge = np.minimum.reduce([xx_, yy_, h.shape[1] - 1 - xx_, h.shape[0] - 1 - yy_]) / (4.0 * mm)
        taper = np.clip(d_edge, 0, 1) ** 1.5
        a = a * taper
        r = np.clip(relief[y0:y1, x0:x1], 0, None) * taper
        ys, xs = np.nonzero(core)
        ang = 0.0
        if len(xs) > 5:
            c = np.cov(np.stack([xs, ys]))
            ev, evec = np.linalg.eigh(c)
            ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
            elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        L = max(w, hh) * px
        kind = "lip" if elong > 6 and r.max() > 0.12 else "drag" if elong > 3 else "fleck" if L < 6 else "daub"
        sub_alb = alb[y0:y1, x0:x1]
        marks.append(Mark(albedo=sub_alb.copy(), relief=r.astype(np.float32), alpha=a.astype(np.float32),
                          gloss=gloss[y0:y1, x0:x1].copy(), kind=kind,
                          size_mm=(round(w * px, 1), round(hh * px, 1)), angle_deg=round(ang, 1),
                          mean_rgb=tuple(int(v) for v in sub_alb[core > 0].mean(0)),
                          source=source, meta={"peak_mm": round(float(r.max()), 3), "elongation": round(elong, 1)}))
    fill = {"albedo": alb, "height": base + np.clip(relief, None, min_relief_mm), "gloss": gloss}
    return marks, fill


def _place(arr, M, size, interp=cv2.INTER_LINEAR):
    return cv2.warpAffine(arr, M, size, flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compose(H, W, px_mm, placements, marks, ground=None):
    """placements: list of dicts {mark: index, x, y (px), angle_deg, scale, rgb_shift (3,), relief_gain}."""
    if ground is None:
        albedo = np.full((H, W, 3), 200, np.float32); height = np.zeros((H, W), np.float32)
        gloss = np.full((H, W), 0.3, np.float32)
    else:
        albedo = ground["albedo"].astype(np.float32).copy(); height = ground["height"].astype(np.float32).copy()
        gloss = ground["gloss"].astype(np.float32).copy()
    owner = np.full((H, W), -1, np.int32)
    for k, p in enumerate(placements):
        mk = marks[p["mark"]]
        mh, mw = mk.alpha.shape
        c = (mw / 2, mh / 2)
        M = cv2.getRotationMatrix2D(c, p.get("angle_deg", 0.0), p.get("scale", 1.0))
        M[0, 2] += p["x"] - c[0]; M[1, 2] += p["y"] - c[1]
        a = _place(mk.alpha, M, (W, H))
        if a.max() < 1e-3:
            continue
        r = _place(mk.relief, M, (W, H)) * p.get("relief_gain", 1.0)
        col = _place(mk.albedo.astype(np.float32), M, (W, H), cv2.INTER_CUBIC)
        # keep the mark's own detail, take its broad colour from the surface it lands on (paint picks up
        # and carries the local colour), so marks never read as pasted-in tiles
        s_ = max(6.0, 2.0 / px_mm)
        aw = cv2.GaussianBlur(a, (0, 0), s_) + 1e-4
        mark_low = cv2.GaussianBlur(col * a[..., None], (0, 0), s_) / aw[..., None]
        ground_low = cv2.GaussianBlur(albedo, (0, 0), s_)
        k_ = p.get("own_colour", 0.25)              # 0 = fully local colour, 1 = mark keeps its own
        col = col - mark_low + (ground_low * (1 - k_) + mark_low * k_)
        col = np.clip(col * (1 + p.get("brightness", 0.0)), 0, 255)
        g = _place(mk.gloss, M, (W, H))
        albedo = albedo * (1 - a[..., None]) + col * a[..., None]
        height = height + a * r                     # new paint sits on what is below
        gloss = gloss * (1 - a) + g * a
        owner[a > 0.5] = k
    return {"albedo": np.uint8(np.clip(albedo, 0, 255)), "height_mm": height, "gloss": gloss, "owner": owner}


def normals_from_height(h, px_mm):
    gy, gx = np.gradient(cv2.GaussianBlur(h, (0, 0), 0.6), px_mm)
    n = np.stack([-gx, -gy, np.ones_like(gx)], 2)
    return n / np.linalg.norm(n, axis=2, keepdims=True)
