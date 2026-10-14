"""Band-aware exemplar transfer for knife-dragged band paintings.

1. find_bands(crop): locate the real bands in a 100% crop. Each band ends in a lip whose shadow
   makes a dark row; minima of the row-mean lightness (smoothed) give the band boundaries.
2. transfer(plan_bounds, exemplars, ...): for every band in the engine's plan (top(x), bot(x)),
   pick a real band of similar height, take a horizontal window of it, and warp it vertically so
   its top and lip land on the plan's boundaries column by column. Real lips, drags and grit stay
   intact; the arrangement (heights, wander, order, which bands) is new.
Neighbouring bands are cross-faded over a few pixels so no seam shows.
"""
import cv2
import numpy as np


def find_bands(rgb, min_mm=6.0, px_mm=25.4 / 300):
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    prof = L.mean(1)
    prof = prof - cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 40).ravel()
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 2).ravel()
    min_px = int(min_mm / px_mm)
    cands = [i for i in range(2, len(prof) - 2) if prof[i] < prof[i - 1] and prof[i] <= prof[i + 1] and prof[i] < -1.0]
    cands.sort(key=lambda i: prof[i])           # darkest first
    chosen = []
    for i in cands:
        if all(abs(i - j) >= min_px for j in chosen):
            chosen.append(i)
    chosen.sort()
    bands = [(a, b) for a, b in zip(chosen[:-1], chosen[1:])]
    return bands


def transfer(bounds, exemplars, rng, lip_margin_px=6, feather_px=3):
    """bounds: list of arrays (len W) of band boundaries in the target, top to bottom."""
    W = len(bounds[0])
    H = int(np.ceil(max(b.max() for b in bounds))) + 1
    lib = []
    for e in exemplars:
        for a, b in find_bands(e):
            lib.append((e, a, b))
    heights = np.array([b - a for _, a, b in lib], np.float32)
    out = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, 1 if False else W), np.float32)
    xs = np.arange(W, dtype=np.float32)
    used = np.zeros(len(lib), np.float32)
    for top, bot in zip(bounds[:-1], bounds[1:]):
        hmean = float(np.mean(bot - top))
        # prefer a real band of similar height (less stretching), with some randomness
        score = np.abs(heights - hmean) / max(hmean, 1) + 0.25 * rng.random(len(lib)) + 0.6 * used
        pick = int(np.argmin(score)); used[pick] += 1
        e, a, b = lib[pick]
        a2, b2 = max(0, a - lip_margin_px), min(e.shape[0] - 1, b + lip_margin_px)
        ew = e.shape[1]
        flip = rng.random() < 0.5              # drag direction can run either way
        x0 = rng.integers(0, max(1, ew - min(ew, W)))
        y0i, y1i = int(np.floor(top.min())) - lip_margin_px - 2, int(np.ceil(bot.max())) + lip_margin_px + 2
        y0i, y1i = max(0, y0i), min(H, y1i)
        yy = np.arange(y0i, y1i, dtype=np.float32)[:, None]
        # fractional position across the band (0 = top lip, 1 = bottom lip), extended into margins
        t = (yy - top[None, :]) / np.maximum(bot - top, 1)[None, :]
        src_y = a + t * (b - a)
        src_x = (x0 + (xs % ew if W > ew else xs))[None, :].repeat(len(yy), 0)
        if flip:
            src_x = (ew - 1) - src_x
        map_x = src_x.astype(np.float32); map_y = src_y.astype(np.float32)
        patch = cv2.remap(e.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # weight: 1 inside the band, feathering out over a few pixels past each boundary
        d_top = (yy - top[None, :]); d_bot = (bot[None, :] - yy)
        wgt = np.clip((np.minimum(d_top, d_bot) + feather_px) / (2 * feather_px), 0, 1)
        out[y0i:y1i] += patch * wgt[..., None]
        wsum[y0i:y1i] += wgt
    out /= np.maximum(wsum, 1e-3)[..., None]
    return np.uint8(np.clip(out, 0, 255))
