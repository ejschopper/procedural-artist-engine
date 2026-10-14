"""Exemplar surface synthesis: real paint detail, arranged to the engine's plan.

The procedural engine decides the structure (where bands, lips, gaps, edges and tones go). This
module fills that plan with surface detail taken patch by patch from real 100% crops (the artist's
own work), so micro-texture, grit, colour variation and accidents are real rather than noise.

Method: guided image quilting (Efros & Freeman 2001, with a guidance term as in Image Analogies).
For each patch position, candidates from the exemplars are scored on
  (a) agreement with already-placed neighbours in the overlap region, and
  (b) agreement between the exemplar's structure and the engine's structure at that location;
the best few are kept, one is chosen, and it is stitched in along a minimum-error seam.
No exemplar is reproduced as a whole: every output is a new arrangement of small patches.

synthesize(target_guide, exemplars, exemplar_guides, patch=64, overlap=16, ...) -> uint8 RGB
"""
import numpy as np


def _min_cut_vertical(err):
    """Minimum-error vertical seam through err (h x ov); returns mask (h x ov), 1 = take new patch."""
    h, w = err.shape
    E = err.copy()
    for i in range(1, h):
        left = np.r_[np.inf, E[i - 1, :-1]]
        right = np.r_[E[i - 1, 1:], np.inf]
        E[i] += np.minimum(np.minimum(left, E[i - 1]), right)
    mask = np.zeros((h, w), np.float32)
    j = int(np.argmin(E[-1]))
    for i in range(h - 1, -1, -1):
        mask[i, j:] = 1
        if i > 0:
            lo, hi = max(0, j - 1), min(w, j + 2)
            j = lo + int(np.argmin(E[i - 1, lo:hi]))
    return mask


def synthesize(target_guide, exemplars, exemplar_guides, patch=(48, 240), overlap=(14, 60), candidates=700,
               keep=3, guide_weight=1.6, coherence=0.55, rng=None, progress=False):
    """patch/overlap are (height, width): wide, short patches follow a horizontal drag direction."""
    rng = rng or np.random.default_rng(0)
    if np.isscalar(patch): patch = (patch, patch)
    if np.isscalar(overlap): overlap = (overlap, overlap)
    ph, pw = patch; oy, ox = overlap
    H, W = target_guide.shape
    sy, sx = ph - oy, pw - ox
    ny = int(np.ceil((H - oy) / sy)); nx = int(np.ceil((W - ox) / sx))
    Ho, Wo = ny * sy + oy, nx * sx + ox
    tg = np.pad(target_guide.astype(np.float32), ((0, Ho - H), (0, Wo - W)), mode="edge")
    out = np.zeros((Ho, Wo, 3), np.float32)
    ex = [e.astype(np.float32) for e in exemplars]
    eg = [g.astype(np.float32) for g in exemplar_guides]
    src = {}                                  # (by, bx) -> (k, cy, cx) chosen source
    sub = 4
    for by in range(ny):
        for bx in range(nx):
            y, x = by * sy, bx * sx
            tgp = tg[y:y + ph, x:x + pw]
            k = list(rng.integers(0, len(ex), candidates))
            cy = [int(rng.integers(0, ex[i].shape[0] - ph)) for i in k]
            cx = [int(rng.integers(0, ex[i].shape[1] - pw)) for i in k]
            # coherence candidates: continue the neighbours' source strips
            coh = set()
            for (dy, dx) in ((0, -1), (-1, 0)):
                if (by + dy, bx + dx) in src:
                    kk, a, b = src[(by + dy, bx + dx)]
                    for jit in (-2, 0, 2):
                        a2, b2 = a - dy * sy + jit, b - dx * sx
                        if 0 <= a2 <= ex[kk].shape[0] - ph and 0 <= b2 <= ex[kk].shape[1] - pw:
                            k.append(kk); cy.append(a2); cx.append(b2); coh.add(len(k) - 1)
            g_small = np.stack([eg[i][a:a + ph:sub, b:b + pw:sub] for i, a, b in zip(k, cy, cx)])
            gcost = ((g_small - tgp[::sub, ::sub][None]) ** 2).mean(axis=(1, 2))
            pre = list(np.argsort(gcost)[:40]) + [c for c in coh]
            costs = []
            for c in pre:
                P = ex[k[c]][cy[c]:cy[c] + ph, cx[c]:cx[c] + pw]
                cost = guide_weight * ((eg[k[c]][cy[c]:cy[c] + ph, cx[c]:cx[c] + pw] - tgp) ** 2).mean()
                n = 0; ov = 0.0
                if bx > 0:
                    ov += ((P[:, :ox] - out[y:y + ph, x:x + ox]) ** 2).mean(); n += 1
                if by > 0:
                    ov += ((P[:oy, :] - out[y:y + oy, x:x + pw]) ** 2).mean(); n += 1
                cost += ov / n if n else 0.0
                if c in coh:
                    cost *= coherence
                costs.append(cost)
            order = np.argsort(costs)[:keep]
            c = pre[order[0]] if rng.random() < 0.7 else pre[order[rng.integers(len(order))]]
            src[(by, bx)] = (k[c], cy[c], cx[c])
            P = ex[k[c]][cy[c]:cy[c] + ph, cx[c]:cx[c] + pw].copy()
            M = np.ones((ph, pw), np.float32)
            if bx > 0:
                err = ((P[:, :ox] - out[y:y + ph, x:x + ox]) ** 2).sum(2)
                M[:, :ox] = np.minimum(M[:, :ox], _min_cut_vertical(err))
            if by > 0:
                err = ((P[:oy, :] - out[y:y + oy, x:x + pw]) ** 2).sum(2)
                M[:oy, :] = np.minimum(M[:oy, :], _min_cut_vertical(err.T).T)
            Mb = M.copy()
            Mb[1:-1, 1:-1] = (M[:-2, 1:-1] + M[2:, 1:-1] + M[1:-1, :-2] + M[1:-1, 2:] + 4 * M[1:-1, 1:-1]) / 8
            if bx == 0 and by == 0:
                Mb[:] = 1
            region = out[y:y + ph, x:x + pw]
            out[y:y + ph, x:x + pw] = region * (1 - Mb[..., None]) + P * Mb[..., None]
        if progress:
            print(f"row {by + 1}/{ny}", flush=True)
    return np.uint8(np.clip(out[:H, :W], 0, 255))
