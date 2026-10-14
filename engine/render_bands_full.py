"""Full-size knife-band painting from real exemplar crops.

The engine plans the bands (count, heights 9-24 mm, gentle wander, order). Each band is built
along its length from several real band segments (from the 100% crops), each warped to the band's
top and lip, joined end to end on minimum-error seams, with tone matched across each join. A
broad, low-contrast tone field (matched to the real painting's large-scale variation) is applied last.

Usage: python render_bands_full.py --size-cm 44 44 --dpi 300 --out out_dir [--seed 1]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_transfer import find_bands            # noqa: E402
from exemplar import _min_cut_vertical          # noqa: E402
from knife_bands import _noise1d                # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-cm", type=float, nargs=2, default=[44, 44])
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--out", default="out_bands")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exemplars", nargs="+", default=[
        f"library/closeups/structures8_crop0{i}.png" for i in (1, 2, 3, 4, 5)])
    ap.add_argument("--overlap", type=int, default=220)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    mm = a.dpi / 25.4
    W, H = int(round(a.size_cm[0] * 10 * mm)), int(round(a.size_cm[1] * 10 * mm))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    # exemplar band library (crop 1 shows the canvas edge: only its band area is usable)
    lib = []
    for path in a.exemplars:
        e = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
        if path.endswith("crop01.png"):
            e = e[600:, 600:]                     # painted bands only (skip border, tape, corner)
        Lx = cv2.cvtColor(e.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        for top, bot in find_bands(e.astype(np.uint8)):
            if bot - top >= 6 * mm:
                core = Lx[top + 8:bot - 8]
                busy = float(np.percentile(np.abs(core - cv2.GaussianBlur(core, (0, 0), 6)), 99)) if core.size else 99
                lib.append((e, top, bot, Path(path).stem, busy, float(core.mean()) if core.size else 100))
    heights = np.array([l[2] - l[1] for l in lib], np.float32)
    busy = np.array([l[4] for l in lib], np.float32); busy = (busy - busy.mean()) / (busy.std() + 1e-6)
    tones = np.array([l[5] for l in lib], np.float32)
    used = np.zeros(len(lib), np.float32)
    print(f"{W}x{H}px, {len(lib)} real bands in library")

    # plan: band boundaries
    bounds = []
    y = -rng.uniform(0, 20) * mm
    while y < H + 25 * mm:
        b = (y + _noise1d(rng, W, 250 * mm) * 3.0 * mm + _noise1d(rng, W, 90 * mm) * 1.8 * mm
             + _noise1d(rng, W, 25 * mm) * 1.0 * mm + _noise1d(rng, W, 8 * mm) * 0.35 * mm)
        bounds.append(b.astype(np.float32))
        y += rng.uniform(9, 24) * mm

    out = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)
    xs_all = np.arange(W, dtype=np.float32)
    margin, feather = 6, 3
    log = []
    for bi, (top, bot) in enumerate(zip(bounds[:-1], bounds[1:])):
        y0 = max(0, int(np.floor(top.min())) - margin - 2); y1 = min(H, int(np.ceil(bot.max())) + margin + 2)
        if y1 <= y0:
            continue
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        t = (yy - top[None, :]) / np.maximum(bot - top, 1)[None, :]
        hmean = float(np.mean(bot - top))
        strip = np.zeros((y1 - y0, W, 3), np.float32)
        x = 0; prev = None; segs = []
        band_tone = None
        while x < W:
            score = (np.abs(heights - hmean) / max(hmean, 1) + 0.25 * rng.random(len(lib)) + 0.35 * used
                     + 0.35 * np.clip(busy, 0, None))
            cands = list(np.argsort(score)[: (1 if prev is None else 10)])

            def make(k):
                e, st, sb = lib[k][0], lib[k][1], lib[k][2]
                ew = e.shape[1]
                seg_len = ew - 10 if prev is not None else int(rng.uniform(0.3, 1.0) * (ew - 10))   # stagger joins
                x0s = int(rng.integers(0, max(1, ew - seg_len)))
                flip = rng.random() < 0.5
                xa, xb = x, min(W, x + seg_len)
                src_x = x0s + (np.arange(xa, xb, dtype=np.float32) - xa)
                if flip:
                    src_x = (ew - 1) - src_x
                map_y = (st + t[:, xa:xb] * (sb - st)).astype(np.float32)
                map_x = np.broadcast_to(src_x[None, :], map_y.shape).astype(np.float32)
                return cv2.remap(e, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), xa, xb
            best = None
            for k in cands:
                seg, xa, xb = make(k)
                if band_tone is not None:            # one band keeps one tone along its length
                    seg = seg * (band_tone / (seg.mean((0, 1)) + 1e-3))[None, None, :]
                if prev is None:
                    cost = 0.0
                else:
                    ov = min(a.overlap, xb - xa, prev)
                    cost = float(((seg[:, :ov] - strip[:, xa:xa + ov]) ** 2).mean())
                if best is None or cost < best[0]:
                    best = (cost, k, seg, xa, xb)
            _, k, seg, xa, xb = best
            used[k] += 1; name = lib[k][3]
            if band_tone is None:
                band_tone = seg.mean((0, 1))
            if prev is None:
                strip[:, xa:xb] = seg
            else:
                ov = min(a.overlap, xb - xa, prev)
                old = strip[:, xa:xa + ov]
                g = (old.mean((0, 1)) + 1) / (seg[:, :ov].mean((0, 1)) + 1)
                ramp = np.clip(1 - np.arange(xb - xa) / max(1.0, 0.6 * (xb - xa)), 0, 1)[None, :, None]
                seg = seg * (1 + (g[None, None, :] - 1) * ramp)
                err = ((seg[:, :ov] - old) ** 2).sum(2)
                M = _min_cut_vertical(err)
                M = cv2.GaussianBlur(M, (0, 0), 1.2)
                strip[:, xa:xa + ov] = old * (1 - M[..., None]) + seg[:, :ov] * M[..., None]
                strip[:, xa + ov:xb] = seg[:, ov:]
            segs.append(name)
            prev = xb - xa
            if xb >= W:
                break
            x = xb - min(a.overlap, xb - xa - 1)
        d_top = yy - top[None, :]; d_bot = bot[None, :] - yy
        wgt = np.clip((np.minimum(d_top, d_bot) + feather) / (2 * feather), 0, 1)
        out[y0:y1] += strip * wgt[..., None]
        wsum[y0:y1] += wgt
        log.append({"band": bi, "height_mm": round(hmean / mm, 1), "segments": len(segs)})
        if bi % 10 == 0:
            print(f"band {bi}/{len(bounds) - 1}  {time.time() - t0:.0f}s", flush=True)
    out /= np.maximum(wsum, 1e-3)[..., None]
    del wsum
    # broad tone field matched to the real painting (std ~2.5 L at 1 cm, ~1.1 at 8 cm)
    def field(cell_cm, amp):
        g = rng.standard_normal((int(H / (cell_cm * 10 * mm)) + 4, int(W / (cell_cm * 10 * mm)) + 4)).astype(np.float32)
        f = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        return (f - f.mean()) / (f.std() + 1e-6) * amp
    tone = field(8, 0.025) + field(3, 0.015)
    for r0 in range(0, H, 1024):
        out[r0:r0 + 1024] *= (1 + tone[r0:r0 + 1024])[..., None]
    img = np.uint8(np.clip(out, 0, 255)); del out
    Image.fromarray(img).save(out_dir / "painting.png", dpi=(a.dpi, a.dpi))
    Image.fromarray(img).save(out_dir / "painting.jpg", quality=90, dpi=(a.dpi, a.dpi))
    Image.fromarray(img).resize((W // 6, H // 6), Image.LANCZOS).save(out_dir / "preview.jpg", quality=90)
    meta = {"size_cm": a.size_cm, "dpi": a.dpi, "px": [W, H], "seed": a.seed, "bands": len(log),
            "library_bands": len(lib), "max_reuse": int(used.max()), "mean_reuse": round(float(used.mean()), 1),
            "lighting": "baked from source photos (light from below)", "exemplars": [Path(p).name for p in a.exemplars],
            "seconds": round(time.time() - t0), "band_log": log}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "band_log"}))


if __name__ == "__main__":
    main()
