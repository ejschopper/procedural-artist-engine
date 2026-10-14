"""Build the reference library: crop each photo to the painting, compute statistics, write manifest.

The library is used only for statistics (palette, texture, structure) and as a visual comparison
set. The engine never copies a reference composition or pixels.

Usage: python build_library.py            (reads images/, tags.json -> manifest.json, crops/)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
STAT_PX = 600  # all statistics are computed at this long-side size so they are comparable


def auto_crop(rgb):
    """Find the painting inside a photo by separating it from the border colour (wall/background)."""
    h, w = rgb.shape[:2]
    b = max(4, int(0.025 * min(h, w)))
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                             rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)]).astype(np.float32)
    wall = np.median(border, 0)
    spread = np.median(np.abs(border - wall).sum(1))
    if spread > 40:          # border is busy: the image is already (mostly) the painting
        return (0, 0, w, h), "full"
    d = np.abs(rgb.astype(np.float32) - wall).sum(2)
    m = (d > max(28, 3 * spread)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return (0, 0, w, h), "full"
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, cw, ch = st[i, :4]
    frac = cw * ch / (w * h)
    if frac < 0.12 or frac > 0.97:
        return (0, 0, w, h), "full"
    inset = int(0.03 * min(cw, ch))    # step inside frame edges / shadows
    return (int(x + inset), int(y + inset), int(x + cw - inset), int(y + ch - inset)), "auto"


def stats(rgb):
    h, w = rgb.shape[:2]
    s = STAT_PX / max(h, w)
    img = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; bb = lab[..., 2] - 128
    C = np.hypot(a, bb)
    hf = L - cv2.GaussianBlur(L, (0, 0), 2.0)
    mf = cv2.GaussianBlur(L, (0, 0), 2.0) - cv2.GaussianBlur(L, (0, 0), 12.0)
    tot = L.var() + 1e-6
    hh, ww = L.shape
    e = max(2, int(0.06 * min(hh, ww)))
    outer = np.concatenate([L[:e].ravel(), L[-e:].ravel(), L[:, :e].ravel(), L[:, -e:].ravel()])
    inner = L[hh // 4: 3 * hh // 4, ww // 4: 3 * ww // 4]
    px = lab.reshape(-1, 3)[:: max(1, (hh * ww) // 20000)].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(1)
    _, lbl, cen = cv2.kmeans(px, 6, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lbl.ravel(), minlength=6) / len(lbl)
    order = np.argsort(-cnt)
    rgbc = cv2.cvtColor(np.uint8(np.clip(cen[order][None], 0, 255)), cv2.COLOR_LAB2RGB)[0]
    return {
        "lightness_mean": round(float(L.mean()), 2),
        "lightness_std": round(float(L.std()), 2),
        "chroma_mean": round(float(C.mean()), 2),
        "texture_fine": round(float(hf.std()), 3),       # grain, scratches, dry brush (at 600 px)
        "texture_mid": round(float(mf.std()), 3),        # mottling, scumbles, patches
        "band_horizontal": round(float(L.mean(1).var() / tot), 3),  # 1 = pure horizontal bands
        "band_vertical": round(float(L.mean(0).var() / tot), 3),
        "edge_darkening": round(float(inner.mean() - outer.mean()), 2),
        "palette": [{"rgb": [int(v) for v in c], "weight": round(float(cnt[o]), 3)} for c, o in zip(rgbc, order)],
    }


def main():
    tags = json.loads((ROOT / "tags.json").read_text())
    index = json.loads((ROOT / "index.json").read_text())
    (ROOT / "crops").mkdir(exist_ok=True)
    out = []
    for r in index:
        rid = r["id"]
        t = tags.get(rid, {})
        rgb = np.asarray(Image.open(ROOT / "images" / f"{rid}.jpg").convert("RGB"))
        if "crop_frac" in t:
            hh, ww = rgb.shape[:2]; f = t["crop_frac"]
            box, how = (int(f[0] * ww), int(f[1] * hh), int(f[2] * ww), int(f[3] * hh)), "manual"
        else:
            box, how = auto_crop(rgb)
        x0, y0, x1, y1 = box
        crop = rgb[y0:y1, x0:x1]
        Image.fromarray(crop).save(ROOT / "crops" / f"{rid}.jpg", quality=90)
        out.append({"id": rid, **{k: v for k, v in t.items() if k != "crop_frac"},
                    "crop_xyxy": list(box), "crop_method": how, "stats": stats(crop)})
    (ROOT / "manifest.json").write_text(json.dumps({
        "description": "Reference library: user-supplied images, for statistics and visual comparison only.",
        "stat_long_side_px": STAT_PX, "count": len(out), "images": out}, indent=1))
    fams = {}
    for o in out:
        fams[o.get("family", "untagged")] = fams.get(o.get("family", "untagged"), 0) + 1
    print(len(out), "images;", dict(sorted(fams.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
