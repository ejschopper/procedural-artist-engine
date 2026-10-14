#!/usr/bin/env python3
"""Make full-scale detail crops of artwork photos for the Procedural Artist Engine reference library.

Run this on YOUR computer over your original high-resolution photos or scans. It writes small crops
that show how paint, edges and surfaces look at print scale, with the real-world scale recorded in
each file name, plus one downsized "whole" view per artwork and a CSV index. Send the output folder
(zipped) back; the originals stay with you.

    pip install pillow numpy
    python make_detail_crops.py PHOTOS_DIR OUT_DIR --width-in 30           # every photo is a 30" wide painting
    python make_detail_crops.py PHOTOS_DIR OUT_DIR --sizes sizes.csv        # per-file sizes
    python make_detail_crops.py PHOTOS_DIR OUT_DIR                          # unknown size: crops still made, scale = "unk"

sizes.csv (optional): filename,width_in[,family]   e.g.  IMG_0443.jpg,30,weathered_field
The photo should be cropped to the painting (or nearly) for the scale to be right: scale is computed
as photo_pixels_across / painting_width_in = pixels per inch (ppi) of real canvas.

Output per artwork:
    <name>__whole.jpg                    long side 1500 px (composition, palette)
    <name>__crop01_250ppi.jpg ...        CROP x CROP px detail crops at native resolution
    index.csv                            file, source, family, ppi, crop box, detail score
Crops are chosen to be varied and busy: low-detail (blank, out-of-focus, background) patches are
skipped, and crops are spread over the picture. Photos below --min-ppi get no detail crops (they
cannot show print-scale surface), only the whole view.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic"}


def detail_score(gray):
    """Fine-scale contrast: high for brushwork, scratches, texture; low for flat or blurred areas."""
    g = np.asarray(gray, np.float32)
    blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(2)), np.float32)
    return float(np.std(g - blur))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos"); ap.add_argument("out")
    ap.add_argument("--width-in", type=float, help="painting width in inches, same for all photos")
    ap.add_argument("--sizes", help="CSV: filename,width_in[,family]")
    ap.add_argument("--family", default="", help="family tag for all photos (optional)")
    ap.add_argument("--crop", type=int, default=768, help="crop size in pixels (default 768)")
    ap.add_argument("--per-image", type=int, default=6, help="detail crops per artwork (default 6)")
    ap.add_argument("--min-ppi", type=float, default=100, help="skip detail crops below this real-canvas ppi")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    sizes = {}
    if a.sizes:
        for row in csv.reader(open(a.sizes)):
            if row and not row[0].startswith("#"):
                sizes[row[0].strip()] = (float(row[1]), row[2].strip() if len(row) > 2 else "")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    rows = []
    files = sorted(p for p in Path(a.photos).rglob("*") if p.suffix.lower() in EXTS)
    for i, f in enumerate(files, 1):
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            print("skip", f.name, e); continue
        name = f.stem.replace(" ", "_")
        w_in, fam = sizes.get(f.name, (a.width_in, a.family))
        ppi = (im.width / w_in) if w_in else None
        whole = im.copy(); whole.thumbnail((1500, 1500))
        wf = f"{name}__whole.jpg"; whole.save(out / wf, quality=a.quality)
        rows.append([wf, f.name, fam, f"{ppi:.0f}" if ppi else "unk", "", "", "whole"])
        C = a.crop
        if im.width < C * 1.5 or im.height < C * 1.5 or (ppi and ppi < a.min_ppi):
            print(f"[{i}/{len(files)}] {f.name}: whole only ({'low ppi' if ppi else 'small'})"); continue
        # candidates on a jittered grid, scored for detail; keep the best spread-out ones
        cands = []
        for _ in range(60):
            x = rng.randint(0, im.width - C); y = rng.randint(0, im.height - C)
            g = im.crop((x, y, x + C, y + C)).convert("L").resize((C // 4, C // 4))
            cands.append((detail_score(g), x, y))
        cands.sort(reverse=True)
        chosen = []
        for sc, x, y in cands:
            if all(abs(x - cx) > C * 0.75 or abs(y - cy) > C * 0.75 for _, cx, cy in chosen):
                chosen.append((sc, x, y))
            if len(chosen) >= a.per_image:
                break
        for k, (sc, x, y) in enumerate(chosen, 1):
            tag = f"{ppi:.0f}ppi" if ppi else "unkppi"
            cf = f"{name}__crop{k:02d}_{tag}.jpg"
            im.crop((x, y, x + C, y + C)).save(out / cf, quality=a.quality)
            rows.append([cf, f.name, fam, f"{ppi:.0f}" if ppi else "unk", x, y, f"{sc:.2f}"])
        print(f"[{i}/{len(files)}] {f.name}: {len(chosen)} crops at {f'{ppi:.0f} ppi' if ppi else 'unknown scale'}")
    with open(out / "index.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["file", "source", "family", "ppi", "x", "y", "detail_score"]); wr.writerows(rows)
    print(f"done: {len(rows)} files in {out}. Zip the folder and send it (keep zips under 30 MB).")


if __name__ == "__main__":
    main()
