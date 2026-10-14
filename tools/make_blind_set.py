"""Build a blind A/B round: random 100% windows from real crops and from an engine render, shuffled.

python tools/make_blind_set.py --real 'library/closeups/structures8_crop*.png' \
    --engine /home/claude/bands_44cm_v2/painting.png --n 8 --size 560 --round r1 --out blind_r1.html \
    --exclude 'structures8_crop01.png:0,0,1500,600;0,0,600,1500'

Writes the page (images embedded as JPEG data URIs, no resampling) and <out>.key.json with the answer key.
The key is also in the page, lightly encoded, so the page can reveal the score after the last vote.
"""
import argparse
import base64
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def chroma95(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.percentile(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128), 95))


def windows(img, n, size, rng, min_std=6.0, max_chroma=None, max_light=None, exclude=()):
    H, W = img.shape[:2]
    out, tries = [], 0
    while len(out) < n and tries < 400 * n:
        tries += 1
        y, x = rng.randrange(0, H - size), rng.randrange(0, W - size)
        w = img[y:y + size, x:x + size]
        if cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).std() < min_std:     # skip empty/flat windows
            continue
        # fair test: a real window holding colours the engine never makes (e.g. a collage edge) gives itself away
        if max_chroma is not None and chroma95(w) > max_chroma:
            continue
        # ... and likewise much lighter areas (bare canvas, collage paper) than any engine window
        if max_light is not None and (cv2.cvtColor(w, cv2.COLOR_BGR2GRAY) > max_light).mean() > 0.02:
            continue
        if any(x < x1 and x + size > x0 and y < y1 and y + size > y0 for x0, y0, x1, y1 in exclude):
            continue
        if any(abs(y - a) < size * 0.7 and abs(x - b) < size * 0.7 for a, b, _ in out):
            continue
        out.append((y, x, w))
    return out


def jpeg_uri(bgr, q=90):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--engine", required=True, nargs="+")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--round", default="r1")
    ap.add_argument("--label", default="Round 1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="name.png:x0,y0,x1,y1[;x0,y0,x1,y1] px regions of a real crop that hold non-paint material "
                         "(bare canvas, collage) the engine does not make; windows touching them are skipped")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    excl = {}
    for e in a.exclude:
        name, rects = e.split(":", 1)
        excl[name] = [tuple(int(v) for v in r.split(",")) for r in rects.split(";")]
    items = []
    eper = [a.n // len(a.engine) + (1 if i < a.n % len(a.engine) else 0) for i in range(len(a.engine))]
    for f, k in zip(a.engine, eper):
        for y, x, w in windows(cv2.imread(f), k, a.size, rng):
            items.append({"truth": "engine", "src": f"{Path(f).parent.name}/{Path(f).name}@{x},{y}", "img": w})
    cmax = max(chroma95(it["img"]) for it in items) * 1.15
    lmax = max(float(np.percentile(cv2.cvtColor(it["img"], cv2.COLOR_BGR2GRAY), 99.5)) for it in items) + 8
    reals = sorted(glob.glob(a.real))
    per = [a.n // len(reals) + (1 if i < a.n % len(reals) else 0) for i in range(len(reals))]
    got = {}
    for f, k in zip(reals, per):
        got[f] = windows(cv2.imread(f), k, a.size, rng, max_chroma=cmax, max_light=lmax,
                         exclude=excl.get(Path(f).name, ()))
    short = a.n - sum(len(v) for v in got.values())
    for f in reals:                                    # top up from crops with room to spare
        if short <= 0:
            break
        more = windows(cv2.imread(f), len(got[f]) + short, a.size, rng, max_chroma=cmax, max_light=lmax,
                       exclude=excl.get(Path(f).name, ()))
        short -= len(more) - len(got[f]); got[f] = more
    for f, ws in got.items():
        for y, x, w in ws:
            items.append({"truth": "real", "src": f"{Path(f).name}@{x},{y}", "img": w})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"{a.round}-{i + 1:02d}"
    key = {it["id"]: {"truth": it["truth"], "src": it["src"]} for it in items}
    Path(a.out + ".key.json").write_text(json.dumps(key, indent=1))
    # light encoding so the answer isn't readable at a glance in the page source
    enc = base64.b64encode(bytes(b ^ 0x5A for b in json.dumps({k: v["truth"] for k, v in key.items()}).encode())).decode()
    data = {"round": a.round, "label": a.label, "size_px": a.size,
            "items": [{"id": it["id"], "src": jpeg_uri(it["img"])} for it in items], "k": enc}
    tpl = (HERE / "blind_test_template.html").read_text()
    Path(a.out).write_text(tpl.replace("/*__DATA__*/null", json.dumps(data)))
    print(f"{len(items)} items -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
