"""2x print upscale with a procedural detail pass (deterministic "super-resolution").

Rather than inventing texture with a neural upscaler, this enlarges the engine's maps with Lanczos
and then synthesises the detail that only exists at the higher resolution, in register across maps:
  * linen weave (0.7 mm pitch) in the height map, buried where paint is thick, visible where thin
  * slight pigment pooling in weave valleys where the paint film is thin (albedo, +-2%)
  * fine pigment grain (albedo, ~+-1%)
Normals, preview and height encoding are then recomputed at the new resolution.

Usage: python upscale_detail.py IN_DIR OUT_DIR [--factor 2] [--seed 1]
"""
import argparse
import json
import math
import resource
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None
CHUNK = 3_000_000


def chunks(h, w, budget=CHUNK):
    step = max(1, budget // w)
    for a in range(0, h, step):
        yield a, min(h, a + step)


def coarse_noise(rng, h, w, cell):
    """Smooth unit noise, generated coarse and resized (memory-light)."""
    gh, gw = int(math.ceil(h / cell)) + 3, int(math.ceil(w / cell)) + 3
    g = rng.normal(size=(gh, gw)).astype(np.float32)
    return g


def sample(g, h, w, a, b):
    gh, gw = g.shape
    mx = (np.arange(w, dtype=np.float32) + 0.5) * (gw - 3) / w + 1
    my = (np.arange(a, b, dtype=np.float32) + 0.5) * (gh - 3) / h + 1
    MX, MY = np.meshgrid(mx, my)
    return cv2.remap(g, MX, MY, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--weave-pitch-mm", type=float, default=0.7)
    ap.add_argument("--preview-exaggeration", type=float, default=3.0)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    a = ap.parse_args()
    t0 = time.time()
    src, out = Path(a.inp), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "metadata.json").read_text())
    f = a.factor
    rng = np.random.default_rng(a.seed)

    alb = np.asarray(Image.open(src / "albedo.png").convert("RGB"))
    h0, w0 = alb.shape[:2]; H, W = h0 * f, w0 * f
    if meta.get("dpi"):
        dpi = float(meta["dpi"]) * f
        px_mm = 25.4 / dpi
    else:
        px_mm = meta["canvas_mm"][0] / (w0 * f)
        dpi = round(25.4 / px_mm, 2)
    alb = cv2.resize(alb, (W, H), interpolation=cv2.INTER_LANCZOS4)
    for ya, yb in chunks(H, W):   # unsharp mask, radius ~1.5 px at the new resolution
        a0, b0 = max(0, ya - 8), min(H, yb + 8)
        blk = alb[a0:b0].astype(np.float32)
        sharp = blk + 0.7 * (blk - cv2.GaussianBlur(blk, (0, 0), 1.6))
        alb[ya:yb] = np.clip(sharp[ya - a0: ya - a0 + yb - ya], 0, 255)

    scale_mm = meta["height_encoding"]["scale_mm"]
    hgt = cv2.imread(str(src / "height_16bit.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * (scale_mm / 65535)
    hgt = cv2.resize(hgt, (W, H), interpolation=cv2.INTER_LANCZOS4)
    rough = cv2.resize(np.asarray(Image.open(src / "roughness.png")), (W, H), interpolation=cv2.INTER_LINEAR)
    mat = cv2.resize(np.asarray(Image.open(src / "material_id.png")), (W, H), interpolation=cv2.INTER_NEAREST)

    # Detail synthesis ---------------------------------------------------------------------------
    pitch = a.weave_pitch_mm / px_mm
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from weave import Weave
    weave = Weave(rng, H, W, pitch)
    grain_g = coarse_noise(rng, H, W, 0.6 / px_mm)   # ~0.6 mm pigment mottling
    gesso_mm = 0.02
    for ya, yb in chunks(H, W):
        wv = weave.rows(ya, yb)
        film = np.clip(hgt[ya:yb] - gesso_mm, 0, None)               # paint thickness, mm
        show = np.exp(-film / 0.06)                                   # thick paint buries the weave
        hgt[ya:yb] += 0.012 * wv * show
        pool = 1.0 - 0.02 * np.clip(-wv, 0, 1) * show                 # thin paint pools in valleys
        grain = 1.0 + 0.010 * sample(grain_g, H, W, ya, yb) + 0.006 * rng.standard_normal((yb - ya, W), dtype=np.float32)
        alb[ya:yb] = np.clip(alb[ya:yb].astype(np.float32) * (pool * grain)[..., None], 0, 255)
    hgt = np.clip(hgt, 0, None)

    # Export -------------------------------------------------------------------------------------
    tags = {"dpi": (dpi, dpi)}
    Image.fromarray(alb, "RGB").save(out / "albedo.png", **tags)
    Image.fromarray(alb, "RGB").save(out / "albedo.jpg", quality=a.jpeg_quality, subsampling=0, **tags)
    peak = float(hgt.max()); scale2 = math.ceil(peak * 1.25 * 100) / 100
    h16 = np.empty((H, W), np.uint16)
    for ya, yb in chunks(H, W):
        h16[ya:yb] = np.rint(np.clip(hgt[ya:yb] / scale2, 0, 1) * 65535)
    cv2.imwrite(str(out / "height_16bit.png"), h16); del h16
    Image.fromarray(rough, "L").save(out / "roughness.png", **tags)
    Image.fromarray(mat, "L").save(out / "material_id.png", **tags)
    z = gaussian_filter(hgt, 0.6); del hgt
    n16 = np.empty((H, W, 3), np.uint16); n8 = np.empty((H, W, 3), np.uint8); lit = np.empty((H, W, 3), np.uint8)
    light = np.array([-0.45, -0.55, 0.70], np.float32); light /= np.linalg.norm(light)
    ex = a.preview_exaggeration
    for ya, yb in chunks(H, W, CHUNK // 2):
        a0, b0 = max(0, ya - 1), min(H, yb + 1)
        gy, gx = np.gradient(z[a0:b0], px_mm)
        gy, gx = gy[ya - a0: ya - a0 + yb - ya], gx[ya - a0: ya - a0 + yb - ya]
        n = np.stack([-gx, -gy, np.ones_like(gx)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
        n16[ya:yb] = np.rint((n * 0.5 + 0.5) * 65535)
        ns = np.stack([-gx * ex, -gy * ex, np.ones_like(gx)], 2); ns /= np.linalg.norm(ns, axis=2, keepdims=True)
        n8[ya:yb] = np.clip(np.rint((ns * 0.5 + 0.5) * 255), 0, 255)
        shade = np.clip((ns * light).sum(2), 0, 1)
        lit[ya:yb] = np.clip(alb[ya:yb].astype(np.float32) * (1 + 0.45 * (shade - light[2]))[..., None], 0, 255)
    del z
    cv2.imwrite(str(out / "normal_16bit_physical.png"), n16[..., ::-1]); del n16
    Image.fromarray(n8, "RGB").save(out / "normal.png", **tags); del n8
    Image.fromarray(lit, "RGB").save(out / "artwork_preview.png", **tags)
    Image.fromarray(lit, "RGB").save(out / "artwork_preview.jpg", quality=a.jpeg_quality, subsampling=0, **tags)

    m2 = dict(meta)
    m2.update({
        "version": meta["version"] + "+upscale",
        "canvas_px": [W, H], "dpi": dpi, "pixel_pitch_mm": round(px_mm, 5),
        "upscale": {"factor": f, "method": "Lanczos-4 + procedural detail pass (linen weave in height, "
                    "thin-film pigment pooling and grain in albedo)", "source_px": [w0, h0], "seed": a.seed,
                    "weave_pitch_mm": a.weave_pitch_mm, "weave_pitch_px": round(pitch, 2)},
        "height_encoding": {"file": "height_16bit.png", "dtype": "uint16", "scale_mm": scale2,
                            "decode": "height_mm = value / 65535 * scale_mm", "peak_mm": round(peak, 4)},
        "preview_relief_exaggeration": ex,
    })
    (out / "metadata.json").write_text(json.dumps(m2, indent=2))
    print(f"upscaled {w0}x{h0} -> {W}x{H} @ {dpi} dpi in {time.time() - t0:.0f}s, "
          f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


if __name__ == "__main__":
    main()
