"""Compare renders against a reference painting on what matters, not on pixel equality.

Metrics (each render is compared with the reference crop it was aiming at):
  colour_dE_lowfreq   mean CIE76 dE after a broad blur: composition + palette fidelity (lower = closer)
  texture_ratio_*     high-frequency luminance energy, render / reference, in the ground and the field
                      (1.0 = as much surface activity as the painting; v1.5-style blur scores << 1)
  lower_edge_px       10-90% transition width of the field's lower edge (the dry-brush fringe)
  hf_correlation      correlation of fine detail with the reference (near 0 = nothing traced/copied)

Usage: python compare.py REF.jpeg v15_dir v16_dir out_dir
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from engine_v16 import detect_canvas_box, wall_gain  # noqa: E402


def lab(img):
    return cv2.cvtColor(np.uint8(np.clip(img, 0, 255)), cv2.COLOR_RGB2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]


def hf(Lc, s=2.0):
    return Lc - cv2.GaussianBlur(Lc, (0, 0), s)


def lower_edge_width(img, fbox):
    """Median over columns of the 10-90% fall-off distance of chroma across the field's lower edge."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v * d) for v, d in zip(fbox, (w, h, w, h))]
    C = np.hypot(*np.moveaxis(lab(img)[..., 1:], -1, 0))
    widths = []
    for x in range(x0 + 10, x1 - 10, 3):
        col = cv2.GaussianBlur(C[:, x:x + 1], (1, 0), 1.0).ravel() if False else C[:, x]
        a = col[int(y1 - 0.08 * h): int(y1 + 0.10 * h)]
        a = np.convolve(a, np.ones(3) / 3, mode="same")
        hi = np.median(col[int(y0 + 0.5 * (y1 - y0)): int(y1 - 0.02 * h)]); lo = np.median(col[int(y1 + 0.12 * h): int(y1 + 0.16 * h)])
        if hi - lo < 10:
            continue
        t = (a - lo) / (hi - lo)
        i90 = np.where(t > 0.9)[0]; i10 = np.where(t > 0.1)[0]
        if len(i90) and len(i10):
            widths.append(i10[-1] - i90[-1])
    return float(np.median(widths)) if widths else float("nan")


def metrics(render, ref, fbox, bwidth):
    h, w = ref.shape[:2]
    Lr, Lf = lab(render), lab(ref)
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 0.015 * w)
    dE = float(np.sqrt(((blur(Lr) - blur(Lf)) ** 2).sum(2)).mean())
    x0, y0, x1, y1 = [int(v * d) for v, d in zip(fbox, (w, h, w, h))]
    field = np.zeros((h, w), bool); field[y0 + 8: y1 - 8, x0 + 8: x1 - 8] = True
    ground = np.zeros((h, w), bool)
    bx = int(bwidth * w * 1.6)
    ground[int(y1 + 0.15 * h): h - bx, bx: w - bx] = True
    out = {"colour_dE_lowfreq": round(dE, 2)}
    for name, m in [("ground", ground), ("field", field)]:
        out[f"texture_ratio_{name}"] = round(float(hf(Lr[..., 0])[m].std() / (hf(Lf[..., 0])[m].std() + 1e-6)), 2)
    out["lower_edge_px"] = lower_edge_width(render, fbox)
    a, b = hf(Lr[..., 0]).ravel(), hf(Lf[..., 0]).ravel()
    out["hf_correlation"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    return out


def label(img, text, h=34):
    im = Image.fromarray(img); W = im.width
    canvas = Image.new("RGB", (W, im.height + h), (246, 244, 239)); canvas.paste(im, (0, h))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        f = ImageFont.load_default()
    d.text((8, 6), text, fill=(30, 30, 30), font=f)
    return np.asarray(canvas)


def main(ref_path, v15_dir, v16_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta16 = json.loads((Path(v16_dir) / "metadata.json").read_text())
    meta15 = json.loads((Path(v15_dir) / "metadata.json").read_text())
    brief = meta16["brief"]
    rgb = np.array(Image.open(ref_path).convert("RGB"))
    r16 = np.array(Image.open(Path(v16_dir) / "artwork_preview.png").convert("RGB"))
    r15 = np.array(Image.open(Path(v15_dir) / "artwork_preview.png").convert("RGB"))
    H, W = r16.shape[:2]
    # v1.6 target: full detected canvas, wall-calibrated
    box = brief["reference_canvas_box_xyxy"]; gain = np.asarray(brief["photo_calibration"]["gain_rgb"], np.float32)
    ref16 = np.clip(rgb[box[1]:box[3], box[0]:box[2]].astype(np.float32) * gain, 0, 255)
    ref16 = cv2.resize(ref16, (W, H), interpolation=cv2.INTER_AREA)
    # v1.5 target: its own crop, uncalibrated
    x0, y0, x1, y1 = meta15["reference_crop_xyxy"]
    ref15 = cv2.resize(rgb[y0:y1, x0:x1].astype(np.float32), (r15.shape[1], r15.shape[0]), interpolation=cv2.INTER_AREA)
    # field box for v1.5 crop coordinates
    fb = brief["field_box_norm"]; cw, ch = box[2] - box[0], box[3] - box[1]
    fb15 = [((box[0] + fb[0] * cw) - x0) / (x1 - x0), ((box[1] + fb[1] * ch) - y0) / (y1 - y0),
            ((box[0] + fb[2] * cw) - x0) / (x1 - x0), ((box[1] + fb[3] * ch) - y0) / (y1 - y0)]
    bw = brief["border_width_norm_ltrb"][0]
    res = {
        "reference_crop_coverage": {
            "v1.5_crop_xyxy": meta15["reference_crop_xyxy"], "detected_canvas_xyxy": box,
            "v1.5_fraction_of_canvas_area": round(((min(x1, box[2]) - max(x0, box[0])) * (min(y1, box[3]) - max(y0, box[1]))) / (cw * ch), 3)},
        "v1.5": metrics(r15.astype(np.float32), ref15, fb15, bw),
        "v1.6": metrics(r16.astype(np.float32), ref16, fb, bw),
        "reference_self_check": {"lower_edge_px": lower_edge_width(ref16, fb)},
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    # Triptych at common height
    th = 900
    rs = lambda im: cv2.resize(np.uint8(np.clip(im, 0, 255)), (int(im.shape[1] * th / im.shape[0]), th), interpolation=cv2.INTER_AREA)
    gap = np.full((th + 34, 16, 3), 246, np.uint8)
    tri = np.concatenate([label(rs(ref16), "Reference (full canvas, calibrated)"), gap,
                          label(rs(r15), "v1.5"), gap, label(rs(r16), "v1.6")], 1)
    Image.fromarray(tri).save(out / "comparison_v15_v16.png")
    # Detail: lower edge of field, 2x
    def detail(img, f):
        h, w = img.shape[:2]
        xa, ya = int((f[0] + 0.12 * (f[2] - f[0])) * w), int((f[3] - 0.06) * h)
        cw_, ch_ = int(0.32 * w), int(0.17 * h)
        crop = np.uint8(np.clip(img[ya:ya + ch_, xa:xa + cw_], 0, 255))
        return cv2.resize(crop, (640, int(640 * ch_ / cw_)), interpolation=cv2.INTER_CUBIC)
    d = [detail(ref16, fb), detail(r15.astype(np.float32), fb15), detail(r16, fb)]
    hh = min(x.shape[0] for x in d); d = [x[:hh] for x in d]
    gap2 = np.full((hh + 34, 16, 3), 246, np.uint8)
    Image.fromarray(np.concatenate([label(d[0], "Reference: lower edge"), gap2, label(d[1], "v1.5"), gap2,
                                    label(d[2], "v1.6")], 1)).save(out / "detail_lower_edge.png")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])
