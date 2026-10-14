"""Capture intake: read one captured area folder, parse notes.txt, load and validate every map.

Expected folder (names are matched loosely, case-insensitive):
  light-top / light-bottom / light-left / light-right  (.tif/.png/.jpg)   optional if height+normal given
  height, normal, gloss (or roughness), metalness, albedo (or diffuse/basecolor)                optional
  notes.txt  (see docs/capture_notes_template.txt)

load_area(folder) -> dict(maps, meta, report)
  maps:   albedo uint8 HxWx3, height_mm float32, normal float32 HxWx3 (+Y up converted to image +Y down),
          gloss float32 0..1, metal float32 0..1, lights {dir: uint8}
  report: list of checks (shape agreement, bit depth, height range in mm, normal-vs-height agreement,
          which convention fits better, relight correlation per light if lights + height present)
"""
import re
from pathlib import Path

import cv2
import numpy as np

from capture import check_relight, integrate_normals, light_vector, photometric_stereo

KEYS = {"albedo": ["albedo", "diffuse", "basecolor", "base_color", "color"],
        "height": ["height", "depth", "displacement"],
        "normal": ["normal"],
        "gloss": ["gloss", "rough"],
        "metal": ["metal"]}
EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".exr")


def parse_notes(path):
    meta = {}
    if not Path(path).exists():
        return meta
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta


def _num(s, default=None):
    m = re.findall(r"[-+]?\d*\.?\d+", s or "")
    return float(m[0]) if m else default


def height_scale(meta, raw):
    """Turn raw height values into mm using notes ('0-65535 = 0.0-3.0 mm', 'value = microns', 'float, mm')."""
    u = (meta.get("height_units") or "").lower()
    if "micron" in u or "µm" in u or "um" in u:
        return raw.astype(np.float32) / 1000.0, "microns"
    m = re.findall(r"[-+]?\d*\.?\d+", u)
    if "=" in u and len(m) >= 4:
        r0, r1, mm0, mm1 = map(float, m[:4])
        return (raw.astype(np.float32) - r0) / (r1 - r0) * (mm1 - mm0) + mm0, "linear map"
    if raw.dtype.kind == "f":
        return raw.astype(np.float32), "float assumed mm"
    return None, "unknown: add height_units to notes.txt"


def _read(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f"cannot read {path}")
    if im.ndim == 3:
        im = im[..., :3][..., ::-1]                   # BGR -> RGB
    return im


def load_area(folder):
    folder = Path(folder)
    meta = parse_notes(folder / "notes.txt")
    files = [p for p in folder.iterdir() if p.suffix.lower() in EXT]
    maps, report, lights = {}, [], {}
    for p in files:
        n = p.stem.lower()
        for d in ("top", "bottom", "left", "right"):
            if "light" in n and d in n:
                lights[d] = _read(p)
        for k, names in KEYS.items():
            if any(x in n for x in names) and "light" not in n and k not in maps:
                maps[k] = (p, _read(p))
    shapes = {k: v[1].shape[:2] for k, v in maps.items()} | {f"light-{k}": v.shape[:2] for k, v in lights.items()}
    report.append(("same pixel grid", len(set(shapes.values())) <= 1, str(shapes)))
    ppi = _num(meta.get("ppi"), 300.0); px_mm = 25.4 / ppi
    out = {"meta": meta, "px_mm": px_mm, "lights": lights}
    if "albedo" in maps:
        a = maps["albedo"][1]
        out["albedo"] = (a / 257).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8)
    if "height" in maps:
        raw = maps["height"][1]
        raw = raw[..., 0] if raw.ndim == 3 else raw
        report.append(("height bit depth", raw.dtype != np.uint8, f"{raw.dtype} (8-bit loses fine relief)"))
        h, how = height_scale(meta, raw)
        if h is not None:
            out["height_mm"] = h - np.percentile(h, 0.5)
            report.append(("height range mm", True, f"{how}: p1 {np.percentile(h, 1):.3f} .. p99 {np.percentile(h, 99):.3f}"))
        else:
            report.append(("height units", False, how))
    if "normal" in maps:
        nm = maps["normal"][1].astype(np.float32)
        nm = nm / (65535.0 if maps["normal"][1].dtype == np.uint16 else 255.0) * 2 - 1
        nm /= np.linalg.norm(nm, axis=2, keepdims=True) + 1e-8
        conv = (meta.get("normal_convention") or "").lower()
        flip = "opengl" in conv or "+y up" in conv
        # verify against the height map: which green-channel sign matches the height gradient?
        if "height_mm" in out:
            gy, gx = np.gradient(cv2.GaussianBlur(out["height_mm"], (0, 0), 1.0), px_mm)
            agree_down = np.corrcoef((-gy).ravel(), nm[..., 1].ravel())[0, 1]
            agree_up = np.corrcoef((gy).ravel(), nm[..., 1].ravel())[0, 1]
            fits = "DirectX (+Y down)" if agree_down > agree_up else "OpenGL (+Y up)"
            report.append(("normal convention", True, f"notes say '{conv or '?'}', data fits {fits} "
                                                         f"(r={max(agree_down, agree_up):.2f})"))
            flip = agree_up > agree_down
        if flip:
            nm[..., 1] *= -1                          # store internally as image +Y down
        out["normal"] = nm
    for k in ("gloss", "metal"):
        if k in maps:
            g = maps[k][1]; g = g[..., 0] if g.ndim == 3 else g
            g = g.astype(np.float32) / (65535.0 if g.dtype == np.uint16 else 255.0)
            if k == "gloss" and "rough" in maps[k][0].stem.lower():
                g = 1 - g
            out[k] = g
    # light shots: reconstruct if no measured maps, and always run the relight check
    if len(lights) >= 3:
        el = _num(meta.get("light_elevation_deg"), 35.0)
        names = sorted(lights)
        lv = [light_vector(d, el) for d in names]
        shots = [lights[d] for d in names]
        if "albedo" not in out or "height_mm" not in out:
            alb, n = photometric_stereo(shots, lv)
            out.setdefault("albedo", alb)
            out.setdefault("normal", n)
            out.setdefault("height_mm", integrate_normals(n, px_mm))
            missing = [k for k in ("albedo", "normal", "height") if k not in maps]
            report.append(("reconstructed from lights", True, "photometric stereo filled: " + ", ".join(missing)))
        cors = check_relight(out["albedo"], out["height_mm"], px_mm, shots, lv)
        report.append(("relight check", min(cors) > 0.8, ", ".join(f"{d} {c:.2f}" for d, c in zip(names, cors))))
    out["report"] = report
    return out


def print_report(area):
    for name, ok, detail in area["report"]:
        print(f"  [{'ok ' if ok else 'FIX'}] {name}: {detail}")
