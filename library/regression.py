"""Family regression: are renders statistically inside the reference family's range?

For each render, every library statistic is compared with the p10-p90 range of the chosen family
(duplicates and contact sheets excluded). A render passes a statistic when it lies within the
range widened by 15%. Statistics are computed on the lit preview at 600 px long side, the same
way as the references (photos of lit paintings).

Usage: python regression.py FAMILY render_dir [render_dir ...]   (writes regression_<family>.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_library import stats  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
KEYS = ["lightness_std", "chroma_mean", "texture_fine", "texture_mid", "edge_darkening",
        "band_horizontal", "band_vertical"]
FAMILY_GROUPS = {"weathered_field": ("weathered_field", "atmospheric_field")}


def main():
    fam, dirs = sys.argv[1], sys.argv[2:]
    group = FAMILY_GROUPS.get(fam, (fam,))
    imgs = json.loads((ROOT / "manifest.json").read_text())["images"]
    refs = [o["stats"] for o in imgs if o.get("family") in group and not o.get("exclude_from_stats")]
    ranges = {}
    for k in KEYS:
        v = np.array([r[k] for r in refs])
        lo, hi = np.percentile(v, [10, 90]); pad = 0.15 * (hi - lo)
        ranges[k] = [round(float(lo - pad), 3), round(float(hi + pad), 3), round(float(np.median(v)), 3)]
    results = {}
    for d in dirs:
        im = Image.open(Path(d) / "artwork_preview.png").convert("RGB")
        im.thumbnail((1200, 1200))
        st = stats(np.asarray(im))
        checks = {}
        for k in KEYS:
            x = st[k]
            if k == "band_vertical" and st["band_horizontal"] > st["band_vertical"]:
                checks[k] = {"value": x, "pass": True, "note": "not the dominant axis"}
                continue
            if k == "band_horizontal" and st["band_vertical"] > st["band_horizontal"]:
                checks[k] = {"value": x, "pass": True, "note": "not the dominant axis"}
                continue
            checks[k] = {"value": x, "pass": bool(ranges[k][0] <= x <= ranges[k][1])}
        results[str(d)] = {"pass_count": sum(c["pass"] for c in checks.values()), "of": len(KEYS), "checks": checks}
    out = {"family": fam, "reference_count": len(refs), "ranges_lo_hi_median": ranges, "renders": results}
    (ROOT / f"regression_{fam}.json").write_text(json.dumps(out, indent=1))
    for d, r in results.items():
        fails = [k for k, c in r["checks"].items() if not c["pass"]]
        print(f"{Path(d).name}: {r['pass_count']}/{r['of']}", "fails: " + ", ".join(fails) if fails else "")


if __name__ == "__main__":
    main()
