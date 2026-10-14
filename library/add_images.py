"""Add new reference images to the library.

Usage: python add_images.py FAMILY img1.jpg [img2.png ...]
Then run build_library.py. Unknown family names are allowed (they become new families).
Add "crop_frac": [x0, y0, x1, y1] to tags.json for photos where auto-crop misses the painting.
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent


def main():
    fam, files = sys.argv[1], sys.argv[2:]
    index = json.loads((ROOT / "index.json").read_text())
    tags = json.loads((ROOT / "tags.json").read_text())
    known = {r["source_file"] for r in index}
    n = len(index)
    for f in files:
        if Path(f).name in known:
            print("already in library:", f)
            continue
        im = Image.open(f); im.seek(0); im = im.convert("RGB")
        n += 1
        rid = f"ref_{n:03d}"
        im.save(ROOT / "images" / f"{rid}.jpg", quality=92)
        index.append({"id": rid, "source_file": Path(f).name, "size": list(im.size)})
        tags[rid] = {"family": fam}
        print("added", rid, f)
    (ROOT / "index.json").write_text(json.dumps(index, indent=1))
    (ROOT / "tags.json").write_text(json.dumps(tags, indent=1))


if __name__ == "__main__":
    main()
