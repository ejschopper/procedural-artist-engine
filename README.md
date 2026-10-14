# Procedural Artist Engine

Generates new paintings at print scale (e.g. 30×40" at 300 dpi) with colour, height, normal and roughness maps.

**Architecture (v1.9):** the procedural engine plans composition and structure; fine surface detail comes from the
artist's own 100% crops (real paint), arranged to the plan. A physically based capture-lighting pass renders a
"photographed original"; the plain colour + height map serve textured / relief printing.

| Path | What |
|---|---|
| `engine/engine_print.py` | Resolution-independent stroke engine (bristle brush, substrate + weave levelling, chunked export) |
| `engine/weathered.py` | Weathered-field painter (zones, boundaries, scrape-back, marks) |
| `engine/knife_bands.py`, `band_transfer.py`, `render_bands_full.py` | Knife-drag bands; band-aware transfer from real crops; full-size renderer |
| `engine/exemplar.py` | Guided patch synthesis from real crops |
| `engine/lighting.py` | Capture lighting (shadows, occlusion, gloss, falloff) |
| `engine/upscale_detail.py`, `weave.py` | 2× print upscale with procedural detail; plain-weave substrates |
| `library/` | Reference-library tools, tags, manifest (images stored outside git) |
| `tools/make_detail_crops.py` | Run on the artist's machine: true-scale 100% crops for the library |
| `legacy/` | v1.6 engine and comparison metrics |

Artwork images are **not** in this repository. Keep them in a private store (local drive or cloud folder) and point
`library/` at it. See `docs/` for stage notes.
