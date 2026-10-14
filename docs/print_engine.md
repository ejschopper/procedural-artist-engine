# Procedural Artist Engine v1.7-print

v1.6's colour-field painter, made resolution-independent for print, plus a 2× procedural upscaler.

## Two ways to get 30×40" at 300 dpi (9000 × 12000 px)

```
pip install numpy scipy opencv-python pillow

# A. Native render (best quality; ~30 min and ~5–6 GB RAM on a 2-core machine)
python engine_print.py --reference ref_IMG_0443.jpeg --size-in 30 40 --dpi 300 --out print_30x40

# B. Render at 150 dpi, then 2x upscale with a procedural detail pass (~10 min, ~2.6 GB RAM)
python engine_print.py --reference ref_IMG_0443.jpeg --size-in 30 40 --dpi 150 --out r150
python upscale_detail.py r150 print_30x40_up
```

The 150 dpi render is also a true 300 dpi 15×20" print, because it has the same pixel count. Strokes are sized relative to the canvas, so the composition looks the same at every size. For a 15×20" print with the weave and relief at the right physical scale, use `--size-in 15 20 --dpi 300`.

Rendering is deterministic: the same command and seed reproduce byte-identical files on any machine with the same library versions.

## What changed from v1.6

- **Resolution independence.** Stroke counts follow canvas area, not pixel count. Bristle count, along-stroke texture, touch-down ramp and tooth grain are all in physical units. At 850 px the output is pixel-identical to v1.6.
- **Linen weave.** A plain-weave canvas (0.7 mm thread pitch, with thread slubs and wander) is added to the tooth and height once threads span 5 px or more (about 180 dpi and up). Dry brushwork catches on the threads.
- **Anti-aliased bristle ends** at print resolution, so stroke ends no longer stair-step.
- **Memory.** Masks are computed per stroke tile, large strokes are processed in tiles, and noise is evaluated on crops. Fills and exports run in row chunks.
- **Print metadata.** `--size-in W H --dpi N` options; PNGs carry dpi tags, and metadata records size in inches, dpi and weave settings.

## Upscaler (`upscale_detail.py`)

Lanczos-4 2× enlargement of every map, then a deterministic detail pass that adds what only exists at the higher resolution, in register across maps:
- linen weave in the height map, buried under thick paint;
- slight pigment pooling in weave valleys where the paint film is thin;
- fine pigment grain.

Normals and the preview are then recomputed. This is used instead of neural super-resolution, which invents texture the height and normal maps would not match.

## Measured (2-core, 7 GB workspace)

| Render | Pixels | Time | Peak RAM |
|---|---|---|---|
| 850 px reference | 0.9 MP | 14 s | <0.5 GB |
| 30×40 @ 75 dpi | 6.75 MP | 95 s | ~0.6 GB |
| 30×40 @ 150 dpi | 27 MP | ~7 min | 1.4 GB |
| 2× upscale to 300 dpi | 108 MP | 3 min | 2.6 GB |

## Files per render

`albedo.png` (print this), `artwork_preview.png` (raking-light view, not for print), `height_16bit.png` (mm; decode from metadata), `normal_16bit_physical.png`, `normal.png`, `roughness.png`, `material_id.png`, `metadata.json`. The upscaler also writes JPEG copies of the albedo and preview.

## Limits

Same as v1.6: colour-field grammar only, RGB mixing, relief values not calibrated. Colours are sRGB with no print profile; soft-proof before printing.
