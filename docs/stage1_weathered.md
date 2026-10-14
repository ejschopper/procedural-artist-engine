# Stage 1 — Weathered field painter (v1.8-alpha)

Paints new compositions in the weathered/atmospheric field family: 2–4 horizontal bands or vertical panels, a chosen boundary type between each pair, and physical wear.

## Run

```
pip install numpy scipy opencv-python pillow
python weathered.py --seed 2 --width 850 --out preview_s2                       # 30-second preview
python weathered.py --seed 2 --size-in 30 40 --dpi 150 --out s2_150             # ~21 min, 1.6 GB
python upscale_detail.py s2_150 s2_300                                            # 2x to 300 dpi, ~3.5 min
python weathered.py --seed 9 --scheme navy_cream --orientation vertical ...       # steer a render
```

The 150 dpi render is also a true 300 dpi 15×20" file.

## How a painting is made

1. **Toned ground** in a contrasting under-colour. Saved as reveal layer S0.
2. **First coat** per zone, mixed toward a neighbour and the accent colour. Saved as reveal layer S1.
3. **Body coats** with bristle-brush strokes, tonal variants and a drier second coat.
4. **Mottling**: uneven absorption, stretched along the bands.
5. **Boundaries**: one of fringe (dry drags across the edge), ragged (scumble plus drips), fog (wide translucent veils), hard, or scored (an incised line). An optional accent stripe can ride a boundary.
6. **Scrape-back**: wear concentrates at canvas edges, boundaries and broad zones. It shows as abrasion of the tooth peaks, crisp-edged flaking, granular pits and tapered knife smears, revealing S1 or S0 below.
7. **Scumbles, granular edge wear, marks** (scratches, sgraffito, tally, X, dry bar, daubs, drips) and a final thin glaze.

Nine curated colour schemes are included. Every composition is sampled from the seed; nothing is copied from the reference library.

## Checks

- `reference_library/regression.py weathered_field <render dirs>` compares lightness spread, chroma, fine and mid texture, edge darkening and band strength with the p10–p90 range of the 27 weathered and atmospheric references, widened by 15%.
- The v1.6 colour-field render at 850 px remains pixel-identical, so the print-scale changes don't alter old results.

## Known limits

- Reveal layers are per pass (S0, S1), not per stroke.
- The scratched zigzag (sgraffito) reads as a flat line at 100%.
- Vertical splits sit outside the reference statistics, because the family is mostly horizontal.
- Navy/cream can run more contrasty than the references.
- Colour mixing is RGB, and relief values are not calibrated.
