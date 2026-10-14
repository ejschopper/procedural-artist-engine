# Mark library: original paintings from the artist's own gestures

**Decision (14 Oct 2026):** output paintings are composed from a library of the artist's recorded *marks*
(individual gestures with true colour, height and normals), placed, deformed, recoloured and layered by the
procedural planner. No region of an existing painting is reproduced. Whole-crop rearrangement (band transfer)
is kept only as a realism benchmark, not for output.

## Pipeline

1. **Capture** (artist): 4-light sets on a tripod. Same area, camera fixed, light from top / bottom / left / right
   at the same low angle (30–45°), plus one shot with a scale reference (coin or ruler). True 300 ppi.
2. **Reconstruct** (engine): photometric stereo per set -> shadow-free albedo + normals -> integrated height (mm,
   scaled by the reference). Check: relight albedo+height under each capture light and compare with the photo.
3. **Extract marks**: segment each set into individual gestures (lip, knife drag, dry-brush edge, scumble patch,
   fleck, scratch, drip, bare-canvas window). Each mark = albedo, height, normals, alpha, plus descriptors
   (type, size, direction, medium, colour family, source painting).
4. **Compose** (planner): composition -> regions and boundaries -> mark placement rules per technique
   (density, direction, overlap order, size distribution, edge behaviour) -> marks warped to the plan,
   recoloured to the palette, layered with height-aware compositing (later marks sit on earlier ones;
   height adds; thin marks level the substrate weave).
5. **Output**: albedo, height (mm), normals, roughness by medium, capture-lit preview. All channels come from
   the same marks through the same transforms, so they agree by construction.
6. **Test**: realism statistics vs held-out real crops (texture, tone, lip shape, relief), repetition check
   (no mark instance dominant), relight check.

## Interim (before 4-light captures)

Marks extracted from existing single-light crops with *estimated* height (shape-from-shading, known light
direction). Labelled "estimated" in metadata.

## Data targets

Per technique: 3–5 paintings, 20–40 crops each; 4-light sets for at least 3–5 areas per painting.
