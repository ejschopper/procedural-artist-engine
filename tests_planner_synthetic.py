"""Planner end-to-end on the synthetic capture: marks -> two plans (bands, fields) -> compose -> lit previews.
Run tests_marks_synthetic.py first (it writes the synthetic capture folder)."""
import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, 'engine')
from capture import light_vector, render
from intake import load_area
from marks import extract_marks, compose
from planner import plan_bands, plan_fields, place
from PIL import Image, ImageDraw, ImageFont

S = Path('/tmp/claude-0/-home-claude-procedural-artist-engine/c89ec983-9bc5-567e-af91-9e8922b59e27/scratchpad')
area = load_area(S / 'synthetic_capture/area01')
marks, fill = extract_marks(area, source='synthetic/area01')
px = area['px_mm']
print(f'library: {len(marks)} marks', {k: sum(m.kind == k for m in marks) for k in ('lip', 'drag', 'daub', 'fleck')})
palette = [(88, 98, 108), (70, 80, 92), (104, 112, 118), (60, 66, 74), (120, 126, 128)]
H, W = 1400, 1100                                                       # 119 x 93 mm at 300 ppi
tiles = []
for name, fn in (('bands', plan_bands), ('fields', plan_fields)):
    rng = np.random.default_rng(21)
    t0 = time.time()
    plan = fn(H, W, px, rng, palette)
    pl, ground, rep = place(plan, marks, rng, fill=fill)
    out = compose(H, W, px, pl, marks, ground)
    print(f'{name}: {rep} | height p1 {np.percentile(out["height_mm"], 1):.3f} p99 {np.percentile(out["height_mm"], 99):.3f} mm '
          f'| {time.time() - t0:.1f}s')
    lit = render(out['albedo'], out['height_mm'], px, light_vector('bottom', 35), ambient=0.3)
    lit2 = render(out['albedo'], out['height_mm'], px, light_vector((-0.6, -1.0), 35), ambient=0.3)
    h = out['height_mm']; hv = np.stack([np.uint8(255 * (h - h.min()) / (np.ptp(h) + 1e-9))] * 3, 2)
    tiles += [(f'{name}: colour', out['albedo']), (f'{name}: height', hv), (f'{name}: lit from bottom', lit),
              (f'{name}: lit from upper left', lit2)]
    cv2.imwrite(str(S / f'planner_{name}_lit.png'), lit[..., ::-1])
f = ImageFont.truetype('DejaVuSans.ttf', 18)
tw, th = 380, int(380 * H / W)
o = Image.new('RGB', (4 * (tw + 10) + 10, 2 * (th + 40) + 10), (246, 244, 239)); d = ImageDraw.Draw(o)
for k, (t, im) in enumerate(tiles):
    cx, cy = 10 + (k % 4) * (tw + 10), 10 + (k // 4) * (th + 40)
    d.text((cx, cy), t, fill=(30, 30, 30), font=f)
    o.paste(Image.fromarray(im).resize((tw, th), Image.LANCZOS), (cx, cy + 26))
o.save(S / 'planner_synthetic_test.jpg', quality=90)
print('saved', S / 'planner_synthetic_test.jpg')
