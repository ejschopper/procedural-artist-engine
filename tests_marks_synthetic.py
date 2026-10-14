"""Synthetic end-to-end: make a capture folder (4 lights + notes) from a procedural surface with known truth,
run intake, extract marks, compose a NEW arrangement, and render it under a new light."""
import sys, numpy as np, cv2, shutil
from pathlib import Path
sys.path.insert(0, 'engine')
from knife_bands import paint_knife_bands
from capture import light_vector, render
from intake import load_area, print_report
from marks import extract_marks, compose, normals_from_height
from PIL import Image, ImageDraw, ImageFont
px = 25.4 / 300
D = Path('/tmp/claude-0/-home-claude-procedural-artist-engine/c89ec983-9bc5-567e-af91-9e8922b59e27/scratchpad/synthetic_capture/area01')
shutil.rmtree(D, ignore_errors=True); D.mkdir(parents=True)
alb_t, h_t, _ = paint_knife_bands(1200, 1200, px, np.random.default_rng(9), lip_mm=(0.15, 0.6))
alb_t = np.uint8(alb_t)
for d in ('top', 'bottom', 'left', 'right'):
    cv2.imwrite(str(D / f'synthetic_area01_light-{d}.png'), render(alb_t, h_t, px, light_vector(d, 35), ambient=0.08)[..., ::-1])
h16 = np.uint16(np.clip(h_t / 1.0, 0, 1) * 65535)
cv2.imwrite(str(D / 'synthetic_area01_height.png'), h16)
gy, gx = np.gradient(h_t, px); n = np.stack([-gx, -gy, np.ones_like(gx)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
n_gl = n.copy(); n_gl[..., 1] *= -1                                        # write as OpenGL (+Y up)
cv2.imwrite(str(D / 'synthetic_area01_normal.png'), np.uint16((n_gl * 0.5 + 0.5) * 65535)[..., ::-1])
(D / 'notes.txt').write_text("painting: synthetic\nppi: 300\nheight_units: 0-65535 = 0.0-1.0 mm\n"
                             "normal_convention: OpenGL (+Y up)\nlight_elevation_deg: 35\n")
area = load_area(D)
print('INTAKE REPORT'); print_report(area)
marks, fill = extract_marks(area, source='synthetic/area01')
kinds = {}
for m in marks: kinds[m.kind] = kinds.get(m.kind, 0) + 1
print(f'{len(marks)} marks extracted: {kinds}; peak relief mm (top 5): {[m.meta["peak_mm"] for m in marks[:5]]}')
# new composition: a smaller canvas, marks re-placed in a new order, staggered, some mirrored by rotation 180
rng = np.random.default_rng(3); H = W = 900
ground = {"albedo": cv2.resize(fill["albedo"], (W, H)), "height": cv2.resize(fill["height"], (W, H)), "gloss": np.full((H, W), 0.4, np.float32)}
pl = []
lips = [i for i, m in enumerate(marks) if m.kind in ('lip', 'drag')]
y = 40
while y < H - 20 and lips:
    x = rng.uniform(-200, 200)
    while x < W:
        i = int(rng.choice(lips)); mw = marks[i].alpha.shape[1]
        pl.append(dict(mark=i, x=x + mw / 2, y=y, angle_deg=(180 if rng.random() < 0.3 else 0) + rng.normal(0, 1.5),
                       scale=rng.uniform(0.95, 1.05), brightness=rng.normal(0, 0.03), relief_gain=rng.uniform(0.85, 1.15)))
        x += mw * rng.uniform(0.7, 0.95)
    y += rng.uniform(70, 150)
out = compose(H, W, px, pl, marks, ground)
lit_new = render(out["albedo"], out["height_mm"], px, light_vector((-0.6, -1.0), 35), ambient=0.25)
lit_cap = render(out["albedo"], out["height_mm"], px, light_vector('bottom', 35), ambient=0.25)
print(f'composed {len(pl)} placements; height p1 {np.percentile(out["height_mm"], 1):.3f} p99 {np.percentile(out["height_mm"], 99):.3f} mm')
f = ImageFont.truetype('DejaVuSans.ttf', 20)
hv = lambda h: np.stack([np.uint8(255 * (h - h.min()) / (np.ptp(h) + 1e-9))] * 3, 2)
tiles = [('Source capture (light from bottom)', np.asarray(Image.open(D / 'synthetic_area01_light-bottom.png'))[:900, :900]),
         ('New composition: colour', out['albedo']), ('New composition: height', hv(out['height_mm'])),
         ('Lit from bottom', lit_cap), ('Lit from upper left', lit_new)]
o = Image.new('RGB', (5 * 460, 500), (246, 244, 239)); d = ImageDraw.Draw(o)
for k, (t, im) in enumerate(tiles):
    o.paste(Image.fromarray(im).resize((450, 450), Image.LANCZOS), (5 + k * 460, 40)); d.text((10 + k * 460, 8), t, fill=(30, 30, 30), font=f)
o.save('/home/claude/lighting/marks_synthetic_test.jpg', quality=90)
