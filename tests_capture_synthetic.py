"""End-to-end test of capture reconstruction on a synthetic capture with known ground truth."""
import sys, numpy as np, cv2
sys.path.insert(0, 'engine')
from knife_bands import paint_knife_bands
from capture import light_vector, render, photometric_stereo, integrate_normals, check_relight
from PIL import Image, ImageDraw, ImageFont
px = 25.4 / 300
alb_t, h_t, _ = paint_knife_bands(900, 900, px, np.random.default_rng(5))
alb_t = np.uint8(alb_t)
el = 35.0
names = ['top', 'bottom', 'left', 'right']
lights = [light_vector(d, el) for d in names]
shots = [render(alb_t, h_t, px, l, ambient=0.08) for l in lights]          # what the camera would see
alb_r, n_r = photometric_stereo(shots, lights)
h_r = integrate_normals(n_r, px)
# accuracy vs ground truth (relief compared after removing broad tilt; albedo by correlation)
hp = lambda x, s=40: x - cv2.GaussianBlur(x, (0, 0), s)
corr_h = np.corrcoef(hp(h_t).ravel(), hp(h_r).ravel())[0, 1]
scale = np.std(hp(h_t)) / (np.std(hp(h_r)) + 1e-9)
L = lambda im: im.astype(np.float32).mean(2)
corr_a = np.corrcoef(L(alb_t).ravel(), L(alb_r).ravel())[0, 1]
gy, gx = np.gradient(h_t, px); n_t = np.stack([-gx, -gy, np.ones_like(gx)], 2); n_t /= np.linalg.norm(n_t, axis=2, keepdims=True)
ang = np.degrees(np.arccos(np.clip((n_t * n_r).sum(2), -1, 1)))
print(f'height detail correlation {corr_h:.3f}  (relief amplitude ratio true/recovered {scale:.2f})')
print(f'albedo correlation {corr_a:.3f}')
print(f'normal angle error: median {np.median(ang):.2f} deg, 95% {np.percentile(ang, 95):.2f} deg')
print('relight check per light:', [round(c, 3) for c in check_relight(alb_r, h_r * scale, px, shots, lights)])
f = ImageFont.truetype('DejaVuSans.ttf', 20)
hv = lambda h: np.stack([np.uint8(255 * (h - h.min()) / (np.ptp(h) + 1e-9))] * 3, 2)
tiles = [('Synthetic shot: light from top', shots[0]), ('...light from bottom', shots[1]), ('True colour', alb_t), ('Recovered colour', alb_r),
         ('True height', hv(h_t)), ('Recovered height', hv(h_r))]
o = Image.new('RGB', (3 * 455, 2 * 490), (246, 244, 239)); d = ImageDraw.Draw(o)
for k, (t, im) in enumerate(tiles):
    x, y = (k % 3) * 455, (k // 3) * 490
    o.paste(Image.fromarray(im).crop((200, 200, 650, 650)), (x + 3, y + 36)); d.text((x + 8, y + 8), t, fill=(30, 30, 30), font=f)
o.save('/home/claude/lighting/capture_synthetic_test.jpg', quality=90)
