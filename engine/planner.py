"""Composer planner: composition plan -> mark placements (classical, deterministic per seed, no AI).

A *plan* divides the canvas into regions. Each region names a technique, a palette and (optionally) a
direction. The planner lays a colour/height ground per region, then places marks from the library with the
technique's rules, and returns placements for marks.compose().

Techniques (rules, not pixels; all sizes in mm, so they hold at any resolution):
  knife_bands  rows of lips/drags that follow the region's boundaries, overlapping like a knife pulled across
  scumble      broken drags scattered along a direction field
  daubs        short marks at any angle
  flecks       sparse small marks on top of everything

Repetition control: marks are drawn least-used-first, the same mark is never reused within `reuse_mm`
of an earlier copy, and the report gives the most-used mark's share (the blind test's repetition check).

plan_bands(H, W, px_mm, rng, palette, ...)  -> Plan  stacked bands with near-straight drifting boundaries
plan_fields(H, W, px_mm, rng, palette, ...) -> Plan  a few large warped regions with mixed techniques
place(plan, marks, rng)                     -> (placements, ground, report)
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

TECHNIQUES = {
    "knife_bands": dict(kinds=("lip", "drag"), layout="rows", angle_jitter=1.5, scale=(0.95, 1.05), step=(0.7, 0.95),
                        row_jitter_mm=0.8, flip180=0.3, relief=(0.85, 1.15), bright=0.03, own_colour=0.25, layer=1),
    "scumble": dict(kinds=("drag", "daub"), layout="scatter", coverage=0.9, angle_jitter=12, scale=(0.85, 1.1),
                    flip180=0.5, relief=(0.5, 0.9), bright=0.05, own_colour=0.2, layer=1),
    "daubs": dict(kinds=("daub", "drag", "fleck"), layout="scatter", coverage=1.1, angle_jitter=180, scale=(0.8, 1.15),
                  flip180=0.5, relief=(0.8, 1.2), bright=0.05, own_colour=0.35, layer=1),
    "flecks": dict(kinds=("fleck",), layout="scatter", coverage=0.04, angle_jitter=180, scale=(0.8, 1.2),
                   flip180=0.5, relief=(0.8, 1.2), bright=0.08, own_colour=0.6, layer=2),
}


@dataclass
class Region:
    mask: np.ndarray                      # H,W bool
    technique: str
    colour: tuple                         # base rgb of the ground (marks take most of their colour from it)
    angle_deg: float = 0.0                # main direction of marks
    order: int = 0                        # paint order between regions
    params: dict = field(default_factory=dict)   # per-region overrides of the technique rules
    rows: list = None                     # knife_bands: list of (y_of_x array) row centre lines


@dataclass
class Plan:
    H: int
    W: int
    px_mm: float
    regions: list
    ground_relief_mm: float = 0.03
    colour_var: float = 0.06              # low-frequency colour wander inside a region (fraction)


def smooth_field(H, W, rng, cell_px, amp=1.0):
    """Low-frequency random field: random values on a coarse grid, bicubic upsampled."""
    gh, gw = max(2, int(H / cell_px) + 2), max(2, int(W / cell_px) + 2)
    f = cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return f / (f.std() + 1e-6) * amp


def _boundary(W, px_mm, rng, base_px, drift_mm=1.0, wobble_mm=0.6):
    """Near-straight line: a gentle drift over the width plus its own small wobble (never synchronized)."""
    x = np.arange(W, dtype=np.float32)
    L = W * px_mm
    y = base_px + rng.uniform(-1, 1) * drift_mm / px_mm * (x / W)            # slight tilt
    for k in range(3):
        period = rng.uniform(0.25, 1.0) * L / px_mm
        y += rng.normal(0, wobble_mm / px_mm / (k + 1)) * np.sin(2 * np.pi * x / period + rng.uniform(0, 6.28))
    return y


def plan_bands(H, W, px_mm, rng, palette, band_mm=(14, 45), rows_per_band=None, row_mm=(6, 12),
               fleck_overlay=True):
    """Structures-like composition: horizontal bands top to bottom, each a knife_bands region in one colour."""
    mm = 1.0 / px_mm
    cuts = [-2 * mm]
    while cuts[-1] < H:
        cuts.append(cuts[-1] + rng.uniform(*band_mm) * mm)
    lines = [np.full(W, cuts[0], np.float32)] + [_boundary(W, px_mm, rng, c) for c in cuts[1:-1]] + \
            [np.full(W, H + 2 * mm, np.float32)]
    for i in range(1, len(lines)):                                           # keep bands at least 60% of planned
        lines[i] = np.maximum(lines[i], lines[i - 1] + 0.6 * (cuts[i] - cuts[i - 1]))
    yy = np.arange(H, dtype=np.float32)[:, None]
    regions = []
    for i in range(len(lines) - 1):
        top, bot = lines[i], lines[i + 1]
        mask = (yy >= top[None]) & (yy < bot[None])
        if mask.sum() < 50:
            continue
        rgb = np.array(palette[int(rng.integers(len(palette)))], np.float32) * rng.uniform(0.93, 1.07)
        h_mean = float((bot - top).mean()) * px_mm
        n_rows = rows_per_band or max(1, int(round(h_mean / rng.uniform(*row_mm))))
        rows = [top + (bot - top) * (j + 0.5) / n_rows for j in range(n_rows)]
        regions.append(Region(mask=mask, technique="knife_bands", colour=tuple(np.clip(rgb, 0, 255)),
                              angle_deg=0.0, order=i, rows=rows))
    if fleck_overlay:
        regions.append(Region(mask=np.ones((H, W), bool), technique="flecks", colour=None, order=10 ** 6))
    return Plan(H, W, px_mm, regions)


def plan_fields(H, W, px_mm, rng, palette, n=5, techniques=("scumble", "daubs", "knife_bands")):
    """A few large regions (warped Voronoi), each with its own technique, colour and direction."""
    mm = 1.0 / px_mm
    pts = np.stack([rng.uniform(0, W, n), rng.uniform(0, H, n)], 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cell = max(H, W) / 3
    wx, wy = smooth_field(H, W, rng, cell, 0.08 * max(H, W)), smooth_field(H, W, rng, cell, 0.08 * max(H, W))
    d = np.stack([np.hypot(xx + wx - px_, yy + wy - py_) for px_, py_ in pts], 0)
    lab = d.argmin(0)
    regions = []
    for i in range(n):
        mask = lab == i
        tech = techniques[i % len(techniques)]
        rgb = np.array(palette[int(rng.integers(len(palette)))], np.float32)
        ang = float(rng.choice([0, 0, 90, rng.uniform(-30, 30)]))
        r = Region(mask=mask, technique=tech, colour=tuple(rgb), angle_deg=ang, order=i)
        if tech == "knife_bands":
            ys, = np.nonzero(mask.any(1))
            y0, y1 = ys.min(), ys.max()
            step = rng.uniform(7, 11) * mm
            r.rows = [_boundary(W, px_mm, rng, y) for y in np.arange(y0 + step / 2, y1, step)]
        regions.append(r)
    regions.append(Region(mask=np.ones((H, W), bool), technique="flecks", colour=None, order=10 ** 6))
    return Plan(H, W, px_mm, regions)


def make_ground(plan, rng, fill=None):
    """Colour per region (with low-frequency wander), soft region edges, and a low relief film."""
    H, W, px = plan.H, plan.W, plan.px_mm
    mm = 1.0 / px
    alb = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W), np.float32)
    for r in plan.regions:
        if r.colour is None:
            continue
        m = cv2.GaussianBlur(r.mask.astype(np.float32), (0, 0), 0.4 * mm)       # 0.4 mm soft paint edge
        wander = 1 + smooth_field(H, W, rng, 25 * mm, plan.colour_var)
        alb += m[..., None] * (np.array(r.colour, np.float32)[None, None] * wander[..., None])
        wsum += m
    alb /= np.maximum(wsum, 1e-4)[..., None]
    height = np.clip(plan.ground_relief_mm * (1 + 0.5 * smooth_field(H, W, rng, 3 * mm)), 0, None)
    if fill is not None:                                   # carry the capture's fine low-relief texture
        fa = fill["albedo"].astype(np.float32); fh = fill["height"].astype(np.float32)
        th, tw = fh.shape
        reps = (int(np.ceil(H / th)), int(np.ceil(W / tw)))
        tile = lambda a: np.tile(a, reps + ((1,) if a.ndim == 3 else ()))[:H, :W]
        fa_hp = fa - cv2.GaussianBlur(fa, (0, 0), 2 * mm)
        fh_hp = fh - cv2.GaussianBlur(fh, (0, 0), 2 * mm)
        alb += 0.8 * tile(fa_hp); height += tile(fh_hp)
    gloss = np.full((H, W), 0.35, np.float32)
    return {"albedo": np.clip(alb, 0, 255), "height": height.astype(np.float32), "gloss": gloss}


class _Picker:
    """Least-used-first mark choice with a no-reuse radius."""

    def __init__(self, marks, rng, reuse_px):
        self.marks, self.rng, self.reuse2 = marks, rng, reuse_px ** 2
        self.uses = np.zeros(len(marks), np.int32)
        self.where = {i: [] for i in range(len(marks))}

    def pick(self, pool, x, y, max_perp_px=None):
        pool = [i for i in pool if all((x - a) ** 2 + (y - b) ** 2 > self.reuse2 for a, b in self.where[i])] or pool
        if max_perp_px is not None:
            fit = [i for i in pool if self.marks[i].alpha.shape[0] <= max_perp_px]
            pool = fit or pool
        w = 1.0 / (1.0 + self.uses[pool]) ** 2
        i = int(self.rng.choice(pool, p=w / w.sum()))
        self.uses[i] += 1; self.where[i].append((x, y))
        return i


def place(plan, marks, rng, fill=None, reuse_mm=40.0):
    H, W, px = plan.H, plan.W, plan.px_mm
    mm = 1.0 / px
    ground = make_ground(plan, rng, fill)
    pick = _Picker(marks, rng, reuse_mm * mm)
    by_kind = {}
    for i, m in enumerate(marks):
        by_kind.setdefault(m.kind, []).append(i)
    out = []
    for r in sorted(plan.regions, key=lambda r: (TECHNIQUES[r.technique]["layer"], r.order)):
        t = {**TECHNIQUES[r.technique], **r.params}
        pool = [i for k in t["kinds"] for i in by_kind.get(k, [])]
        if not pool:
            continue
        common = lambda: dict(scale=float(rng.uniform(*t["scale"])), brightness=float(rng.normal(0, t["bright"])),
                              relief_gain=float(rng.uniform(*t["relief"])), own_colour=t["own_colour"])
        flip = lambda: 180.0 if rng.random() < t["flip180"] else 0.0
        if t["layout"] == "rows":
            gap = [float(np.median(np.abs(np.diff(np.stack(r.rows), axis=0)))) if len(r.rows) > 1 else 10 * mm]
            for row in r.rows:
                x = float(rng.uniform(-60 * mm, 0))
                while x < W:
                    xc = int(np.clip(x, 0, W - 1))
                    y = float(row[xc] + rng.normal(0, t["row_jitter_mm"] * mm))
                    i = pick.pick(pool, x, y, max_perp_px=1.6 * gap[0])
                    mw = marks[i].alpha.shape[1]
                    cx = x + mw / 2
                    cxi = int(np.clip(cx, 0, W - 1))
                    # follow the row's local slope so lips run with their band
                    slope = np.degrees(np.arctan2(row[min(W - 1, cxi + 20)] - row[max(0, cxi - 20)], 40))
                    if r.mask[int(np.clip(y, 0, H - 1)), cxi]:
                        out.append(dict(mark=i, x=cx, y=y, angle_deg=r.angle_deg + slope + flip() +
                                        float(rng.normal(0, t["angle_jitter"])), **common()))
                    else:
                        pick.uses[i] -= 1; pick.where[i].pop()
                    x += mw * float(rng.uniform(*t["step"]))
        else:
            area = r.mask.sum()
            mean_a = np.mean([marks[i].alpha.sum() for i in pool])
            n = int(t["coverage"] * area / max(mean_a, 1.0))
            ys, xs = np.nonzero(r.mask)
            for _ in range(n):
                k = int(rng.integers(len(xs)))
                x, y = float(xs[k] + rng.uniform(-.5, .5)), float(ys[k] + rng.uniform(-.5, .5))
                i = pick.pick(pool, x, y)
                out.append(dict(mark=i, x=x, y=y, angle_deg=r.angle_deg + flip() + float(rng.normal(0, t["angle_jitter"])),
                                **common()))
    used = pick.uses[pick.uses > 0]
    report = {"placements": len(out), "distinct_marks": int(len(used)), "library": len(marks),
              "max_share": float(used.max() / max(1, used.sum())) if len(used) else 0.0,
              "regions": {k: sum(1 for r in plan.regions if r.technique == k) for k in TECHNIQUES}}
    return out, ground, report
