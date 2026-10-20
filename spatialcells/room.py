"""A 10 m x 10 m textured room rendered by ray casting, plus rat-like foraging walks.

Coordinates: x points east, y points north, z up, in metres. Heading theta is in
radians, 0 = facing east (+x), pi/2 = facing north (+y). The room is deliberately
asymmetric (different wall textures, posters at different places, two pillars)
so every location has a unique view in every direction, like the cue card in a
rodent open-field arena. Layout "B" is a second room with the same 10 m box but
different walls, floor, posters and pillar positions, for remapping tests. random_layout(seed)
makes procedural rooms (random wall and floor patterns and colours, posters cut from skimage's
bundled pictures, 1-3 pillars at random places) for teaching a model the task (experiment 9).
"""
from __future__ import annotations

import numpy as np
from skimage import data, transform

L = 10.0          # room side length
H = 3.0           # wall height
CAM_H = 1.0       # camera height
HFOV = np.deg2rad(90.0)
IMG = 256         # output resolution (square)
SS = 2            # supersampling factor

# (x, y, radius, height, texture)
PILLARS = [(7.0, 3.0, 0.30, 1.6, "red"), (3.0, 7.0, 0.40, 2.2, "ihc")]
PILLARS_B = [(2.2, 4.6, 0.35, 1.9, "green"), (7.6, 5.8, 0.30, 2.4, "checker")]
LIGHT = np.array([5.0, 5.0, 2.8])
MARGIN = 0.35     # walks keep this far from walls and pillars


def _gray(img):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    return img / 255.0 if img.max() > 1.5 else img


def _rgb(img):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, -1)
    img = img[..., :3]
    return img / 255.0 if img.max() > 1.5 else img


def _load_textures():
    brick, grass, gravel = _gray(data.brick()), _gray(data.grass()), _gray(data.gravel())
    noise = (gravel - gravel.mean()) / (gravel.std() + 1e-6)
    t = {}
    t["brick"] = np.clip(brick[..., None] * np.array([1.25, 0.62, 0.48]) * 1.2, 0, 1)
    t["plaster"] = np.clip(np.array([0.92, 0.88, 0.78]) * (1 + 0.04 * noise[..., None]), 0, 1)
    t["blue"] = np.clip(np.array([0.33, 0.48, 0.74]) * (1 + 0.10 * noise[..., None]), 0, 1)
    # vertical wood planks, 0.25 m wide, with grain from the grass texture
    grain = transform.resize(grass, (512, 64), anti_aliasing=True)
    grain = np.tile(grain, (1, 8))
    plank = np.repeat(np.random.default_rng(0).uniform(0.8, 1.15, 8), 64)[None, :]
    t["wood"] = np.clip(np.array([0.55, 0.37, 0.22]) * (plank * (0.8 + 0.4 * grain))[..., None], 0, 1)
    t["floor"] = np.clip(np.array([0.58, 0.56, 0.52]) * (1 + 0.18 * noise[..., None]), 0, 1)
    t["red"] = np.clip(np.array([0.80, 0.16, 0.12]) * (0.85 + 0.3 * grass[..., None]), 0, 1)
    t["ihc"] = _rgb(data.immunohistochemistry())
    # room B (2 m texture tiles at 128 px/m)
    tiles = np.array([0.30, 0.62, 0.60]) * (1 + 0.05 * noise[:256, :256, None])
    grout = (np.arange(256) % 32 < 2)
    tiles[grout] = tiles[:, grout] = 0.85
    t["tiles"] = np.clip(tiles, 0, 1)
    stripe = (np.arange(256) // 32 % 2).astype(bool)[None, :, None]
    t["stripes"] = np.clip(np.where(stripe, [0.55, 0.15, 0.20], [0.85, 0.80, 0.65]) * (1 + 0.05 * noise[:256, :256, None]), 0, 1)
    t["olive"] = np.clip(np.array([0.50, 0.55, 0.25]) * (1 + 0.10 * noise[..., None]), 0, 1)
    slate = np.array([0.28, 0.30, 0.33]) * (1 + 0.12 * noise[..., None])
    slate[np.arange(len(slate)) % 64 < 3] *= 0.5                      # horizontal panel seams every 0.5 m
    t["slate"] = np.clip(slate, 0, 1)
    t["carpet"] = np.clip(np.array([0.40, 0.22, 0.20]) * (1 + 0.15 * noise[..., None]), 0, 1)
    t["green"] = np.clip(np.array([0.20, 0.60, 0.25]) * (0.85 + 0.3 * grass[..., None]), 0, 1)
    chk = _gray(data.checkerboard())[..., None]
    t["checker"] = np.clip(chk * np.array([0.95, 0.85, 0.20]) + (1 - chk) * 0.08, 0, 1)
    return t


def _tint(img, rgb):
    return _rgb(img) * np.array(rgb)


def _posters():
    # wall -> list of (s0, s1, z0, z1, image); s is x on N/S walls and y on E/W walls
    return {
        "N": [(2.0, 4.0, 1.0, 2.6, _rgb(data.astronaut())), (6.5, 9.0, 1.2, 2.5, _rgb(data.rocket()))],
        "E": [(3.0, 5.5, 1.0, 2.4, _rgb(data.coffee())), (7.5, 9.0, 1.6, 2.6, np.full((8, 8, 3), [0.75, 0.88, 0.97]))],
        "S": [(5.0, 7.5, 1.0, 2.5, _rgb(data.chelsea())), (1.0, 2.0, 0.0, 2.2, np.full((8, 8, 3), [0.30, 0.18, 0.10]))],
        "W": [(6.0, 8.5, 1.1, 2.6, _rgb(data.hubble_deep_field())), (1.5, 3.0, 1.3, 2.4, _rgb(data.colorwheel()))],
    }


def _posters_b():
    return {
        "N": [(1.0, 3.0, 1.1, 2.5, _tint(data.camera(), [1.0, 0.85, 0.65])), (5.5, 7.0, 0.0, 2.2, np.full((8, 8, 3), [0.10, 0.30, 0.15]))],
        "E": [(1.5, 3.5, 1.2, 2.5, _rgb(data.retina())), (6.0, 8.5, 1.0, 2.6, _tint(data.coins(), [0.95, 0.80, 0.45]))],
        "S": [(2.5, 4.0, 1.4, 2.6, np.full((8, 8, 3), [0.98, 0.95, 0.70])), (6.5, 9.0, 1.0, 2.4, _tint(data.moon(), [0.6, 0.75, 1.0]))],
        "W": [(1.0, 2.5, 1.0, 2.4, _rgb(data.logo())), (4.5, 7.0, 1.2, 2.6, _tint(data.page(), [0.9, 0.95, 1.0]))],
    }


WALL_TEX = {"N": "brick", "S": "plaster", "E": "blue", "W": "wood"}
LAYOUTS = {"A": dict(pillars=PILLARS, walls=WALL_TEX, floor="floor", posters=_posters),
           "B": dict(pillars=PILLARS_B, walls={"N": "tiles", "S": "stripes", "E": "olive", "W": "slate"},
                     floor="carpet", posters=_posters_b)}
TEX_M = {"brick": 1.0}   # metres per texture tile (default 2)
# Whether image columns run with +s (True) or -s when the wall is viewed head-on.
WALL_FORWARD = {"N": True, "S": False, "E": False, "W": True}
WALL_NORMAL = {"N": (0, -1, 0), "S": (0, 1, 0), "E": (-1, 0, 0), "W": (1, 0, 0)}


def _tile(tex, u, v, px_per_m):
    """Sample a tiled texture at surface coordinates u (horizontal) and v (vertical) in metres."""
    h, w = tex.shape[:2]
    col = np.floor(u * px_per_m).astype(np.int64) % w
    row = np.floor(-v * px_per_m).astype(np.int64) % h
    return tex[row, col]


# ---------------------------------------------------------------- procedural rooms (experiment 9)
POSTER_POOL = ["astronaut", "camera", "cat", "cell", "chelsea", "clock", "coffee", "coins", "colorwheel", "horse",
               "hubble_deep_field", "immunohistochemistry", "logo", "moon", "page", "retina", "rocket", "text",
               "stereo_motorcycle", "shepp_logan_phantom", "binary_blobs", "microaneurysms", "brick", "grass", "gravel"]
PATTERNS = ["plain", "brick", "planks", "tiles", "stripes", "panels", "checker"]


def _colour(rng):
    import colorsys
    return np.array(colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(0.15, 0.8), rng.uniform(0.35, 0.95)))


def _pattern(kind, rng, base):
    """A 256 x 256 surface texture (2 m per tile) of the given kind in random colours."""
    noise = base["noise"][:256, :256, None]
    c1, c2 = _colour(rng), _colour(rng)
    if kind == "plain":
        t = c1 * (1 + rng.uniform(0.03, 0.15) * noise)
    elif kind == "brick":
        t = transform.resize(base["brick"], (256, 256), anti_aliasing=True)[..., None] * c1 * 1.6
    elif kind == "planks":
        w = int(rng.choice([16, 32, 64]))
        plank = np.repeat(rng.uniform(0.8, 1.15, 256 // w), w)[None, :, None]
        t = c1 * plank * (0.8 + 0.4 * base["grain"][..., None])
    elif kind == "tiles":
        t = c1 * (1 + 0.05 * noise)
        g = np.arange(256) % int(rng.choice([32, 64])) < 2
        t[g], t[:, g] = c2 * 0.5 + 0.4, c2 * 0.5 + 0.4
    elif kind == "stripes":
        s = (np.arange(256) // int(rng.choice([16, 32, 64])) % 2).astype(bool)[None, :, None]
        t = np.where(s, c1, c2) * (1 + 0.05 * noise)
    elif kind == "panels":
        t = c1 * (1 + 0.1 * noise)
        t[np.arange(256) % int(rng.choice([32, 64, 128])) < 3] *= 0.5
    else:                                                     # checker
        k = int(rng.choice([32, 64]))
        s = ((np.arange(256)[:, None] // k + np.arange(256)[None] // k) % 2).astype(bool)[..., None]
        t = np.where(s, c1, c2) * (1 + 0.05 * noise)
    return np.clip(t, 0, 1)


def _poster_image(rng):
    """A random poster: a crop of a bundled skimage picture (grey ones tinted), or a plain panel."""
    if rng.random() < 0.2:
        return np.full((8, 8, 3), _colour(rng))
    img = getattr(data, rng.choice(POSTER_POOL))()
    img = img[0] if isinstance(img, tuple) else img                    # stereo_motorcycle: (left, right, disparity)
    img = _rgb(img) if np.asarray(img).ndim == 3 else _rgb(img) * (0.4 + 0.6 * _colour(rng))
    h, w = img.shape[:2]
    ch, cw = int(h * rng.uniform(0.5, 1)), int(w * rng.uniform(0.5, 1))
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    img = img[y0:y0 + ch, x0:x0 + cw]
    return img[:, ::-1] if rng.random() < 0.5 else img


def random_layout(seed: int) -> dict:
    """A procedural room: random wall, floor and pillar textures and colours, 1-3 posters per wall
    (sometimes a door-like panel) and 1-3 pillars at random places. Room(random_layout(seed))."""
    rng = np.random.default_rng(10_000 + seed)
    gravel, grass = _gray(data.gravel()), _gray(data.grass())
    base = {"noise": (gravel - gravel.mean()) / (gravel.std() + 1e-6), "brick": _gray(data.brick()),
            "grain": transform.resize(grass, (256, 256), anti_aliasing=True)}
    kinds = rng.choice(PATTERNS, 4, replace=len(PATTERNS) < 4)
    tex = {f"wall{w}": _pattern(k, rng, base) for w, k in zip("NSEW", kinds)}
    tex["floor_p"] = _pattern(rng.choice(["plain", "tiles", "planks", "checker"]), rng, base)
    pillars = []
    for k in range(rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25])):
        for _ in range(100):
            x, y, r = rng.uniform(1.8, 8.2), rng.uniform(1.8, 8.2), rng.uniform(0.25, 0.45)
            if all(np.hypot(x - px, y - py) > r + pr + 1.2 for px, py, pr, _, _ in pillars):
                break
        tex[f"pillar{k}"] = (_pattern(rng.choice(PATTERNS), rng, base) if rng.random() < 0.7
                             else transform.resize(_poster_image(rng), (256, 256, 3), anti_aliasing=True))
        pillars.append((x, y, r, rng.uniform(1.3, 2.7), f"pillar{k}"))
    posters = {}
    for w in "NSEW":
        posters[w], s = [], rng.uniform(0.3, 1.5)
        for _ in range(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])):
            door = rng.random() < 0.12
            width = rng.uniform(0.9, 1.2) if door else rng.uniform(0.8, 2.5)
            if s + width > L - 0.3:
                break
            z0 = 0.0 if door else rng.uniform(0.8, 1.5)
            z1 = rng.uniform(2.0, 2.3) if door else min(z0 + rng.uniform(0.9, 1.6), 2.8)
            posters[w].append((s, s + width, z0, z1, np.full((8, 8, 3), _colour(rng)) if door else _poster_image(rng)))
            s += width + rng.uniform(0.3, 2.5)
    return dict(pillars=pillars, walls={w: f"wall{w}" for w in "NSEW"}, floor="floor_p", posters=lambda: posters, tex=tex)


class Room:
    def __init__(self, layout="A"):
        """layout: a name in LAYOUTS, or a dict like random_layout()'s."""
        self.layout = layout if isinstance(layout, str) else "procedural"
        lay = LAYOUTS[layout] if isinstance(layout, str) else layout
        self.tex = _load_textures()
        self.tex.update(lay.get("tex", {}))
        self.posters = lay["posters"]()
        self.pillars, self.walls, self.floor = lay["pillars"], lay["walls"], lay["floor"]
        n = IMG * SS
        a = np.tan(HFOV / 2)
        lin = (np.arange(n) + 0.5) / n * 2 - 1
        self.u, self.v = np.meshgrid(lin * a, -lin * a)   # u: left->right, v: top->bottom

    def render(self, x: float, y: float, theta: float) -> np.ndarray:
        """First-person RGB view (IMG x IMG x 3, uint8) from (x, y) facing theta."""
        f = np.array([np.cos(theta), np.sin(theta), 0.0])
        r = np.array([np.sin(theta), -np.cos(theta), 0.0])
        d = f[None, None] + self.u[..., None] * r + self.v[..., None] * np.array([0, 0, 1.0])
        d = d.reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        dx, dy, dz = d.T
        cam = np.array([x, y, CAM_H])
        n_rays = len(d)
        t = np.full(n_rays, np.inf)
        sid = np.zeros(n_rays, dtype=np.int8)
        with np.errstate(divide="ignore", invalid="ignore"):
            # box planes: 1 floor, 2 ceiling, 3 W, 4 E, 5 S, 6 N
            cands = [
                (1, np.where(dz < 0, -CAM_H / dz, np.inf)),
                (2, np.where(dz > 0, (H - CAM_H) / dz, np.inf)),
                (3, np.where(dx < 0, -x / dx, np.inf)),
                (4, np.where(dx > 0, (L - x) / dx, np.inf)),
                (5, np.where(dy < 0, -y / dy, np.inf)),
                (6, np.where(dy > 0, (L - y) / dy, np.inf)),
            ]
            for k, (px, py, pr, ph, _) in enumerate(self.pillars):
                ox, oy = x - px, y - py
                a2 = dx * dx + dy * dy
                b = 2 * (dx * ox + dy * oy)
                c = ox * ox + oy * oy - pr * pr
                disc = b * b - 4 * a2 * c
                tp = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a2)
                zp = CAM_H + tp * dz
                ok = (disc > 0) & (tp > 0) & (zp >= 0) & (zp <= ph)
                cands.append((7 + k, np.where(ok, tp, np.inf)))
        for k, tk in cands:
            better = tk < t
            t[better] = tk[better]
            sid[better] = k
        hit = cam[None] + t[:, None] * d
        col = np.zeros((n_rays, 3), dtype=np.float32)
        normal = np.zeros((n_rays, 3), dtype=np.float32)

        m = sid == 1
        if m.any():
            hx, hy = hit[m, 0], hit[m, 1]
            c = _tile(self.tex[self.floor], hx, hy, 256 / 2.0)
            if self.floor == "floor":
                grout = (np.abs(hx - np.round(hx)) < 0.015) | (np.abs(hy - np.round(hy)) < 0.015)
                c[grout] *= 0.55
            col[m], normal[m] = c, (0, 0, 1)
        m = sid == 2
        if m.any():
            hx, hy = hit[m, 0], hit[m, 1]
            c = np.tile(np.array([0.90, 0.90, 0.92], dtype=np.float32), (m.sum(), 1))
            seam = (np.abs(hx - np.round(hx)) < 0.02) | (np.abs(hy - np.round(hy)) < 0.02)
            c[seam] *= 0.78
            col[m], normal[m] = c, (0, 0, -1)
        for k, wall in [(3, "W"), (4, "E"), (5, "S"), (6, "N")]:
            m = sid == k
            if not m.any():
                continue
            s = hit[m, 1] if wall in "EW" else hit[m, 0]
            z = hit[m, 2]
            tex = self.walls[wall]
            c = _tile(self.tex[tex], s, z, 256 / TEX_M.get(tex, 2.0))
            skirt = z < 0.12
            c[skirt] = (0.25, 0.24, 0.23)
            for s0, s1, z0, z1, img in self.posters[wall]:
                pm = (s >= s0) & (s <= s1) & (z >= z0) & (z <= z1)
                if not pm.any():
                    continue
                ih, iw = img.shape[:2]
                frac = (s[pm] - s0) / (s1 - s0)
                if not WALL_FORWARD[wall]:
                    frac = 1 - frac
                ci = np.clip((frac * iw).astype(int), 0, iw - 1)
                ri = np.clip(((z1 - z[pm]) / (z1 - z0) * ih).astype(int), 0, ih - 1)
                pc = img[ri, ci]
                frame = (np.minimum(s[pm] - s0, s1 - s[pm]) < 0.03) | (np.minimum(z[pm] - z0, z1 - z[pm]) < 0.03)
                pc[frame] = 0.08
                c[pm] = pc
            col[m], normal[m] = c, WALL_NORMAL[wall]
        for k, (px, py, pr, ph, tex) in enumerate(self.pillars):
            m = sid == 7 + k
            if not m.any():
                continue
            ang = np.arctan2(hit[m, 1] - py, hit[m, 0] - px)
            c = _tile(self.tex[tex], ang * pr, hit[m, 2], 256 / 1.0)
            col[m] = c
            nrm = np.stack([hit[m, 0] - px, hit[m, 1] - py, np.zeros(m.sum())], 1) / pr
            normal[m] = nrm

        to_light = LIGHT[None] - hit
        dist = np.linalg.norm(to_light, axis=1)
        lam = np.clip((normal * to_light).sum(1) / dist, 0, 1)
        shade = 0.42 + 0.75 * lam / (1 + (dist / 6.0) ** 2)
        shade[sid == 2] = 0.85  # the light hangs just below the ceiling; keep it bright
        fog = np.exp(-t / 35.0)[:, None]
        rgb = col * shade[:, None] * fog + 0.75 * (1 - fog)
        n = IMG * SS
        rgb = rgb.reshape(IMG, SS, IMG, SS, 3).mean((1, 3))
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def valid(x: float, y: float, margin: float = MARGIN, pillars=PILLARS) -> bool:
    if not (margin <= x <= L - margin and margin <= y <= L - margin):
        return False
    return all(np.hypot(x - px, y - py) >= pr + margin for px, py, pr, _, _ in pillars)


def foraging_walk(n_steps: int, rng: np.random.Generator, step: float = 0.22,
                  turn_sd: float = np.deg2rad(15), p_scan: float = 0.15, pillars=PILLARS):
    """Rat-like foraging: mostly forward motion with drifting heading, occasional
    on-the-spot head scans, and turning away from walls and pillars.
    Returns an (n_steps, 3) array of (x, y, theta)."""
    while True:
        x, y = rng.uniform(MARGIN, L - MARGIN, 2)
        if valid(x, y, pillars=pillars):
            break
    th = rng.uniform(0, 2 * np.pi)
    out = np.zeros((n_steps, 3))
    for i in range(n_steps):
        out[i] = x, y, th % (2 * np.pi)
        if rng.random() < p_scan:
            th += rng.choice([-1, 1]) * rng.uniform(np.deg2rad(30), np.deg2rad(90))
            continue
        th += rng.normal(0, turn_sd)
        for _ in range(20):
            s = step * rng.uniform(0.7, 1.3)
            nx, ny = x + s * np.cos(th), y + s * np.sin(th)
            if valid(nx, ny, pillars=pillars):
                x, y = nx, ny
                break
            th += rng.choice([-1, 1]) * rng.uniform(np.deg2rad(60), np.deg2rad(180))
    return out
