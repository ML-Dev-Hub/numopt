"""README hero: four optimizers on Himmelblau's function, computed with numopt.

    .venv/bin/python docs/brand/scripts/make_hero.py            # SVGs (+ PNG fallbacks in build/)
    .venv/bin/python docs/brand/scripts/make_hero.py --no-raster

Every drawn iterate is a real `numopt.run(...)` trace from x0 = (-3.75, 2.5), the start of the
portal's home-page descent scene (web/src/app/home/previews/labs/unconstrained.ts), chosen so the
four routes to the minimizer (-2.805, 3.131) separate and stay inside the view. All four runs use
ONE stopping test, the first k with ||grad f(x_k)||_2 <= 1e-8; every other parameter is the
registered default. (Each method tests its own norm internally, so it runs with its internal
tolerance at the floor and its trace is cut at the first iterate that passes the common test. The
methods are deterministic, so the cut trace is exactly the run with that test.)

The figure is wide and short: the level-set landscape with the iterates at equal scale on both
axes, and beside it ||grad f(x_k)||_2 against k on log-log axes. The y axis of the convergence
panel is the quantity the stopping test reads, so every curve ends at a dot on (or just under) the
dashed stopping line instead of plunging off the chart. Diamonds mark heavy-ball momentum's x_k at
k = 10 and 100 on both panels. The figure has no background of its own: it sits on the page
(GitHub's #ffffff or #0d1117) and only the landscape panel carries a fill.

Outputs (docs/brand/readme-hero/), each in light and dark:
    hero-{theme}.svg            wide figure (960 units), README on desktop, docs, portal concept
    hero-stacked-{theme}.svg    narrow figure (540 units), panels stacked, README on phones
PNG fallbacks go to docs/brand/build/readme-hero/ (git-ignored).

This module also keeps the Rosenbrock scene that make_social.py (the social card) and
make_sheet.py / palette_check.py (the brand sheet and the palette gates) read: PROBLEM, X0,
DOMAIN, METHODS, run_methods, landscape, drawn_path, path_layers, THEMES and colormap.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import contourpy
import numpy as np
import typeset as ts
from colorlib import blend, rgb_to_hex

import numopt
from numopt import problems

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "readme-hero"
BUILD = HERE.parent / "build" / "readme-hero"
REPO = HERE.parents[2]

# ── Themes (mirror web/src/ui/tokens.css; brand.md documents every value) ───────────
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        border=blend("#141413", "#fcfcfb", 0.10),
        text="#141413",
        text2="#52514e",
        text3="#6b6963",
        grid=blend("#141413", "#fcfcfb", 0.07),
        axis=blend("#141413", "#fcfcfb", 0.22),
        halo="#fcfcfb",
        iso="rgba(30,40,60,0.15)",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#882892"],
        playhead=blend("#141413", "#fcfcfb", 0.45),
    ),
    "dark": dict(
        surface="#141413",
        border=blend("#fffffa", "#141413", 0.10),
        text="#f1f0eb",
        text2="#bab9b0",
        text3="#8a8982",
        grid=blend("#fffffa", "#141413", 0.06),
        axis=blend("#fffffa", "#141413", 0.2),
        halo="#141413",
        iso="rgba(220,230,255,0.10)",
        series=["#3987e5", "#d95926", "#199e70", "#a13bab"],
        playhead=blend("#fffffa", "#141413", 0.5),
    ),
}

# The hero has no card behind it, so hairlines are translucent: they read the same on the
# README's #ffffff / #0d1117 page as on the brand's own surfaces.
HAIRLINES = {
    "light": dict(
        grid="rgba(20,20,19,0.08)", axis="rgba(20,20,19,0.28)", border="rgba(20,20,19,0.12)"
    ),
    "dark": dict(
        grid="rgba(255,255,250,0.07)",
        axis="rgba(255,255,250,0.25)",
        border="rgba(255,255,250,0.12)",
    ),
}


@dataclass
class MethodCfg:
    id: str
    label: str  # legend: textbook name, variant in parentheses (brand.md, "Voice")
    short: str  # social card chips
    slot: int  # 1..4 = --series-1..4
    params: dict = field(default_factory=dict)
    dots: bool = False  # mark every iterate (short traces only)
    width: float = 2.0  # stroke width in the hero


def series(th: dict, cfg: MethodCfg) -> str:
    return th["series"][cfg.slot - 1]


# Contour colormaps: the same OKLCH anchors as web/src/ui/colors.ts (CONTOUR_MAPS), so the hero
# and the lab paint one landscape with one color. t = 0 is the lowest f.
CONTOUR_ANCHORS = {
    "light": [
        (0, 0.8, 0.032, 212),
        (0.3, 0.868, 0.034, 196),
        (0.62, 0.93, 0.03, 150),
        (1, 0.982, 0.03, 88),
    ],
    "dark": [
        (0, 0.42, 0.034, 212),
        (0.32, 0.33, 0.03, 200),
        (0.66, 0.255, 0.022, 170),
        (1, 0.195, 0.012, 95),
    ],
}


def _oklab_to_hex(L: float, a: float, b: float) -> str:
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def gamma(c: float) -> float:
        c = min(1.0, max(0.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return rgb_to_hex((gamma(r), gamma(g), gamma(bb)))


def colormap(theme: str, t: float) -> str:
    """Port of interpAnchors() in web/src/ui/colors.ts (interpolates in OKLab a, b)."""
    anchors = CONTOUR_ANCHORS[theme]
    x = min(1.0, max(0.0, t))
    i = 0
    while i < len(anchors) - 2 and x > anchors[i + 1][0]:
        i += 1
    t0, L0, C0, h0 = anchors[i]
    t1, L1, C1, h1 = anchors[i + 1]
    u = 0.0 if t1 == t0 else (x - t0) / (t1 - t0)
    a0, b0 = C0 * math.cos(math.radians(h0)), C0 * math.sin(math.radians(h0))
    a1, b1 = C1 * math.cos(math.radians(h1)), C1 * math.sin(math.radians(h1))
    return _oklab_to_hex(L0 + (L1 - L0) * u, a0 + (a1 - a0) * u, b0 + (b1 - b0) * u)


# ── Typography and geometry helpers ───────────────────────────────────────────────────


def num(v: float, digits: int = 2) -> str:
    """Fixed-point with U+2212 for the minus sign."""
    return f"{v:.{digits}f}".replace("-", "−")


def count(n: int) -> str:
    return f"{n:,}"


def tex(s: str, size: float) -> list[ts.Run]:
    """ts.math, with \\| set as U+2225 (KaTeX_Main has no U+2016)."""
    runs: list[ts.Run] = []
    for i, part in enumerate(s.split("\\|")):
        if i:
            runs.append(("∥", "math-rm", size, 0.0))
        if part:
            runs += ts.math(part, size)
    return runs


def fill_path(runs, x, y, color, anchor="start", halo=None, tnum=False) -> tuple[str, float]:
    d, w = ts.path_d(runs, x, y, anchor=anchor, tnum=tnum)
    if halo:
        return (
            f'<path d="{d}" fill="{color}" stroke="{halo}" stroke-width="3.2" '
            'stroke-linejoin="round" paint-order="stroke"/>'
        ), w
    return f'<path d="{d}" fill="{color}"/>', w


def rdp(P: np.ndarray, eps: float) -> np.ndarray:
    """Indices kept by Ramer–Douglas–Peucker (iterative)."""
    n = len(P)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = float(np.hypot(*ab))
        seg = P[i + 1 : j]
        if L < 1e-12:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return np.flatnonzero(keep)


def d_poly(P: np.ndarray, closed: bool = False) -> str:
    s = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in P)
    return s + ("Z" if closed else "")


def box_px(xy, box, domain) -> np.ndarray:
    """Data coordinates -> pixels in `box` (x, y, w, h) showing `domain` ((x0, x1), (y0, y1))."""
    (x0, x1), (y0, y1) = domain
    lx, ly, lw, lh = box
    xy = np.asarray(xy, dtype=float)
    px = lx + (xy[..., 0] - x0) / (x1 - x0) * lw
    py = ly + lh - (xy[..., 1] - y0) / (y1 - y0) * lh
    return np.stack([px, py], axis=-1)


def diamond(x: float, y: float, r: float, fill: str, halo: str, extra: str = "") -> str:
    d = f"M{x:.1f} {y - r:.1f}L{x + r:.1f} {y:.1f}L{x:.1f} {y + r:.1f}L{x - r:.1f} {y:.1f}Z"
    return f'<path d="{d}" fill="{fill}" stroke="{halo}" stroke-width="1.4"{extra}/>'


def dot(x: float, y: float, color: str, halo: str, r: float = 3.1) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{halo}" stroke-width="1.3"/>'


def stroke_with_halo(d: str, color: str, width: float, halo: str) -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{halo}" stroke-opacity="0.9" stroke-width="{width + 3:.1f}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def x0_ring(x: float, y: float, th: dict, r: float = 6.5) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{th["halo"]}" stroke-width="4.5"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{th["text"]}" stroke-width="1.8"/>'
    )


def star_cross(x: float, y: float, th: dict, s: float = 7) -> str:
    d = f"M{x - s:.1f} {y:.1f}H{x + s:.1f}M{x:.1f} {y - s:.1f}V{y + s:.1f}"
    return (
        f'<path d="{d}" stroke="{th["halo"]}" stroke-width="5" stroke-linecap="round"/>'
        f'<path d="{d}" stroke="{th["text"]}" stroke-width="1.8" stroke-linecap="round"/>'
    )


# ── Landscape ─────────────────────────────────────────────────────────────────────────


def landscape(p, theme: str, n_levels: int = 15, box=None, domain=None) -> list[str]:
    """Filled sublevel sets {f <= c_i}, painted from the highest level down, each with its iso-line.

    Levels are uniform in t = log(f - f_min + delta) over the plotted domain (the web's 'log'
    level scale, contourField.ts), so the steep walls and the flat valley both get bands. `box`
    and `domain` default to the module's LEFT and DOMAIN (make_social.py sets those).
    """
    box = LEFT if box is None else box
    domain = DOMAIN if domain is None else domain
    (x0, x1), (y0, y1) = domain
    nx, ny = 480, 450
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: float(p.f(np.array([a, b]))))(X, Y)
    fmin, fmax = 0.0, float(Z.max())
    delta = (fmax - fmin) * 2e-4
    T = np.log(Z - fmin + delta)
    t_lo, t_hi = math.log(delta), math.log(fmax - fmin + delta)
    levels = [t_lo + (t_hi - t_lo) * (i / n_levels) for i in range(1, n_levels)]
    gen = contourpy.contour_generator(xs, ys, T, fill_type=contourpy.FillType.OuterCode)
    out = []
    lx, ly, lw, lh = box
    out.append(
        f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="{colormap(theme, 1.0)}"/>'
    )
    for i in reversed(range(len(levels))):
        polys, _codes = gen.filled(-1e9, levels[i])
        color = colormap(theme, (i + 0.5) / n_levels)
        parts = []
        for poly in polys:
            P = box_px(poly, box, domain)
            keep = rdp(P, 0.35)
            if len(keep) < 3:
                continue
            parts.append(d_poly(P[keep], closed=True))
        if not parts:
            continue
        out.append(
            f'<path d="{" ".join(parts)}" fill="{color}" fill-rule="evenodd" '
            f'stroke="{THEMES[theme]["iso"]}" stroke-width="0.8"/>'
        )
    return out


# ══ Rosenbrock scene: the social card (make_social.py) and the brand sheet read these ══

PROBLEM = "rosenbrock"
X0 = (-1.2, 1.0)
DOMAIN = ((-2.0, 2.0), (-1.0, 2.75))  # make_social.py re-targets DOMAIN and LEFT
LEFT = (58, 76, 500, 468.75)
GTOL = 1e-8  # the common stopping test: ||grad f(x_k)||_2 <= GTOL
BUDGET = 20_000  # the common iteration budget
MILESTONES = (10, 100, 1_000, 10_000)
K_MAX = 20_000

METHODS = [
    MethodCfg("gradient_descent", "Gradient descent (Armijo backtracking)", "Gradient descent", 1),
    MethodCfg("momentum", "Heavy-ball momentum", "Momentum", 2),
    MethodCfg("bfgs", "BFGS", "BFGS", 3, dots=True),
    MethodCfg("pure_newton", "Newton", "Newton", 4, dots=True),
]

# Paint order: the long, smooth crawlers share the valley floor, so the slowest is painted last
# with the thinnest stroke; the iterate chains (BFGS, Newton) sit underneath.
PAINT_ORDER = ["momentum", "bfgs", "pure_newton", "gradient_descent"]
STROKE = {"momentum": 2.2, "bfgs": 2.0, "pure_newton": 2.0, "gradient_descent": 1.5}


@dataclass
class Run:
    cfg: MethodCfg
    xs: np.ndarray  # (n+1, 2) iterates
    fs: np.ndarray  # (n+1,) f(x_k)
    converged: bool  # the common test passed within the budget
    n_iter: int


def run_methods() -> tuple[object, list[Run]]:
    p = problems.get(PROBLEM)
    runs = []
    for cfg in METHODS:
        spec = numopt.get_method(cfg.id)
        bounds = {ps.name: ps for ps in spec.params}
        # Internal tolerance at its floor, so the method never stops before the common test.
        params = {"gtol": bounds["gtol"].min, "max_iter": min(BUDGET, int(bounds["max_iter"].max))}
        params.update(cfg.params)
        r = numopt.run(cfg.id, p, x0=list(X0), **params)
        xs = np.array([s.x for s in r.trace], dtype=float)
        fs = np.array([s.fun for s in r.trace], dtype=float)
        g2 = np.array([np.linalg.norm(p.grad(x)) for x in xs])
        hit = np.flatnonzero(g2 <= GTOL)
        if len(hit):
            k = int(hit[0])
            runs.append(Run(cfg, xs[: k + 1], fs[: k + 1], True, k))
        else:
            runs.append(Run(cfg, xs, fs, False, len(xs) - 1))
    return p, runs


def to_px(xy: np.ndarray) -> np.ndarray:
    return box_px(xy, LEFT, DOMAIN)


def inside(P: np.ndarray, pad: float = 0.0) -> np.ndarray:
    lx, ly, lw, lh = LEFT
    return (
        (P[:, 0] >= lx - pad)
        & (P[:, 0] <= lx + lw + pad)
        & (P[:, 1] >= ly - pad)
        & (P[:, 1] <= ly + lh + pad)
    )


@dataclass
class DrawnPath:
    run: Run
    d: str
    kept: np.ndarray
    cum: np.ndarray  # arc length (px) at every iterate, along the drawn polyline
    length: float


def drawn_path(run: Run) -> DrawnPath:
    P = to_px(run.xs)
    keep = rdp(P, 0.3) if len(P) > 60 else np.arange(len(P))
    Q = P[keep]
    seg = np.hypot(*np.diff(Q, axis=0).T)
    cum_kept = np.concatenate([[0.0], np.cumsum(seg)])
    cum = np.interp(np.arange(len(P)), keep, cum_kept)
    return DrawnPath(run, d_poly(Q), keep, cum, float(cum_kept[-1]))


def path_layers(
    dp_list: list[DrawnPath], theme: str, *, animated: bool = False, kmax: float = K_MAX
) -> list[str]:
    """Static trajectories of the Rosenbrock scene (the social card). `animated` and `kmax` are
    accepted for make_social.py's call and must stay False / unused: the hero no longer animates."""
    assert not animated, "the Rosenbrock scene is drawn static only"
    th = THEMES[theme]
    out: list[str] = []
    for dp in dp_list:
        run = dp.run
        color = series(th, run.cfg)
        width = STROKE.get(run.cfg.id, 1.8)
        P = to_px(run.xs)
        ok = inside(P)
        if run.cfg.dots:
            # Iterate chains: one segment per step; segments with an end outside are dashed.
            for i in range(len(P) - 1):
                a, b = P[i], P[i + 1]
                dash = not (ok[i] and ok[i + 1])
                seg = f"M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"
                out.append(
                    f'<path d="{seg}" fill="none" stroke="{th["halo"]}" stroke-opacity="0.85" stroke-width="{width + 3:.1f}" stroke-linecap="round"/>'
                    f'<path d="{seg}" fill="none" stroke="{color}" stroke-width="{width - (0.3 if dash else 0):.1f}" stroke-linecap="round"'
                    + (' stroke-dasharray="5 5"' if dash else "")
                    + "/>"
                )
            for k in range(1, len(P)):
                if not ok[k]:
                    continue
                x, y = P[k]
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{color}" stroke="{th["halo"]}" stroke-width="1.3"/>'
                )
        else:
            out.append(
                f'<path d="{dp.d}" fill="none" stroke="{th["halo"]}" stroke-opacity="0.8" stroke-width="{width + 3:.1f}" '
                'stroke-linecap="round" stroke-linejoin="round"></path>'
            )
            out.append(
                f'<path d="{dp.d}" fill="none" stroke="{color}" stroke-width="{width}" '
                'stroke-linecap="round" stroke-linejoin="round"></path>'
            )
    # Milestones on top of every path: x_k at k = 10, 100, 1000, 10000.
    for dp in dp_list:
        run = dp.run
        color = series(th, run.cfg)
        for k in MILESTONES:
            if k >= run.n_iter:
                continue
            x, y = to_px(run.xs[k])
            if not inside(np.array([[x, y]]))[0]:
                continue
            out.append(diamond(x, y, 4.6, color, th["halo"]))
    return out


DATA: tuple[object, list[Run]]


# ══ README hero: Himmelblau ═══════════════════════════════════════════════════════════

HERO_PROBLEM = "himmelblau"
HERO_X0 = (-3.75, 2.5)
HERO_GTOL = 1e-8  # the one stopping test: the first k with ||grad f(x_k)||_2 <= HERO_GTOL
HERO_MILESTONES = (10, 100)  # diamonds, for runs longer than HERO_DIAMOND_MIN iterations
HERO_DIAMOND_MIN = 30

HERO_METHODS = [
    MethodCfg(
        "gradient_descent",
        "Gradient descent (Armijo backtracking)",
        "Gradient descent",
        1,
        dots=True,
        width=1.8,
    ),
    MethodCfg("momentum", "Heavy-ball momentum", "Momentum", 2, width=1.6),
    MethodCfg("bfgs", "BFGS", "BFGS", 3, dots=True),
    MethodCfg("damped_newton", "Damped Newton", "Newton", 4, dots=True),
]
# Gradient descent, heavy ball and BFGS all take their first step along -grad f(x0), so their
# first segments lie on one ray. Gradient descent's step is the shortest, so it is painted above
# BFGS's; heavy ball's spiral goes first, underneath everything.
HERO_PAINT_ORDER = ["momentum", "bfgs", "gradient_descent", "damped_newton"]


@dataclass
class HeroRun:
    cfg: MethodCfg
    xs: np.ndarray  # (n+1, 2) iterates x_0 .. x_n
    g: np.ndarray  # (n+1,) ||grad f(x_k)||_2
    n: int  # iterations: the first k that passes the stopping test


def run_hero() -> tuple[object, list[HeroRun], np.ndarray]:
    p = problems.get(HERO_PROBLEM)
    runs = []
    for cfg in HERO_METHODS:
        spec = numopt.get_method(cfg.id)
        bounds = {ps.name: ps for ps in spec.params}
        # Internal tolerance at its floor, so the method never stops before the common test.
        r = numopt.run(cfg.id, p, x0=list(HERO_X0), gtol=bounds["gtol"].min, **cfg.params)
        xs = np.array([s.x for s in r.trace], dtype=float)
        g = np.array([np.linalg.norm(p.grad(x)) for x in xs])
        hit = np.flatnonzero(g <= HERO_GTOL)
        assert len(hit), f"{cfg.id} did not pass the stopping test within its default budget"
        k = int(hit[0])
        runs.append(HeroRun(cfg, xs[: k + 1], g[: k + 1], k))
    # The minimizer every run reaches (one of Himmelblau's four), from the problem's metadata.
    minima = np.array(p.minima, dtype=float)
    star = minima[np.argmin(np.linalg.norm(minima - runs[0].xs[-1], axis=1))]
    for r in runs:
        assert np.linalg.norm(r.xs[-1] - star) < 1e-6, f"{r.cfg.id} ended away from x*"
    return p, runs, star


# ── Layouts ───────────────────────────────────────────────────────────────────────────
# GitHub's README column is about 830 px on a repository page and up to 946 px without the
# sidebar. The wide figure is laid out at 960 units, so 1 unit is 0.86-0.99 px there and the
# smallest type (13.5 units) renders at 11.7 px or more. On a phone (<= 640 px viewport) the
# README shows the stacked figure, laid out at 540 units with 17-unit minimum type: 11.3 px in
# the 358 px column of a 390 px phone.


@dataclass
class HeroLayout:
    name: str
    W: int
    domain: tuple[tuple[float, float], tuple[float, float]]
    field_xy: tuple[float, float]  # top-left of the landscape panel
    field_w: float  # its height follows from the domain (equal scale)
    conv: tuple[float, float, float, float] | None  # None: below the landscape
    stacked: bool
    fs: dict


HERO_WIDE = HeroLayout(
    name="wide",
    W=960,
    domain=((-4.45, -1.45), (2.3, 3.8)),
    field_xy=(40, 40),
    field_w=560,
    conv=(688, 40, 250, 276),
    stacked=False,
    fs=dict(title=16, formula=17, tick=13.5, axis=14.5, legend=14, note=13.5, label=16, rate=15),
)

HERO_STACKED = HeroLayout(
    name="stacked",
    W=540,
    domain=((-4.3, -1.6), (2.3, 3.8)),
    field_xy=(40, 74),
    field_w=492,
    conv=None,
    stacked=True,
    fs=dict(title=19, formula=19, tick=17, axis=18, legend=18, note=17, label=19, rate=18),
)

CONV_KMAX = 1_000  # right end of the log-k axis
CONV_DEC = (-10, 2)  # log10 range of ||grad f(x_k)||_2


def hero_tokens(theme: str) -> dict:
    return {**THEMES[theme], **HAIRLINES[theme]}


def hero_field(p, theme: str, box, domain, tag: str, runs: list[HeroRun]) -> list[str]:
    th = hero_tokens(theme)
    lx, ly, lw, lh = box
    out = [
        f'<clipPath id="clip-{tag}"><rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="10"/></clipPath>',
        f'<g clip-path="url(#clip-{tag})">',
    ]
    out += landscape(p, theme, n_levels=16, box=box, domain=domain)
    by_id = {r.cfg.id: r for r in runs}
    for rid in HERO_PAINT_ORDER:
        r = by_id[rid]
        c = series(th, r.cfg)
        P = box_px(r.xs, box, domain)
        Q = P[rdp(P, 0.3)] if len(P) > 60 else P
        out.append(stroke_with_halo(d_poly(Q), c, r.cfg.width, th["halo"]))
        if r.cfg.dots:
            out += [dot(x, y, c, th["halo"]) for x, y in P[1:]]
    for r in runs:
        if r.n <= HERO_DIAMOND_MIN:
            continue
        c = series(th, r.cfg)
        for k in HERO_MILESTONES:
            if k < r.n:
                out.append(diamond(*box_px(r.xs[k], box, domain), 4.4, c, th["halo"]))
    out.append("</g>")
    out.append(
        f'<rect x="{lx + 0.5}" y="{ly + 0.5}" width="{lw - 1}" height="{lh - 1}" rx="10" fill="none" stroke="{th["border"]}"/>'
    )
    return out


def hero_conv(runs: list[HeroRun], theme: str, box, fs: dict) -> list[str]:
    """||grad f(x_k)||_2 against k >= 1, log-log. Each curve ends at its first iterate under the
    dashed stopping line, so the chart's floor is the test itself."""
    th = hero_tokens(theme)
    rx, ry, rw, rh = box
    out = []

    def cx(k):
        return rx + np.log10(k) / math.log10(CONV_KMAX) * rw

    def cy(v):
        lv = np.log10(np.asarray(v, dtype=float))
        return ry + rh - (lv - CONV_DEC[0]) / (CONV_DEC[1] - CONV_DEC[0]) * rh

    for e in range(CONV_DEC[0], CONV_DEC[1] + 1, 4):
        y = float(cy(10.0**e))
        out.append(f'<path d="M{rx} {y:.1f}H{rx + rw}" stroke="{th["grid"]}" stroke-width="1"/>')
        s, _ = fill_path(
            ts.math("1" if e == 0 else f"10^{{{e}}}", fs["tick"]),
            rx - 8,
            y + fs["tick"] * 0.36,
            th["text3"],
            "end",
        )
        out.append(s)
    for e in range(0, round(math.log10(CONV_KMAX)) + 1):
        x = float(cx(10.0**e))
        out.append(f'<path d="M{x:.1f} {ry}V{ry + rh}" stroke="{th["grid"]}" stroke-width="1"/>')
        lab = {0: "1", 1: "10"}.get(e, f"10^{e}")
        s, _ = fill_path(
            ts.math(lab, fs["tick"]), x, ry + rh + fs["tick"] + 8, th["text3"], "middle"
        )
        out.append(s)
    out.append(
        f'<path d="M{rx} {ry + rh}H{rx + rw}M{rx} {ry}V{ry + rh}" stroke="{th["axis"]}" stroke-width="1"/>'
    )
    # The stopping test, labeled above the line where no curve runs (k = 20..200).
    yt = float(cy(HERO_GTOL))
    out.append(
        f'<path d="M{rx} {yt:.1f}H{rx + rw}" stroke="{th["text3"]}" stroke-width="1.1" stroke-dasharray="4 3"/>'
    )
    s, _ = fill_path(
        ts.text("stopping test", fs["note"]), float(cx(60)), yt - 6, th["text3"], "middle"
    )
    out.append(s)
    # Axis name (right) and the milestone key (left) share the row under the ticks.
    base = ry + rh + 2 * fs["tick"] + 14
    s, _ = fill_path(
        ts.text("iteration ", fs["axis"] - 1) + ts.math("k", fs["axis"] + 1),
        rx + rw,
        base,
        th["text3"],
        "end",
    )
    out.append(s)
    out.append(diamond(rx + 5, base - fs["note"] * 0.36, 4.4, th["text2"], th["halo"]))
    s, _ = fill_path(
        ts.math("\\mathbf{x}_k", fs["note"] + 1)
        + ts.text(" at ", fs["note"])
        + ts.math("k = 10, 10^2", fs["note"] + 1),
        rx + 15,
        base,
        th["text3"],
    )
    out.append(s)
    for r in sorted(runs, key=lambda q: -q.n):
        c = series(th, r.cfg)
        k = np.arange(1, r.n + 1, dtype=float)
        Q = np.stack([cx(k), cy(r.g[1:])], axis=-1)
        assert Q[:, 1].min() >= ry and Q[:, 1].max() <= ry + rh, f"{r.cfg.id} leaves the chart"
        keep = rdp(Q, 0.25) if len(Q) > 60 else np.arange(len(Q))
        out.append(stroke_with_halo(d_poly(Q[keep]), c, 2.0, th["halo"]))
        if r.cfg.dots:
            out += [dot(x, y, c, th["halo"], r=2.6) for x, y in Q[:-1]]
        if r.n > HERO_DIAMOND_MIN:
            out += [diamond(*Q[km - 1], 4.0, c, th["halo"]) for km in HERO_MILESTONES if km < r.n]
        out.append(dot(*Q[-1], c, th["halo"], r=3.8))
        if r.cfg.id == "damped_newton":
            # Newton's local rate is quadratic (Nocedal & Wright 2006, Thm. 3.5); its last step
            # lands two decades under the line, right of which the chart is empty.
            s, _ = fill_path(
                ts.text("quadratic", fs["rate"], "serif-italic"),
                Q[-1][0] + 9,
                Q[-1][1] + 5,
                th["text2"],
                "start",
                th["halo"],
            )
            out.append(s)
    return out


def hero_legend(runs: list[HeroRun], theme: str, x: float, y: float, fs: dict, rows: bool):
    th = hero_tokens(theme)
    out = []
    for r in runs:
        c = series(th, r.cfg)
        yy = y - fs["legend"] * 0.33
        out.append(
            f'<path d="M{x} {yy:.1f}h20" stroke="{c}" stroke-width="2.6" stroke-linecap="round"/>'
        )
        if r.cfg.dots:
            out.append(
                f'<circle cx="{x + 10}" cy="{yy:.1f}" r="3.2" fill="{c}" stroke="{th["halo"]}" stroke-width="1.2"/>'
            )
        s, w = fill_path(ts.text(r.cfg.label, fs["legend"], "inter-medium"), x + 28, y, th["text"])
        out.append(s)
        s, w2 = fill_path(
            ts.text(count(r.n), fs["legend"]), x + 28 + w + 6, y, th["text3"], tnum=True
        )
        out.append(s)
        if rows:
            y += fs["legend"] + 12
        else:
            x += 28 + w + 6 + w2 + 24
    return out, x - 24


def hero_description(runs: list[HeroRun], star: np.ndarray) -> str:
    """Screen-reader text with the brand's typography (U+2212, thousands separators)."""
    parts = [f"{r.cfg.label}, {count(r.n)}" for r in sorted(runs, key=lambda r: r.n)]
    return (
        f"Four optimizers on Himmelblau's function from x₀ = ({num(HERO_X0[0])}, {num(HERO_X0[1], 1)}) "
        f"to the minimizer x⋆ ≈ ({num(star[0], 3)}, {num(star[1], 3)}), each stopped at the first "
        "iterate with ‖∇f(xₖ)‖₂ ≤ 10⁻⁸. Iterations: "
        + "; ".join(parts)
        + ". Right: ‖∇f(xₖ)‖₂ against k on log–log axes; Damped Newton's curve drops quadratically, "
        "heavy-ball momentum spirals into the minimizer and its gradient norm oscillates as it falls."
    )


def build_hero(theme: str, lay: HeroLayout) -> tuple[str, float]:
    p, runs, star = HERO_DATA
    th = hero_tokens(theme)
    fs = lay.fs
    (dx0, dx1), (dy0, dy1) = lay.domain
    lw = lay.field_w
    lh = lw * (dy1 - dy0) / (dx1 - dx0)  # equal scale on both axes
    L = (lay.field_xy[0], lay.field_xy[1], lw, lh)
    if lay.stacked:
        C = (92, L[1] + lh + 96, 420, 230)
        legend_y = C[1] + C[3] + 100
        Hh = legend_y + 3 * (fs["legend"] + 12) + 12
    else:
        C = lay.conv
        Hh = L[1] + lh + 74
        legend_y = Hh - 8
    Hh = math.ceil(Hh)
    W = lay.W
    tag = f"hero-{lay.name}-{theme}"

    # Every drawn iterate lies inside the landscape panel (no step leaves the view).
    for r in runs:
        P = box_px(r.xs, L, lay.domain)
        assert (P[:, 0] > L[0] + 6).all() and (P[:, 0] < L[0] + lw - 6).all(), r.cfg.id
        assert (P[:, 1] > L[1] + 6).all() and (P[:, 1] < L[1] + lh - 6).all(), r.cfg.id

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" role="img" '
        f'aria-labelledby="t-{tag} d-{tag}">',
        f'<title id="t-{tag}">Four optimizers on Himmelblau\'s function, computed with numopt</title>',
        f'<desc id="d-{tag}">{hero_description(runs, star)}</desc>',
    ]
    # Header: problem and formula over the landscape; quantity over the convergence chart.
    s, w = fill_path(
        ts.text("Himmelblau", fs["title"], "inter-semibold"),
        L[0],
        26 if lay.stacked else 22,
        th["text"],
    )
    out.append(s)
    formula = ts.math("f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2", fs["formula"])
    if lay.stacked:
        out.append(fill_path(formula, L[0], 56, th["text2"])[0])
    else:
        out.append(fill_path(formula, L[0] + w + 12, 22, th["text2"])[0])
    hx, hy = (L[0], C[1] - 34) if lay.stacked else (C[0], 22)
    s, w = fill_path(ts.text("Convergence", fs["title"], "inter-semibold"), hx, hy, th["text"])
    out.append(s)
    out.append(
        fill_path(
            tex("\\|\\nabla f(\\mathbf{x}_k)\\|_2", fs["formula"]), hx + w + 12, hy, th["text2"]
        )[0]
    )

    out += hero_field(p, theme, L, lay.domain, tag, runs)
    # x0 (hollow ring) and x* (cross), with x*'s coordinates keyed in the empty lower-right corner.
    sx, sy = box_px(HERO_X0, L, lay.domain)
    out.append(x0_ring(sx, sy, th))
    out.append(
        fill_path(
            ts.math("\\mathbf{x}_0", fs["label"]),
            sx - 12,
            sy + fs["label"] + 6,
            th["text"],
            "end",
            th["halo"],
        )[0]
    )
    mx, my = box_px(star, L, lay.domain)
    out.append(star_cross(mx, my, th))
    kx, ky = L[0] + lw - 14, L[1] + lh - 14
    s, w = fill_path(
        ts.math(f"\\mathbf{{x}}^\\star \\approx ({star[0]:.3f}, {star[1]:.3f})", fs["label"]),
        kx,
        ky,
        th["text"],
        "end",
        th["halo"],
    )
    out.append(s)
    out.append(star_cross(kx - w - 14, ky - fs["label"] * 0.33, th, s=6))
    # Ticks outside the frame; equal scale on x and y.
    for xv in range(math.ceil(dx0), math.floor(dx1) + 1):
        x, _ = box_px((xv, 0.0), L, lay.domain)
        out.append(f'<path d="M{x:.1f} {L[1] + lh:.1f}v5" stroke="{th["axis"]}"/>')
        out.append(
            fill_path(
                ts.math(str(xv), fs["tick"]), x, L[1] + lh + fs["tick"] + 8, th["text3"], "middle"
            )[0]
        )
    for yv in (2.5, 3.0, 3.5):
        _, y = box_px((0.0, yv), L, lay.domain)
        out.append(f'<path d="M{L[0]} {y:.1f}h-5" stroke="{th["axis"]}"/>')
        out.append(
            fill_path(
                ts.math(f"{yv:g}", fs["tick"]), L[0] - 8, y + fs["tick"] * 0.36, th["text3"], "end"
            )[0]
        )
    out.append(
        fill_path(
            ts.math("x", fs["axis"] + 1), L[0] + lw, L[1] + lh + fs["tick"] + 8, th["text2"], "end"
        )[0]
    )
    out.append(
        fill_path(ts.math("y", fs["axis"] + 1), L[0] - 8, L[1] + fs["axis"], th["text2"], "end")[0]
    )

    out += hero_conv(runs, theme, C, fs)
    legend, right = hero_legend(runs, theme, L[0], legend_y, fs, rows=lay.stacked)
    out += legend
    out.append("</svg>")
    return "\n".join(out) + "\n", right


def raster(svgs: list[Path]) -> None:
    node = shutil.which("node")
    if not node:
        print("node not found: skipping PNG fallbacks")
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    for svg in svgs:
        png = BUILD / svg.with_suffix(".png").name
        subprocess.run(
            [node, str(HERE / "render.mjs"), str(svg), str(png), "--scale", "2"], check=True
        )
        print("wrote", png.relative_to(REPO))


HERO_DATA: tuple[object, list[HeroRun], np.ndarray]


def main() -> None:
    global HERO_DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-raster", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    HERO_DATA = run_hero()
    for r in HERO_DATA[1]:
        print(f"{r.cfg.id:18s} n={r.n:4d}  ||grad f(x_n)||_2 = {r.g[-1]:.2e}")
    written = []
    for lay in (HERO_WIDE, HERO_STACKED):
        for theme in THEMES:
            svg, right = build_hero(theme, lay)
            assert right <= lay.W, f"legend too wide: {right:.0f} > {lay.W}"
            path = OUT / f"hero{'-stacked' if lay.stacked else ''}-{theme}.svg"
            path.write_text(svg)
            size = path.stat().st_size
            assert size <= 400_000, f"{path.name} is {size / 1e3:.0f} kB (> 400 kB)"
            w, h = (float(v) for v in svg.split('viewBox="0 0 ', 1)[1].split('"', 1)[0].split())
            assert lay.stacked or w / h >= 2.0, f"wide hero aspect {w / h:.2f} < 2"
            print("wrote", path.relative_to(REPO), f"{w:.0f} x {h:.0f}", f"{size / 1e3:.0f} kB")
            written.append(path)
    if not args.no_raster:
        raster(written)


if __name__ == "__main__":
    main()
