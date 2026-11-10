"""Color families for the charts: one hue per ranking, stepped in lightness for its settings.

The user's rule (2026-11-10): a line without discrete data points gets no markers and its own
color; lines of one family (a model's seeds and windows, expand_context 0/1/2/3/6, k values,
chunkers, budgets) get colors close together but still distinguishable: one hue stepped in
lightness with a slight hue shift. The base convention holds: the first stage is the blue
family, each model its own hue family, the bound gray.

family(base, n) gives n colors in OKLCH: lightness evenly spaced over a band that widens with
n, the hue turning a little with lightness, chroma reduced only where sRGB cannot hold it.
distances() checks a family the way the dataviz palette validator does (OKLab distance x100,
for normal vision and simulated deuteranopia and protanopia); tests/test_palette.py requires
adjacent members at least FLOOR apart for normal vision and CVD_FLOOR under color-vision
deficiency (lines carry labels or legend swatches as the second cue).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_hex, to_rgb

FLOOR, CVD_FLOOR = 15.0, 6.0

# Machado et al. (2009), severity 1.0, in linear RGB
CVD = {"deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                                 [-0.011820, 0.042940, 0.968881]]),
       "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                               [-0.003882, -0.048116, 1.051998]])}


def _linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gamma(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def _lab_from_linear(rgb: np.ndarray) -> np.ndarray:
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ np.cbrt(m1 @ rgb)


def _linear_from_lab(lab: np.ndarray) -> np.ndarray:
    m2i = np.array([[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
    m1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    return m1i @ (m2i @ lab) ** 3


def oklab(color: str) -> np.ndarray:
    return _lab_from_linear(_linear(np.asarray(to_rgb(color))))


def _from_lch(L: float, C: float, h: float) -> np.ndarray | None:
    """Linear RGB of an OKLCH color, None when outside sRGB."""
    rgb = _linear_from_lab(np.array([L, C * np.cos(h), C * np.sin(h)]))
    return rgb if np.all(rgb >= -1e-6) and np.all(rgb <= 1 + 1e-6) else None


def family(base: str, n: int) -> list[str]:
    """n colors of base's hue, darkest first: lightness evenly spaced (a band of 0.20 for two,
    widening by 0.14 per member to 0.64, centred near the base's lightness and kept within
    0.22-0.86 so the lightest stays visible on the white surface), the hue turning up to 16
    degrees either way across the family, chroma reduced only to stay inside sRGB."""
    if n <= 1:
        return [to_hex(base)]
    lab = oklab(base)
    L0, C0, h0 = lab[0], float(np.hypot(lab[1], lab[2])), float(np.arctan2(lab[2], lab[1]))
    band = min(0.64, 0.20 + 0.14 * (n - 2))
    centre = min(max(L0, 0.22 + band / 2), 0.86 - band / 2)  # keep the lightest visible on the white surface
    out = []
    for i, L in enumerate(np.linspace(centre - band / 2, centre + band / 2, n)):
        h = h0 + np.radians(16.0) * (i - (n - 1) / 2) / max(1, (n - 1) / 2)
        C = C0
        rgb = _from_lch(L, C, h)
        while rgb is None and C > 0.005:
            C *= 0.95
            rgb = _from_lch(L, C, h)
        out.append(to_hex(np.clip(_gamma(np.clip(rgb, 0, 1)), 0, 1)))
    return out


def distances(colors: list[str]) -> dict[str, list[float]]:
    """Adjacent-pair OKLab distances (x100): normal vision and each simulated deficiency."""
    lin = [_linear(np.asarray(to_rgb(c))) for c in colors]
    out = {}
    for name, matrix in {"normal": np.eye(3), **CVD}.items():
        labs = [_lab_from_linear(np.clip(matrix @ x, 0, 1)) for x in lin]
        out[name] = [float(100 * np.linalg.norm(a - b)) for a, b in zip(labs, labs[1:], strict=False)]
    return out
