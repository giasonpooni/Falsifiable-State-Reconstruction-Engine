"""Comparing a fresh run against a committed results file, and saying what "equal" means.

`results/` is a verified artifact: a fresh run of the source tree must reproduce it. What
that can mean depends on where it is run.

ON THE BUILD THAT GENERATED IT the claim is bitwise: same source, same seeds, identical
arrays, so every value in the file is reproduced exactly. `same_build()` reads the file's
own provenance stamp (python, numpy, platform, machine) and says whether this is that build.

ON ANOTHER BUILD it is not, and never was. numpy 2.5.3 on Windows/AMD64 and on Linux/x86_64
link different LAPACK kernels, so a matrix solve can differ in its last bits; the README has
always put cross-platform bitwise determinism out of scope. Asserting exact equality anyway
would be a claim the repository does not make and cannot keep -- so the reproduction tests
assert a DECLARED tolerance instead, and the declaration is a measurement, not a taste.

Measured 2026-09-12 by regenerating every results file on Linux-6.18 x86_64 (Python 3.12.3,
numpy 2.5.3, gcc 14.2.1) and comparing value by value against the committed files, which
were generated on Windows-11 AMD64 (Python 3.13.5, numpy 2.5.3), excluding the latency
columns and the provenance stamp:

    file               leaves   differ   worst relative difference
    summary.json       75716      4166   5.878e-15
    sweep.json          8918       359   1.191e-13
    calibration.json     330         3   4.828e-16
    real_noaa.json       446        53   3.765e-06

The first three are floating-point noise at the scale of the arithmetic. The fourth is not,
and is the one interesting result of the exercise: every one of real_noaa's larger
deviations is a `tide_kf` log-likelihood at a q FAR FROM the fitted one, where the
nine-state filter is so confident that S is tiny and nu^2/S is a large cancelling sum --
-14290.5616 against -14290.6154 at q = 3.16e-7. At the fitted q the same quantity agrees to
2.8e-10 and the argmax is identical on both builds. The report also prints the non-winning
profile cells: later Windows and Linux CI runs showed that permitted differences can change
their final displayed digit (for example -14470.4 versus -14470.2 or -14470.3). This does not
change the fitted q or relax any decision check. Markdown is therefore checked exactly
against the renderer of its own JSON and saved generation header, while this comparator
checks numerical reproduction across builds. The tolerances remain a tight one per file
and one declared exception for profile log-likelihoods.

A deviation beyond the declared tolerance is a failure on every build. These numbers leave
two to three orders of magnitude of headroom over what was measured, so a real regression
does not hide behind them.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "TOLERANCE", "EXCEPTIONS", "Deviation", "ToleranceException", "compare", "current_build",
    "reproduction_failure", "same_build", "strip",
]

EXCLUDED_KEYS = ("provenance",)
EXCLUDED_PREFIXES = ("latency_us",)


@dataclass(frozen=True)
class ToleranceException:
    """A path pattern allowed a looser tolerance, and why. `contains` matches a path when
    every one of its fragments appears in it, so ('.profile[', '.loglik') names the
    log-likelihood entries of a profile and nothing else."""
    contains: tuple[str, ...]
    rel_tol: float
    why: str
    abs_tol: float = 0.0

    def matches(self, path: str) -> bool:
        return all(fragment in path for fragment in self.contains)


# Per file: the relative tolerance a fresh run must meet on a build other than the one that
# generated the file. Each is roughly 100x the worst deviation measured for that file (see
# the module docstring), so it absorbs build noise and nothing else.
TOLERANCE: dict[str, float] = {
    "summary.json": 1e-12,       # measured 5.878e-15
    "sweep.json": 1e-11,         # measured 1.191e-13
    "calibration.json": 1e-12,   # measured 4.828e-16
    "real_noaa.json": 1e-8,      # measured 2.3e-10 outside the exception below
    "real_water_balance.json": 1e-8,
}

EXCEPTIONS: dict[str, tuple[ToleranceException, ...]] = {
    "real_water_balance.json": (
        ToleranceException(
            contains=(".blind.d.null_space (S + G)",), rel_tol=1e-8, abs_tol=1e-30,
            why="a structurally null direction computed by SVD: measured 0 versus 1.37e-37 "
                "across builds; use the same absolute bound as the analytic null-direction test",
        ),
    ),
    "real_noaa.json": (
        ToleranceException(
            contains=(".profile[", ".loglik"),
            rel_tol=1e-4,        # measured 3.765e-06
            why="the log-likelihood at a q far from the fitted one: S is tiny there and the sum of "
                "nu^2 / S cancels heavily, so the last significant figures are the LAPACK build's. "
                "The argmax and fitted q_scale remain checked; non-winning profile cells may "
                "round differently in the report, which must render its own JSON exactly.",
        ),
    ),
}


@dataclass(frozen=True)
class Deviation:
    path: str
    committed: Any
    fresh: Any
    rel: float | None          # None where the values are not both numbers
    allowed: float
    why: str = ""
    abs_allowed: float = 0.0

    def __str__(self) -> str:
        rel = "not numeric" if self.rel is None else f"{self.rel:.3e}"
        return (f"{self.path}: committed {self.committed!r}, fresh {self.fresh!r} "
                f"(relative {rel}, allowed {self.allowed:.3e}; absolute allowance {self.abs_allowed:.3e})")


def strip(obj):
    """The committed file minus what a rerun is not expected to reproduce: the provenance
    stamp (which records the build, so it differs by construction) and the latency columns
    (wall-clock on whatever machine ran it, and never a claim)."""
    if isinstance(obj, Mapping):
        return {k: strip(v) for k, v in obj.items()
                if k not in EXCLUDED_KEYS and not k.startswith(EXCLUDED_PREFIXES)}
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    return obj


def current_build() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def same_build(generation: Mapping) -> bool:
    """True when this interpreter, numpy, platform and machine are the ones the committed
    file records generating it -- the only circumstance in which bitwise equality is claimed."""
    now = current_build()
    return all(generation.get(k) == v for k, v in now.items())


def _rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    return float("inf") if scale == 0.0 else abs(b - a) / scale


def compare(committed, fresh, *, rel_tol: float,
            exceptions: Sequence[ToleranceException] = ()) -> list[Deviation]:
    """Every leaf of `committed` that `fresh` does not reproduce within tolerance.

    Structure is compared exactly: a missing key, an extra key or a changed list length is a
    deviation whatever the tolerance. Booleans, strings and None must be equal exactly -- a
    tolerance is for arithmetic, not for a claim flag or a label. Numbers must agree to
    `rel_tol` relative, or to the relative/absolute tolerance of the first matching
    exception. Absolute allowances are opt-in by path; labels and counts stay exact.
    """
    out: list[Deviation] = []

    def allowed_for(path: str) -> tuple[float, str, float]:
        for exc in exceptions:
            if exc.matches(path):
                return exc.rel_tol, exc.why, exc.abs_tol
        return rel_tol, "", 0.0

    def walk(a, b, path: str) -> None:
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
                out.append(Deviation(path, type(a).__name__, type(b).__name__, None, *allowed_for(path)))
                return
            for k in sorted(set(a) | set(b)):
                if k in EXCLUDED_KEYS or k.startswith(EXCLUDED_PREFIXES):
                    continue
                if k not in a or k not in b:
                    out.append(Deviation(f"{path}.{k}", "<present>" if k in a else "<absent>",
                                         "<present>" if k in b else "<absent>", None, *allowed_for(path)))
                    continue
                walk(a[k], b[k], f"{path}.{k}")
            return
        if isinstance(a, list) or isinstance(b, list):
            if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
                out.append(Deviation(path, f"list[{len(a)}]" if isinstance(a, list) else type(a).__name__,
                                     f"list[{len(b)}]" if isinstance(b, list) else type(b).__name__,
                                     None, *allowed_for(path)))
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
            return
        numeric = (not isinstance(a, bool) and not isinstance(b, bool)
                   and isinstance(a, (int, float)) and isinstance(b, (int, float)))
        if not numeric:
            if a != b or (isinstance(a, bool) != isinstance(b, bool)):
                out.append(Deviation(path, a, b, None, *allowed_for(path)))
            return
        tol, why, abs_tol = allowed_for(path)
        if not (np.isfinite(a) and np.isfinite(b)):
            out.append(Deviation(path, a, b, float("inf"), tol, why, abs_tol))
            return
        r = _rel(float(a), float(b))
        # Integer-valued report fields represent counts/indices. A change of numeric
        # representation must not turn their exact comparison into a tolerance check.
        counts_changed = (isinstance(a, int) or isinstance(b, int)) and a != b
        if counts_changed or (r > tol and abs(float(a) - float(b)) > abs_tol):
            out.append(Deviation(path, a, b, r, tol, why, abs_tol))

    walk(strip(committed), strip(fresh), "")
    return out


def reproduction_failure(name: str, fresh, committed) -> str | None:
    """None when a fresh run reproduces the committed `results/<name>`, else why not.

    On the build the file records generating it, "reproduces" means bitwise. On any other
    build it means within TOLERANCE[name], with EXCEPTIONS[name] where declared. The
    provenance stamp and the latency columns are excluded either way.
    """
    generation = committed.get("provenance", {}) if isinstance(committed, Mapping) else {}
    generation = generation.get("generation", generation)
    if same_build(generation):
        # Python container equality equates True with 1 and accepts equal infinities.
        # Use the leaf checks on this path too, without cross-build allowances.
        deviations = compare(committed, fresh, rel_tol=0.0)
        if not deviations:
            return None
        return (f"results/{name} does not match a fresh run of the source tree, on the very build that "
                f"generated it ({current_build()}): a source change, not build noise. "
                f"{len(deviations)} value(s) differ:\n  " + "\n  ".join(str(d) for d in deviations[:20]))
    deviations = compare(committed, fresh, rel_tol=TOLERANCE[name], exceptions=EXCEPTIONS.get(name, ()))
    if not deviations:
        return None
    return (f"results/{name} was generated on {dict(generation)} and this is {current_build()}; "
            f"{len(deviations)} value(s) differ by more than the declared cross-build tolerance "
            f"{TOLERANCE[name]:.0e}:\n  " + "\n  ".join(str(d) for d in deviations[:20]))
