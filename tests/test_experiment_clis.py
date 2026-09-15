"""Every runnable experiment refuses an argument it does not understand.

`results/` is a verified artifact, and the command that regenerates it is the only thing
standing between a committed report and an accidental overwrite. Four modules --
`real_noaa`, `real_water_balance`, `calibration` and `sweep` -- parsed no arguments at all.
The first two SILENTLY IGNORED everything and wrote to `results/` regardless, so a
`--out-dir /tmp/...` passed precisely to keep the committed artifacts safe overwrote them
instead, with no error and no clue. The other two turned any flag into `ValueError` from
`int(sys.argv[1])`, which is at least loud. Three of the four also defaulted their output to
`Path.cwd() / "results"`, so where the artifact landed depended on where the command was run.

This walks every experiment module with a `__main__` block and requires the two properties
that make the regeneration commands in docs/USAGE.md safe to trust:

    an unknown flag is an ERROR, not a silent no-op that regenerates anyway; and
    `--out-dir` exists, so a run can be sent somewhere other than the committed artifact.

`--help` is used to exercise the parser because it exits before any experiment runs, which
keeps this test cheap and, more importantly, keeps it from writing to `results/` itself.
"""
import ast
import pathlib
import subprocess
import sys

import pytest

from set_lcm.experiments.provenance import REPO_ROOT

EXPERIMENTS = REPO_ROOT / "src" / "set_lcm" / "experiments"


def _runnable():
    """Experiment modules with a `__main__` block, i.e. the ones a person can invoke."""
    found = []
    for path in sorted(EXPERIMENTS.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if (isinstance(node, ast.If) and ast.dump(node.test).find("__main__") != -1):
                found.append(path.stem)
                break
    assert len(found) >= 15, f"expected most experiments to be runnable, found {found}"
    return found


RUNNABLE = _runnable()


def _run(module, *args):
    return subprocess.run([sys.executable, "-m", f"set_lcm.experiments.{module}", *args],
                          capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)


@pytest.mark.parametrize("module", RUNNABLE)
def test_an_unknown_flag_is_refused_rather_than_ignored(module):
    """The defect this exists for: an ignored flag meant the run happened anyway."""
    result = _run(module, "--this-is-not-a-real-flag")
    assert result.returncode != 0, (
        f"{module} accepted an unknown flag; if it ignored it and ran, a --out-dir meant to "
        f"protect results/ would overwrite it instead")
    assert result.returncode == 2, f"{module} should fail argument parsing, not crash: {result.stderr[-400:]}"


@pytest.mark.parametrize("module", RUNNABLE)
def test_every_experiment_can_be_sent_somewhere_other_than_the_committed_artifact(module):
    result = _run(module, "--help")
    assert result.returncode == 0, result.stderr[-400:]
    assert "--out-dir" in result.stdout, f"{module} cannot write anywhere but results/"


def test_no_experiment_defaults_its_output_to_the_working_directory():
    """`Path.cwd() / "results"` made the artifact's location depend on where the command ran.

    Checked in the source rather than by running, because the failure is a default value and
    running from the repository root would hide it.
    """
    offenders = [path.name for path in sorted(EXPERIMENTS.glob("*.py"))
                 if 'Path.cwd() / "results"' in path.read_text(encoding="utf-8")]
    assert not offenders, f"output location depends on the working directory: {offenders}"
