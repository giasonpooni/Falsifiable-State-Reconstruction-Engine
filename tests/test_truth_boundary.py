"""The truth boundary, made structural, and the runner's generality.

A real record has no hidden truth. These tests pin that the estimator side cannot
receive one: runner.run() takes PublicInputs (the clock and the commanded input,
copied from a simulated truth by PublicInputs.from_truth), the oracle bound's
hidden arrays travel only through run()'s explicit oracle_inputs keyword, and the
only source code that reads hidden truth fields is the code that routes them
(experiments/phase1.oracle_inputs_for), the observation operator (degrade.observe
reads the masses it measures) and the truth evaluator (evaluate.py, scoring).

They also pin what the runner no longer assumes: the number of sensors comes from
the observations, the reported-state dimension from the estimator (n_report), an
extra component may declare no flag (nominal None), and every observation's
evidence ids are carried to the report step that ingested it. Every existing path
is bit-identical (the reproduction test and the grid say so value for value).
"""
import ast
import inspect
import re
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import set_lcm
from set_lcm.experiments.phase1 import SCENARIOS, SPECS, constraint_for, declared_prior, run_spec
from set_lcm.lcm import chi2_quantile
from set_lcm.schema import ConstraintSet, Observation, Status
from set_lcm.testbed import cusum, estimators, estimators_water, inputs, runner, truth_free
from set_lcm.testbed.degrade import observe
from set_lcm.testbed.estimators import ESTIMATORS, KFConfig
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import PARAM_FLAG_DEBOUNCE, EstimatorSpec, run
from set_lcm.testbed.simulator import simulate

SRC = Path(set_lcm.__file__).parent
HIDDEN_FIELDS = {"m", "leak", "u_actual"}


def spec(name: str) -> EstimatorSpec:
    return next(s for s in SPECS if s.name == name)


def _scenario(name: str = "closed_noise"):
    sc = SCENARIOS[name]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    m0, m0_std = declared_prior(sc, sc.sim)
    return sc, truth, obs, m0, m0_std


# ---------------------------------------------------------------------------
# PublicInputs and the signature of run()
# ---------------------------------------------------------------------------

def test_public_inputs_copy_only_the_public_fields():
    _, truth, _, _, _ = _scenario()
    pi = PublicInputs.from_truth(truth)
    assert {f.name for f in fields(PublicInputs)} == {"t", "u_commanded"}
    np.testing.assert_array_equal(pi.t, truth.t)
    np.testing.assert_array_equal(pi.u_commanded, truth.u_commanded)
    # copies, read-only: nothing on the estimator side aliases or edits the simulator's arrays
    assert not np.shares_memory(pi.t, truth.t) and not np.shares_memory(pi.u_commanded, truth.u_commanded)
    with pytest.raises(ValueError):
        pi.u_commanded[0] = 1.0
    # the hidden fields do not reach it: a truth that differs only in them gives the same inputs
    other = replace(truth, m=truth.m + 9.0, leak=truth.leak + 1.0, u_actual=truth.u_actual * 3.0)
    po = PublicInputs.from_truth(other)
    assert np.array_equal(po.t, pi.t) and np.array_equal(po.u_commanded, pi.u_commanded)
    with pytest.raises(ValueError):
        PublicInputs(t=np.arange(3.0), u_commanded=np.zeros(4))
    with pytest.raises(ValueError):
        PublicInputs(t=np.array([0.0, 2.0, 1.0]), u_commanded=np.zeros(3))


def test_runner_and_estimators_never_name_truth():
    """runner.py and estimators.py (and the other estimator-side modules: the water-level
    filters, the CUSUM, the public inputs, the truth-free evaluator) contain no reference to Truth or
    simulator.Truth -- not in code, not in prose -- and import nothing named Truth; run()
    has no truth parameter and takes PublicInputs first."""
    for mod in (runner, estimators, estimators_water, cusum, inputs, truth_free):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert re.search(r"\bTruth\b", src) is None, mod.__name__
        assert "simulator.Truth" not in src, mod.__name__
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom):
                assert "Truth" not in {a.name for a in node.names}, mod.__name__
                if mod is not estimators:
                    assert not (node.module or "").endswith("simulator"), mod.__name__
                else:   # the linear model's version string is the only thing it takes from there
                    assert (node.module or "") != "simulator" or {a.name for a in node.names} == {"MODEL_VERSION"}
    params = inspect.signature(run).parameters
    assert not any("truth" in p.lower() for p in params)
    assert list(params)[0] == "inputs"
    assert params["oracle_inputs"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["oracle_inputs"].default is None
    tf_params = inspect.signature(truth_free.evaluate_truth_free).parameters
    assert list(tf_params) == ["run", "windows"]


def _hidden_reads(path: Path) -> set[tuple[str | None, str]]:
    """(enclosing function, field) for every `<name containing 'truth'>.<hidden field>`."""
    found: set[tuple[str | None, str]] = set()

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if (isinstance(child, ast.Attribute) and child.attr in HIDDEN_FIELDS
                    and isinstance(child.value, ast.Name) and "truth" in child.value.id.lower()):
                found.add((f, child.attr))
            visit(child, f)

    visit(ast.parse(path.read_text(encoding="utf-8")), None)
    return found


def test_hidden_truth_is_read_only_where_it_is_routed_measured_or_scored():
    reads = {p.relative_to(SRC).as_posix(): _hidden_reads(p) for p in SRC.rglob("*.py")}
    reads = {k: v for k, v in reads.items() if v}
    assert set(reads) == {"experiments/phase1.py", "testbed/degrade.py", "testbed/evaluate.py"}, reads
    # routing: the oracle's two hidden arrays, in one function, and never the masses
    assert reads["experiments/phase1.py"] == {("oracle_inputs_for", "u_actual"), ("oracle_inputs_for", "leak")}
    # the observation operator measures the masses and nothing else hidden
    assert reads["testbed/degrade.py"] == {("observe", "m")}


def test_run_refuses_misrouted_hidden_inputs():
    sc, truth, obs, m0, m0_std = _scenario()
    cs = constraint_for(truth)
    pi = PublicInputs.from_truth(truth)
    hidden = (truth.u_actual, truth.leak)
    for sp in SPECS:
        if sp.kind == "oracle":
            with pytest.raises(ValueError):
                run(pi, obs, cs, sp, m0, m0_std)                        # the bound needs them
        else:
            with pytest.raises(ValueError):
                run(pi, obs, cs, sp, m0, m0_std, oracle_inputs=hidden)  # nothing else may get them
    with pytest.raises(TypeError):
        run(truth, obs, cs, spec("kf"), m0, m0_std)                     # a Truth is not PublicInputs
    with pytest.raises(ValueError):
        run(PublicInputs(t=truth.t[:-1], u_commanded=truth.u_commanded[:-1]), obs, cs, spec("kf"), m0, m0_std)


# ---------------------------------------------------------------------------
# generality: sensors from the observations, reported dimension from the estimator
# ---------------------------------------------------------------------------

class _ToyWalk:
    """A test-only estimator: n_report = 3 components, each a random walk read one-to-one
    by one of three sensors, plus two unobserved constant extra components. 'd' declares
    nominal None (recorded, never flagged); 'e' declares nominal 0.0 against a prior mean
    of 1.0 with sd 0.1, so it is flagged from the third report on."""

    model_version = "toy-walk-v0"
    config_cls = KFConfig
    n_report = 3
    aug_names = ("d", "e")
    aug_nominal = (None, 0.0)
    N = 5

    def __init__(self, x0, p0_std, dt, u_cmd, cfg=KFConfig()):
        self.Q = np.diag([cfg.sigma_w ** 2] * 3 + [0.0, 0.0])
        self.x0 = np.array([*np.asarray(x0, dtype=float), 5.0, 1.0])
        self.P0 = np.diag([float(p0_std) ** 2] * 3 + [0.01, 0.01])
        self._x, self._P = self.x0.copy(), self.P0.copy()
        self.xf, self.Pf = [], []
        self.innov, self.innov_var, self.innov_z = [], [], []

    def ingest(self, obs, j):
        assert j == len(self.xf)
        if j > 0:
            self._P = self._P + self.Q
        m = obs.mask
        nu_full = np.full(obs.y.size, np.nan)
        s_full = np.full(obs.y.size, np.nan)
        if m.any():
            H = np.eye(self.N)[:3][m]
            R = obs.R[np.ix_(m, m)]
            S = H @ self._P @ H.T + R
            K = self._P @ H.T @ np.linalg.inv(S)
            nu = obs.y[m] - H @ self._x
            self._x = self._x + K @ nu
            I_KH = np.eye(self.N) - K @ H
            P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
            self._P = 0.5 * (P + P.T)
            nu_full[m] = nu
            s_full[m] = np.diag(S)
        self.xf.append(self._x.copy())
        self.Pf.append(self._P.copy())
        self.innov.append(nu_full)
        self.innov_var.append(s_full)
        self.innov_z.append(nu_full / np.sqrt(s_full))

    def report(self, k):
        j = len(self.xf) - 1
        if j < 0:
            return self.x0.copy(), self.P0 + k * self.Q
        return self.xf[j].copy(), self.Pf[j] + (k - j) * self.Q


def _toy_record(n: int = 60, seed: int = 5):
    rng = np.random.default_rng(seed)
    x_true = np.array([10.0, 20.0, 70.0])
    R = np.diag([1.0, 1.0, 1.0])
    obs = []
    for j in range(n):
        mask = rng.random(3) > 0.1
        y = np.where(mask, x_true + rng.normal(0.0, 1.0, 3), np.nan)
        obs.append(Observation(float(j), float(j), y, R, mask, ("a", "b", "c"), evidence_ids=(f"ev{j}",)))
    return PublicInputs(t=np.arange(float(n)), u_commanded=np.zeros(n)), obs


def test_runner_sizes_from_the_record_and_the_estimator(monkeypatch):
    monkeypatch.setitem(ESTIMATORS, "toy_walk", _ToyWalk)
    pi, obs = _toy_record()
    n = len(obs)
    cs = ConstraintSet("toy-sum", np.array([[1.0, 1.0, 1.0]]), np.array([100.0]), "a + b + c = 100")
    rr = run(pi, obs, cs, EstimatorSpec("toy+hard", "toy_walk", "hard"), (12.0, 18.0, 68.0), 3.0)
    assert rr.x.shape == rr.x_unproj.shape == rr.corr.shape == (n, 3)
    assert rr.P.shape == rr.P_unproj.shape == (n, 3, 3)
    assert rr.innov.shape == rr.innov_z.shape == rr.observed.shape == rr.cusum_stat.shape == (n, 3)
    # the reconciliation acted on the first n_report components; the unprojected state is kept
    assert all(s is Status.OK for s in rr.status)
    assert np.abs(rr.x @ cs.A.T - cs.b).max() < 1e-9
    assert np.abs(rr.x_unproj @ cs.A.T - cs.b).max() > 1e-3
    np.testing.assert_allclose(rr.corr, rr.x - rr.x_unproj)
    assert rr.threshold == chi2_quantile(1, 0.999)
    # extra components: both recorded; only the one with a nominal is flagged
    assert set(rr.extra) == {"d_hat", "d_sd", "e_hat", "e_sd", "flag_e"}
    assert np.all(rr.extra["d_hat"] == 5.0) and np.all(rr.extra["e_hat"] == 1.0)   # unobserved, uncorrelated
    assert np.all(rr.extra["d_sd"] == 0.1) and np.all(rr.extra["e_sd"] == 0.1)
    expect = np.arange(n) >= PARAM_FLAG_DEBOUNCE - 1
    np.testing.assert_array_equal(rr.extra["flag_e"], expect)
    np.testing.assert_array_equal(rr.observed, np.array([o.mask for o in obs]))
    assert rr.ingested_evidence == [(f"ev{j}",) for j in range(n)]
    # the runner refuses a constraint whose columns do not match n_report, and a record
    # whose observations disagree about the number of sensors
    with pytest.raises(ValueError):
        run(pi, obs, constraint_for(simulate(SCENARIOS["closed_noise"].sim)),
            EstimatorSpec("toy+hard", "toy_walk", "hard"), (12.0, 18.0, 68.0), 3.0)
    bad = list(obs)
    bad[7] = Observation(7.0, 7.0, np.zeros(2), np.eye(2), np.ones(2, dtype=bool), ("a", "b"))
    with pytest.raises(ValueError):
        run(pi, bad, cs, EstimatorSpec("toy+hard", "toy_walk", "hard"), (12.0, 18.0, 68.0), 3.0)


WATER_LEVEL_KINDS = {"level_trend", "tide_kf"}   # estimators_water: one series, the level

# Every estimator's reconciled width, declared here as a table rather than a rule, because the
# families genuinely differ and a new one must be added deliberately. The reconciliation stage
# sees exactly these first components, and a declared constraint's columns must match.
REPORTED_DIMENSION = {
    "kf": 2, "hold_last": 2, "kf_aug": 2, "kf_closedq": 2, "oracle": 2,   # the two masses
    "level_trend": 1, "tide_kf": 1, "tide_month": 1,                      # the water level
    "wb_closed": 1,        # storage; closure is in the dynamics, so no constraint columns
    "wb_open": 2,          # storage and cumulative gauged inflow: A = [1, -1]
    "wb_aug": 3,           # ... and the cumulative ungauged term: A = [1, -1, -1]
}


def test_every_estimator_in_the_tree_declares_its_reported_dimension():
    """The two-reservoir estimators report the two masses; the water-level filters (P4) report
    one component, the level; the balance filters (P4b) report the states their constraint is
    written over. The table above must cover the registry exactly, so a new estimator cannot
    arrive without someone stating its reconciled width."""
    assert WATER_LEVEL_KINDS <= set(ESTIMATORS)
    assert set(REPORTED_DIMENSION) == set(ESTIMATORS), \
        f"undeclared: {sorted(set(ESTIMATORS) - set(REPORTED_DIMENSION))}, " \
        f"stale: {sorted(set(REPORTED_DIMENSION) - set(ESTIMATORS))}"
    for kind, cls in ESTIMATORS.items():
        assert cls.n_report == REPORTED_DIMENSION[kind], kind
        assert len(cls.aug_nominal) == len(cls.aug_names), kind


# ---------------------------------------------------------------------------
# provenance: evidence ids per report step
# ---------------------------------------------------------------------------

def test_observation_evidence_ids_default_and_validation():
    o = Observation(0.0, 0.0, np.zeros(2), np.eye(2), np.ones(2, dtype=bool), ("s1", "s2"))
    assert o.evidence_ids == ()
    assert Observation(0.0, 0.0, np.zeros(2), np.eye(2), np.ones(2, dtype=bool), ("s1", "s2"),
                       evidence_ids=["x", "y"]).evidence_ids == ("x", "y")
    with pytest.raises(TypeError):
        Observation(0.0, 0.0, np.zeros(2), np.eye(2), np.ones(2, dtype=bool), ("s1", "s2"), evidence_ids="xy")
    with pytest.raises(TypeError):
        Observation(0.0, 0.0, np.zeros(2), np.eye(2), np.ones(2, dtype=bool), ("s1", "s2"), evidence_ids=(1,))


def test_ingested_evidence_follows_the_ingest_clock():
    """Simulated runs carry () at every report step. With ids attached and one late
    observation blocking the sampling order (obs 100 arrives at step 104, so 100..104 are
    ingested together at 104), each report step lists the ids of exactly what it
    ingested, in sampling order -- and the estimate is bit-identical to the same record
    without ids: provenance is carried, never read."""
    sc, truth, obs, m0, m0_std = _scenario()
    cs = constraint_for(truth)
    pi = PublicInputs.from_truth(truth)
    plain = run(pi, obs, cs, spec("kf+hard+guard"), m0, m0_std)
    assert plain.ingested_evidence == [()] * len(obs)

    late = list(obs)
    late[100] = replace(late[100], arrival_t=float(truth.t[104]))
    tagged = [replace(o, evidence_ids=(f"noaa:{j}", f"doc:{j}")) for j, o in enumerate(late)]
    a = run(pi, late, cs, spec("kf+hard+guard"), m0, m0_std)
    b = run(pi, tagged, cs, spec("kf+hard+guard"), m0, m0_std)
    for k in (0, 99, 105, 599):
        assert b.ingested_evidence[k] == (f"noaa:{k}", f"doc:{k}")
    for k in (100, 101, 102, 103):
        assert b.ingested_evidence[k] == ()
    assert b.ingested_evidence[104] == tuple(x for j in range(100, 105) for x in (f"noaa:{j}", f"doc:{j}"))
    assert sum(len(ids) for ids in b.ingested_evidence) == 2 * len(obs)
    for key in ("x", "P", "x_unproj", "P_unproj", "stat", "flag", "innov_z", "cusum_stat", "observed"):
        assert np.array_equal(getattr(a, key), getattr(b, key), equal_nan=getattr(a, key).dtype.kind == "f"), key
    assert a.status == b.status
    # the oracle path carries it too (routed through the experiment code)
    rr = run_spec(truth, tagged, cs, spec("oracle (bound)"), m0, m0_std)
    assert rr.ingested_evidence == b.ingested_evidence
