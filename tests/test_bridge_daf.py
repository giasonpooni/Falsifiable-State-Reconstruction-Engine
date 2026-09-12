"""The DAF bridge: serialized DAF observations -> Observation, with refusals and provenance.

Reads data/daf/*.observations.json: DAF's own serialisation (observation_to_dict) of what
DAF's per-measurement NOAA extractor admitted from DAF's committed fixtures, exported by
tools/export_daf_fixtures.py at the DAF commit named in data/daf/manifest.json. No DAF code
is needed, except by the one test that asks DAF to recompute the ids, which skips unless
DAF_ROOT points at a DAF checkout. The two files marked SYNTHETIC come from DAF's synthetic
revision fixtures (station 9999999, not a real NOAA station) and are used only for the
conflict tests. Where a test needs input DAF never produced (an off-grid time, a reading
with no stated uncertainty, a unit NOAA was not asked for) it edits a copy of a real record
and says so; such a copy's id no longer matches its content, which is exactly what
verify_ids would catch and the bridge, which never re-derives ids, does not check.
"""
import ast
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import set_lcm.bridge.daf as bridge_mod
from set_lcm.bridge.daf import (
    BridgeRefusal, bridge, load_records, series_source_id, strict_json_loads, verify_ids,
)
from set_lcm.testbed.estimators import ESTIMATORS, KFConfig
from set_lcm.testbed.runner import EstimatorSpec, run
from set_lcm.testbed.truth_free import evaluate_truth_free

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "daf"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
DAF_COMMIT = MANIFEST["daf_commit"]

MLLW = ("8454000", "MLLW", "m")
STND = ("8454000", "STND", "m")
SYN = ("9999999", "MLLW", "m")          # SYNTHETIC station
F_MLLW = "noaa_live_8454000_20240115_mllw.observations.json"
F_STND = "noaa_live_8454000_20240115_stnd.observations.json"
F_PRELIM = "noaa_live_8454000_preliminary.observations.json"
F_SYN = "SYNTHETIC_noaa_window_20260101_20260103.observations.json"
F_SYN_REV = "SYNTHETIC_noaa_window_20260101_20260103_revised.observations.json"


def _load(name: str) -> list[dict]:
    return load_records(DATA / name)


def _bridge(records, **kw):
    args = dict(series=[MLLW], time_zone="UTC", cadence_s=360, arrival_policy="replay", latency_s=0.0,
                conflict_policy="refuse", daf_commit=DAF_COMMIT)
    args.update(kw)
    return bridge(records, **args)


def _refusal(reason: str, records, **kw) -> BridgeRefusal:
    with pytest.raises(BridgeRefusal) as exc:
        _bridge(records, **kw)
    assert exc.value.reason == reason, str(exc.value)
    return exc.value


# ---------------------------------------------------------------------------
# the committed evidence
# ---------------------------------------------------------------------------

def test_committed_daf_files_match_their_manifest():
    names = sorted(p.name for p in DATA.glob("*.observations.json"))
    assert names == sorted(f["output"] for f in MANIFEST["files"])
    for f in MANIFEST["files"]:
        raw = (DATA / f["output"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == f["output_sha256"], f["output"]
        assert b"\r" not in raw
        assert len(load_records(DATA / f["output"])) == f["n_observations"] == f["n_readings"]
        assert f["output"].startswith("SYNTHETIC_") == f["synthetic"] == ("synthetic" in f["source_fixture"])
        assert f["binding"]["time_zone"] == "gmt" and "time_zone=gmt" in f["request_url"]
    # the replayed MLLW bytes are the bytes DAF fetched live: its document id is the version id
    # DAF's docs/PHASE_17_LIVE_SCIENTIFIC_OBSERVATION.md transcript records (3bc9041f042eb48f...)
    mllw = next(f for f in MANIFEST["files"] if f["output"] == F_MLLW)
    assert mllw["daf_version_id"].startswith("3bc9041f042eb48f")
    prov = (DATA / "PROVENANCE.md").read_text(encoding="utf-8")
    assert DAF_COMMIT in prov and MANIFEST["vendored_substrate"]["commit"] in prov
    assert "public domain" in prov
    for f in MANIFEST["files"]:
        assert f["source_fixture_sha256"] in prov


def test_strict_loader_refuses_what_is_not_json():
    for bad in ('[{"v": NaN}]', '[{"v": Infinity}]', '[{"v": -Infinity}]', '[{"v": 1, "v": 2}]'):
        with pytest.raises(BridgeRefusal) as exc:
            strict_json_loads(bad)
        assert exc.value.reason == "non_json"
    assert strict_json_loads('[{"v": 1.5}]') == [{"v": 1.5}]


# ---------------------------------------------------------------------------
# a real day
# ---------------------------------------------------------------------------

def test_mllw_day_is_240_points_at_360_s_with_stated_R_and_evidence_ids():
    recs = _load(F_MLLW)
    before = copy.deepcopy(recs)
    bs = _bridge(recs)
    assert recs == before                                    # the evidence is read, never edited
    assert len(bs.observations) == 240
    np.testing.assert_array_equal(bs.inputs.t, np.arange(240) * 360.0)
    assert np.all(bs.inputs.u_commanded == 0.0)
    assert bs.source_ids == ("noaa:8454000:MLLW:m",) == (series_source_id(MLLW),)
    by_time = sorted(recs, key=lambda r: r["content"]["measurement_time"])
    for k, (o, r) in enumerate(zip(bs.observations, by_time)):
        c = r["content"]
        assert o.t == 360.0 * k and o.arrival_t == o.t
        assert o.mask.tolist() == [True] and o.y[0] == c["value"]
        assert c["uncertainty_kind"] == "stated" and o.R[0, 0] == c["uncertainty"] ** 2
        assert o.evidence_ids == (r["id"],) and bs.component_evidence[k] == ((r["id"],),)
        assert o.source_ids == bs.source_ids
    p = bs.provenance
    assert p["daf_commit"] == DAF_COMMIT
    assert (p["n_records_in"], p["n_used"], p["n_ignored"], p["n_deduplicated"], p["n_in_conflict"]) == \
        (240, 240, 0, 0, 0)
    assert p["conflicts"] == [] and p["off_grid"]["n_refused"] == 0 and p["off_grid"]["max_abs_offset_s"] == 0.0
    assert (p["time_zone"], p["epoch_iso"], p["cadence_s"], p["n_grid"]) == ("UTC", "2024-01-15T00:00:00Z", 360.0, 240)
    assert p["arrival_policy"]["policy"] == "replay" and p["arrival_policy"]["latency_s"] == 0.0
    rs = p["R_source"]["noaa:8454000:MLLW:m"]
    assert rs["kind"] == "source-stated" and rs["n_points"] == 240
    assert rs["n_zero"] == sum(1 for r in recs if r["content"]["uncertainty"] == 0.0)   # passed through, never floored
    ids = sorted(r["id"] for r in recs)
    assert bs.evidence_ids_used == tuple(ids) and p["n_evidence_ids"] == 240
    assert p["evidence_ids_sha256"] == hashlib.sha256("\n".join(ids).encode()).hexdigest()
    json.dumps(p, allow_nan=False)                           # provenance is plain JSON


def test_preliminary_day_bridges_on_its_own_clock():
    bs = _bridge(_load(F_PRELIM))
    p = bs.provenance
    assert (p["epoch_iso"], p["n_grid"], p["n_used"]) == ("2026-08-23T00:00:00Z", 240, 240)
    assert all(o.mask[0] for o in bs.observations) and p["R_source"]["noaa:8454000:MLLW:m"]["n_zero"] == 0


def test_stnd_is_its_own_series_and_never_pooled_with_mllw():
    mllw, stnd = _load(F_MLLW), _load(F_STND)
    both = _bridge(mllw + stnd, series=[MLLW, STND])
    assert both.source_ids == ("noaa:8454000:MLLW:m", "noaa:8454000:STND:m")
    y = np.array([o.y for o in both.observations])
    assert y.shape == (240, 2) and np.all(np.isfinite(y))
    m_ids = {r["id"] for r in mllw}
    s_ids = {r["id"] for r in stnd}
    for comp in both.component_evidence:
        assert comp[0][0] in m_ids and comp[1][0] in s_ids
    # the same water surface on two datums: two different numbers, a constant 1.064 m apart
    np.testing.assert_allclose(y[:, 1] - y[:, 0], 1.064, atol=1e-9)
    assert [both.provenance["R_source"][s]["n_zero"] for s in both.source_ids] == [2, 2]   # stated 0.000
    # listing only one datum takes that one and counts the other as ignored, not pooled
    only_m = _bridge(mllw + stnd, series=[MLLW])
    assert only_m.provenance["n_ignored"] == 240 and only_m.provenance["n_used"] == 240
    assert only_m.provenance["ignored"] == [{"series_key": list(STND), "n": 240}]
    assert set(only_m.evidence_ids_used) == m_ids
    only_s = _bridge(mllw + stnd, series=[STND])
    assert set(only_s.evidence_ids_used) == s_ids
    np.testing.assert_array_equal(np.array([o.y[0] for o in only_s.observations]), y[:, 1])


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_refuses_without_a_declared_time_zone_or_with_an_unsupported_one():
    recs = _load(F_MLLW)
    kw = dict(series=[MLLW], cadence_s=360, arrival_policy="replay", latency_s=0.0, conflict_policy="refuse",
              daf_commit=DAF_COMMIT)
    with pytest.raises(TypeError):
        bridge(recs, **kw)                                   # no default: it must be declared
    _refusal("time_zone", recs, time_zone=None)              # naive parsing refused
    for zone in ("America/New_York", "EST", "local", "+00:00", ""):
        _refusal("time_zone", recs, time_zone=zone)
    assert _bridge(recs, time_zone="gmt").provenance["time_zone"] == "UTC"


def test_refuses_off_grid_readings_listing_every_one():
    recs = _load(F_MLLW)
    moved = copy.deepcopy(recs)
    moved[17]["content"]["measurement_time"] = "2024-01-15 01:45"     # edited copy: 3 min off the 6-min grid
    e = _refusal("off_grid", moved)
    assert e.evidence_ids == (moved[17]["id"],)
    # within the 1 s tolerance is on the grid
    near = copy.deepcopy(recs)
    near[17]["content"]["measurement_time"] = "2024-01-15 01:42:01"
    assert _bridge(near).provenance["off_grid"]["max_abs_offset_s"] == 1.0
    # a wrong cadence is refused, not resampled: at 720 s every other real reading is off the grid
    e = _refusal("off_grid", recs, cadence_s=720)
    assert len(e.evidence_ids) == 120


def test_refuses_readings_without_stated_uncertainty_unless_declared_for_the_whole_series():
    recs = _load(F_MLLW)
    stripped = copy.deepcopy(recs)
    for r in stripped[:3]:                                   # edited copies: the source stated none
        del r["content"]["uncertainty"], r["content"]["uncertainty_kind"]
    e = _refusal("missing_uncertainty", stripped)
    assert set(e.evidence_ids) == {r["id"] for r in stripped[:3]}
    # a declared sigma is never mixed with source-stated ones in one series
    e = _refusal("mixed_uncertainty", stripped, declared_sigma={MLLW: 0.01})
    assert len(e.evidence_ids) == 237
    _refusal("mixed_uncertainty", recs, declared_sigma={MLLW: 0.01})
    # a series that states none at all may be given one, recorded as consumer-declared
    bare = copy.deepcopy(recs)
    for r in bare:
        del r["content"]["uncertainty"], r["content"]["uncertainty_kind"]
    _refusal("missing_uncertainty", bare)
    bs = _bridge(bare, declared_sigma={MLLW: 0.01})
    assert all(o.R[0, 0] == pytest.approx(1e-4, rel=0, abs=1e-18) for o in bs.observations)
    assert bs.provenance["R_source"]["noaa:8454000:MLLW:m"]["kind"] == "consumer-declared"
    with pytest.raises(ValueError):
        _bridge(bare, declared_sigma={STND: 0.01})           # not a listed series


def test_refuses_mixed_units_and_never_pools_a_unit_it_was_not_asked_for():
    recs = _load(F_MLLW)
    e = _refusal("mixed_units", recs, series=[MLLW, ("8454000", "MLLW", "ft")])
    assert "converts no units" in str(e)
    feet = copy.deepcopy(recs)
    for r in feet:                                           # edited copies: the same day re-labelled in feet
        r["content"]["unit"] = "ft"
        r["content"]["value"] = r["content"]["value"] / 0.3048
        r["id"] = "ft-" + r["id"]
    bs = _bridge(recs + feet)
    assert bs.provenance["n_ignored"] == 240
    assert bs.provenance["ignored"] == [{"series_key": ["8454000", "MLLW", "ft"], "n": 240}]
    assert not any(e.startswith("ft-") for e in bs.evidence_ids_used)
    _refusal("empty_series", feet)                           # a listed series that matched nothing


def test_refuses_what_is_not_a_per_measurement_noaa_observation():
    recs = _load(F_MLLW)
    other = copy.deepcopy(recs[0])
    other["extraction_method"] = "json:noaa_water_level_v1"
    _refusal("not_noaa_measurement", recs + [other])
    nan = copy.deepcopy(recs)
    nan[5]["content"]["value"] = float("nan")
    _refusal("malformed", nan)
    iso = copy.deepcopy(recs)
    iso[5]["content"]["measurement_time"] = "2024-01-15T00:30:00Z"   # a zone inside the string: refused, not guessed
    _refusal("measurement_time", iso)
    twice = copy.deepcopy(recs)
    twice.append(copy.deepcopy(twice[9]))
    twice[-1]["content"]["value"] += 0.1                     # one id, two contents
    _refusal("id_reused", twice)
    moved = copy.deepcopy(recs)
    moved[7]["id"] = moved[6]["id"]                          # one id, two contents at two grid points
    assert _refusal("id_reused", moved).evidence_ids == (moved[6]["id"],)
    elsewhere = copy.deepcopy(recs[6])                       # ... or in a group the caller did not list
    elsewhere["content"]["datum"] = elsewhere["content"]["conditions"]["datum"] = "STND"
    _refusal("id_reused", recs + [elsewhere])


# ---------------------------------------------------------------------------
# SYNTHETIC revision fixtures: conflicts and deduplication
# ---------------------------------------------------------------------------

def test_synthetic_revision_conflict_is_refused_naming_both_ids():
    orig, rev = _load(F_SYN), _load(F_SYN_REV)
    ids = sorted(r["id"] for r in orig + rev if r["content"]["measurement_time"] == "2026-01-03 00:00")
    e = _refusal("conflict", orig + rev, series=[SYN], cadence_s=21600)
    assert e.evidence_ids == tuple(ids) and all(i in str(e) for i in ids)


def test_synthetic_revision_conflict_report_and_keep_both_leaves_the_point_missing():
    orig, rev = _load(F_SYN), _load(F_SYN_REV)
    bs = _bridge(orig + rev, series=[SYN], cadence_s=21600, conflict_policy="report_and_keep_both")
    p = bs.provenance
    assert p["n_grid"] == 9 and p["epoch_iso"] == "2026-01-01T00:00:00Z"
    present = [k for k, o in enumerate(bs.observations) if o.mask[0]]
    assert present == [0, 1, 4]                              # 2026-01-03 00:00 (k = 8) is missing
    assert np.isnan(bs.observations[8].y[0]) and bs.observations[8].evidence_ids == ()
    assert [bs.observations[k].y[0] for k in present] == [1.1, 1.3, 1.15]
    (c,) = p["conflicts"]
    assert c["series"] == "noaa:9999999:MLLW:m" and c["measurement_time"] == "2026-01-03 00:00"
    assert sorted((x["value"], x["sigma"]) for x in c["readings"]) == [(1.2, 0.011), (1.207, 0.006)]
    conflicting = {x["evidence_id"] for x in c["readings"]}
    # the three unchanged readings: deduplicated, both ids kept, on the one value
    for k in present:
        mt = bs.observations[k]
        both = tuple(sorted(r["id"] for r in orig + rev
                            if r["content"]["value"] == mt.y[0] and r["content"]["measurement_time"] != "2026-01-03 00:00"))
        assert len(both) == 2 and mt.evidence_ids == both == bs.component_evidence[k][0]
    assert (p["n_records_in"], p["n_used"], p["n_deduplicated"], p["n_in_conflict"]) == (8, 6, 3, 2)
    assert not conflicting & set(bs.evidence_ids_used) and len(bs.evidence_ids_used) == 6
    # neither window wins: each alone is a clean record with its own value at 2026-01-03 00:00
    for recs, v in ((orig, 1.2), (rev, 1.207)):
        assert _bridge(recs, series=[SYN], cadence_s=21600).observations[8].y[0] == v


def test_identical_re_extraction_is_deduplicated():
    recs = _load(F_SYN)
    bs = _bridge(recs + copy.deepcopy(recs), series=[SYN], cadence_s=21600)
    assert bs.provenance["n_deduplicated"] == 4 and bs.provenance["n_used"] == 8
    assert bs.evidence_ids_used == tuple(sorted(r["id"] for r in recs))
    assert all(len(o.evidence_ids) == len(set(o.evidence_ids)) for o in bs.observations)


# ---------------------------------------------------------------------------
# arrival
# ---------------------------------------------------------------------------

def test_arrival_policies():
    recs = _load(F_MLLW)
    late = _bridge(recs, latency_s=720.0)
    assert all(o.arrival_t == o.t + 720.0 for o in late.observations)
    with pytest.raises(ValueError):
        _bridge(recs, latency_s=None)                        # replay needs an explicit latency
    with pytest.raises(ValueError):
        _bridge(recs, arrival_policy="as_acquired", latency_s=5.0)
    acq = _bridge(recs, arrival_policy="as_acquired", latency_s=None)
    # DAF's extracted_at for this replayed recording is 2026-08-25T00:00:00Z (data/daf/PROVENANCE.md)
    lag = (np.datetime64("2026-08-25T00:00:00") - np.datetime64("2024-01-15T00:00:00")) / np.timedelta64(1, "s")
    assert all(o.arrival_t == lag for o in acq.observations)
    assert acq.provenance["arrival_policy"]["policy"] == "as_acquired"
    missing = copy.deepcopy(recs)
    del missing[3]["extracted_at"]
    assert _refusal("arrival", missing, arrival_policy="as_acquired", latency_s=None).evidence_ids == (missing[3]["id"],)
    naive = copy.deepcopy(recs)
    naive[3]["extracted_at"] = "2026-08-25T00:00:00"
    _refusal("arrival", naive, arrival_policy="as_acquired", latency_s=None)
    early = copy.deepcopy(recs)
    early[3]["extracted_at"] = "2023-01-01T00:00:00Z"
    _refusal("arrival", early, arrival_policy="as_acquired", latency_s=None)
    # a deduplicated SYNTHETIC reading arrives when the later of its two records did
    syn = _bridge(_load(F_SYN) + _load(F_SYN_REV), series=[SYN], cadence_s=21600, arrival_policy="as_acquired",
                  latency_s=None, conflict_policy="report_and_keep_both")
    t_rev = (np.datetime64("2026-08-26T00:00:00") - np.datetime64("2026-01-01T00:00:00")) / np.timedelta64(1, "s")
    assert syn.observations[0].arrival_t == t_rev
    assert syn.observations[8].arrival_t == syn.observations[8].t     # no reading: nothing to wait for


# ---------------------------------------------------------------------------
# through run()
# ---------------------------------------------------------------------------

class _Walk:
    """TEST-ONLY, not a water-level model: one random walk per sensor, read one-to-one,
    Var(w) = Q_STEP per grid step. It exists to push a bridged record through run(); its
    numbers are arbitrary and nothing is claimed about them."""

    model_version = "test-walk-v0"
    config_cls = KFConfig
    n_report = 1
    aug_names: tuple = ()
    aug_nominal: tuple = ()
    Q_STEP = 0.05 ** 2

    def __init__(self, x0, p0_std, dt, u_cmd, cfg=KFConfig()):
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.P0 = np.eye(1) * float(p0_std) ** 2
        self._x, self._P = self.x0.copy(), self.P0.copy()
        self.xf, self.Pf = [], []
        self.innov, self.innov_var, self.innov_z = [], [], []

    def ingest(self, obs, j):
        assert j == len(self.xf)
        if j > 0:
            self._P = self._P + self.Q_STEP
        m = obs.mask
        nu_full, s_full = np.full(obs.y.size, np.nan), np.full(obs.y.size, np.nan)
        if m.any():
            H = np.eye(1)[m]
            R = obs.R[np.ix_(m, m)]
            S = H @ self._P @ H.T + R
            K = self._P @ H.T @ np.linalg.inv(S)
            nu = obs.y[m] - H @ self._x
            self._x = self._x + K @ nu
            I_KH = np.eye(1) - K @ H
            self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
            nu_full[m], s_full[m] = nu, np.diag(S)
        self.xf.append(self._x.copy())
        self.Pf.append(self._P.copy())
        self.innov.append(nu_full)
        self.innov_var.append(s_full)
        self.innov_z.append(nu_full / np.sqrt(s_full))

    def report(self, k):
        j = len(self.xf) - 1
        if j < 0:
            return self.x0.copy(), self.P0 + k * self.Q_STEP
        return self.xf[j].copy(), self.Pf[j] + (k - j) * self.Q_STEP


def test_a_bridged_record_runs_and_ingests_exactly_the_used_ids(monkeypatch):
    monkeypatch.setitem(ESTIMATORS, "test_walk", _Walk)
    spec = EstimatorSpec("walk", "test_walk", None)
    bs = _bridge(_load(F_MLLW))
    rr = run(bs.inputs, bs.observations, None, spec, (0.0,), 2.0)
    flat = [e for ids in rr.ingested_evidence for e in ids]
    assert len(flat) == len(set(flat)) and tuple(sorted(flat)) == bs.evidence_ids_used
    assert rr.ingested_evidence == [o.evidence_ids for o in bs.observations]
    assert rr.observed[:, 0].all() and np.isfinite(rr.innov_z).all()
    free = evaluate_truth_free(rr, {"day": (0, 240)})
    assert free["windows"]["day"]["n_evidence_ids"] == 240
    # SYNTHETIC revision with the conflict kept out: the conflicting ids are never ingested
    syn = _bridge(_load(F_SYN) + _load(F_SYN_REV), series=[SYN], cadence_s=21600,
                  conflict_policy="report_and_keep_both")
    rs = run(syn.inputs, syn.observations, None, spec, (1.0,), 1.0)
    got = sorted(e for ids in rs.ingested_evidence for e in ids)
    assert tuple(got) == syn.evidence_ids_used
    assert not {x["evidence_id"] for x in syn.provenance["conflicts"][0]["readings"]} & set(got)
    # under as_acquired the 2024 record was acquired long after the day ended: nothing arrives in the run
    acq = _bridge(_load(F_MLLW), arrival_policy="as_acquired", latency_s=None)
    ra = run(acq.inputs, acq.observations, None, spec, (0.0,), 2.0)
    assert ra.ingested_evidence == [()] * 240 and not ra.observed.any()


# ---------------------------------------------------------------------------
# the boundary: no DAF at runtime; DAF recomputes the ids when asked
# ---------------------------------------------------------------------------

def test_bridge_imports_nothing_from_daf_outside_verify_ids():
    tree = ast.parse(Path(bridge_mod.__file__).read_text(encoding="utf-8"))
    forbidden = {"daf", "evidence", "scout", "retrieval", "materials"}

    def roots(node) -> set:
        if isinstance(node, ast.Import):
            return {a.name.split(".")[0] for a in node.names}
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            return {(node.module or "").split(".")[0]}
        return set()

    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    verify = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "verify_ids")
    in_verify = {id(n) for n in ast.walk(verify)}
    outside = set().union(*(roots(n) for n in imports if id(n) not in in_verify))
    inside = set().union(*(roots(n) for n in imports if id(n) in in_verify))
    assert not outside & forbidden, outside & forbidden
    assert inside & forbidden == {"daf"}


def _daf_state(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--ignored"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(ln for ln in out.splitlines() if ".venv" not in ln)


needs_daf = pytest.mark.skipif(not os.environ.get("DAF_ROOT"), reason="set DAF_ROOT to a DAF checkout")


@needs_daf
def test_daf_recomputes_every_committed_id():
    root = Path(os.environ["DAF_ROOT"])
    before = _daf_state(root)
    records = [r for f in MANIFEST["files"] for r in _load(f["output"])]
    assert verify_ids(records, root) == len(records) == 728
    tampered = copy.deepcopy(records[0])
    tampered["content"]["value"] += 0.001
    with pytest.raises(Exception) as exc:
        verify_ids([tampered], root)
    assert type(exc.value).__name__ == "ArtifactIdentityMismatch"
    assert _daf_state(root) == before                         # the checkout gained no files


@needs_daf
def test_export_tool_reproduces_the_committed_files(tmp_path):
    """Re-running DAF's extraction on DAF's fixtures (tools/export_daf_fixtures.py) gives the
    committed data/daf/ byte for byte (manifest and PROVENANCE.md up to the Python version
    they record), and leaves the DAF checkout as it found it."""
    root = Path(os.environ["DAF_ROOT"])
    before = _daf_state(root)
    subprocess.run(["uv", "run", "--python", "3.13", "python", str(REPO / "tools" / "export_daf_fixtures.py"),
                    "--out", str(tmp_path)], cwd=REPO, check=True, capture_output=True, timeout=600)
    for f in MANIFEST["files"]:
        assert (tmp_path / f["output"]).read_bytes() == (DATA / f["output"]).read_bytes(), f["output"]
    fresh = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in fresh.items() if k != "python"} == {k: v for k, v in MANIFEST.items() if k != "python"}
    py = MANIFEST["python"]
    assert (tmp_path / "PROVENANCE.md").read_text(encoding="utf-8").replace(fresh["python"], py) == \
        (DATA / "PROVENANCE.md").read_text(encoding="utf-8")
    assert _daf_state(root) == before
