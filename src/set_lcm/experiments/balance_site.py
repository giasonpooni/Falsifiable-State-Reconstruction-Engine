"""One reservoir's declaration, resolved into what a water-balance study needs from it.

The first site's constants lived in `real_water_balance.py` as module globals; the
declaration format moved them into `declarations/ridgway.toml`. This is the piece that was
still missing when a SECOND site arrived: the experiment's functions read those globals
directly, so a second reservoir would still have needed a second copy of them.

A Site is a view of one Declaration. Nothing here is site-specific -- that is the point. It
resolves the sensors into bridge selectors, the declared sigmas into DeclaredSigma with their
citations, the declared scalars into a BalanceConfig, and the variants into ConstraintSets
with b taken from a storage reading.

WHAT THE SECOND SITE COST, since that was the claim under test. The declaration format needed
nothing: taylor_park.toml uses the same schema Ridgway does. The FILTERS needed generalising
-- `estimators_balance` assumed exactly two gauged inflows, which is a property of Ridgway and
not of a water balance -- and this module needed to exist. Both are one-time costs that a
third site does not pay again, and neither was visible until a site with three inflow gauges
was actually attempted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..bridge.daf import DeclaredSigma, SeriesSelector
from ..declaration import Declaration
from ..declaration import load as load_declaration
from ..schema import ConstraintSet
from ..testbed.estimators_balance import BalanceConfig
from .provenance import REPO_ROOT

__all__ = ["Site", "site_from"]

STORAGE_ROLE = "storage"
CLOSURE_ROW = "closure"


@dataclass(frozen=True)
class Site:
    """Everything a balance study reads from one declaration, resolved once."""
    decl: Declaration

    # ---- identity ----------------------------------------------------------------------
    @property
    def key(self) -> str:
        return self.decl.key

    @property
    def label(self) -> str:
        return self.decl.label

    @property
    def volume_unit(self) -> str:
        """The unit every volume in this study is quoted in, from the state that declares it."""
        return self.decl.states[STORAGE_ROLE].unit

    # ---- evidence ----------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / self.decl.record.directory

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / self.decl.record.manifest

    @property
    def observations(self) -> str:
        return self.decl.record.observations

    @property
    def columns(self) -> tuple[str, ...]:
        """Sensor roles in declared order, which IS the column order the filters read."""
        return tuple(s.role for s in self.decl.sensors)

    @property
    def selectors(self) -> tuple[SeriesSelector, ...]:
        return tuple(SeriesSelector(source_id=s.source_id, extraction_method=s.extraction_method,
                                    property=s.property, match=s.match)
                     for s in self.decl.sensors)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(s.source_id for s in self.decl.sensors)

    @property
    def n_inflows(self) -> int:
        """Sensors other than the storage gauge and the outlet gauge.

        Cross-checked against the declared scalar where one exists, because two ways of saying
        the same number is two ways for them to disagree.
        """
        counted = len(self.decl.sensors) - 2
        if "n_inflows" in self.decl.scalars:
            declared = int(self.decl.scalar("n_inflows"))
            if declared != counted:
                raise ValueError(
                    f"{self.key!r} declares n_inflows = {declared} but carries {len(self.decl.sensors)} "
                    f"sensors, which is {counted} inflow gauge(s) after the storage and outlet gauges")
        return counted

    # ---- declared uncertainty ----------------------------------------------------------
    @property
    def storage_sigma_base(self) -> float:
        return self.decl.sensor(STORAGE_ROLE).sigma

    @property
    def storage_sigma_citation(self) -> str:
        return self.decl.sensor(STORAGE_ROLE).citation

    def _a_flow(self):
        return self.decl.sensor(self.columns[1])

    @property
    def flow_sigma_relative(self) -> float:
        return self._a_flow().relative

    @property
    def flow_sigma_floor(self) -> float:
        return self._a_flow().sigma_floor

    @property
    def flow_sigma_citation(self) -> str:
        return self._a_flow().citation

    def declared_sigma(self, storage_sigma: float) -> dict:
        """The consumer-declared R for each column. `storage_sigma` is the sweep's current
        point; every other number, and every citation, is the declaration's."""
        storage = self.decl.sensor(STORAGE_ROLE)
        out = {storage.source_id: DeclaredSigma(sigma=storage_sigma, citation=storage.citation)}
        for sen in self.decl.sensors:
            if sen.role != STORAGE_ROLE:
                out[sen.source_id] = DeclaredSigma(relative=sen.relative, sigma_floor=sen.sigma_floor,
                                                   citation=sen.citation)
        return out

    # ---- declared model ----------------------------------------------------------------
    @property
    def cfg(self) -> BalanceConfig:
        return BalanceConfig(
            q_storage=self.decl.scalar("q_storage"),
            q_flow=self.decl.scalar("q_flow"),
            q_ungauged=self.decl.scalar("q_ungauged"),
            cumulative0_std=self.decl.scalar("cumulative0_std"),
            flow0=self.decl.scalar("flow0"),
            flow0_std=self.decl.scalar("flow0_std"),
            n_inflows=self.n_inflows,
        )

    @property
    def prior_storage_std(self) -> float:
        return self.decl.scalar("prior_storage_std")

    @property
    def gauged_area(self) -> float:
        return self.decl.scalar("gauged_drainage_area")

    @property
    def total_area(self) -> float:
        return self.decl.scalar("total_drainage_area")

    def constraint(self, variant: str, s0: float, s0_var: float) -> ConstraintSet:
        """One of the declaration's variants, with b resolved against the record: b is the
        day-0 storage READING, which is why the declaration names a sensor and not a value."""
        return self.decl.constraint_set(variant, readings={CLOSURE_ROW: s0},
                                        variances={CLOSURE_ROW: s0_var})


def site_from(path: Path | str) -> Site:
    return Site(load_declaration(path))
