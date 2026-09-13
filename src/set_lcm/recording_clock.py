"""Read-only checks of declared capture timestamps for recording preflight.

No sample is paired by proximity, sorted, interpolated, rounded, or assigned a
different timestamp. Explicit offsets are normalized to UTC only for comparison;
the original strings remain in the result. Agreement of declarations does not
establish physical simultaneity, clock accuracy, timestamp semantics, or timing
uncertainty. Those require a separately documented acquisition/clock model.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
import math
from numbers import Real
import re

__all__ = ["parse_capture_utc", "compare_capture_clocks"]

_CAPTURE_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:[.,](?P<fraction>[0-9]+))?(?:Z|[+-][0-9]{2}:[0-9]{2})"
)


def parse_capture_utc(value: str) -> datetime:
    """Parse an explicit capture instant without silently truncating precision.

    Supported ISO-8601 spelling is YYYY-MM-DDTHH:MM:SS[.ffffff] followed by Z or
    an explicit +/-HH:MM offset. A comma decimal separator is also accepted.
    Fractions must contain one through six digits. A date alone, naive clock,
    whitespace, omitted seconds, zone abbreviation, leap second, or unsupported
    shorthand is refused. UTC conversion must fit Python's datetime range.
    This deliberately bounded parser does not infer a timezone or clock origin.
    """
    if not isinstance(value, str):
        raise ValueError("capture timestamp must be an explicit ISO-8601 string with a timezone")
    match = _CAPTURE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("capture timestamp must use YYYY-MM-DDTHH:MM:SS[.ffffff] with Z or +/-HH:MM")
    fraction = match.group("fraction")
    if fraction is not None and len(fraction) > 6:
        raise ValueError("capture timestamp fractions finer than six digits are unsupported; timestamps are not truncated")
    # datetime accepts over-range minute/second components in some offset forms
    # by carrying them; prohibit that silent normalization in a declared offset.
    if value[-1] != "Z":
        offset_hour, offset_minute = int(value[-5:-3]), int(value[-2:])
        if offset_hour > 23 or offset_minute > 59:
            raise ValueError("capture timezone offset must have hours 00..23 and minutes 00..59")
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("capture timestamp must include an explicit timezone")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError("capture timestamp is invalid or outside the supported UTC datetime range") from exc


def _source(value, name: str) -> tuple[dict[str, str | None], dict[str, datetime | None]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must map nonempty sample IDs to timestamp strings or None")
    originals, parsed = {}, {}
    for sample_id, timestamp in value.items():
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{name} sample IDs must be nonempty strings")
        originals[sample_id] = timestamp
        if timestamp is None:
            parsed[sample_id] = None
        else:
            try:
                parsed[sample_id] = parse_capture_utc(timestamp)
            except ValueError as exc:
                raise ValueError(f"{name} timestamp for sample {sample_id!r}: {exc}") from exc
    return originals, parsed


def compare_capture_clocks(camera: Mapping[str, str | None], gauge: Mapping[str, str | None],
                           reference: Mapping[str, str | None], *, max_skew_seconds) -> dict:
    """Compare three declarations for each exact sample ID under a caller limit.

    The ID union preserves camera insertion order, then previously unseen gauge
    IDs, then previously unseen reference IDs. Missing keys and explicit None
    are retained as missing. Every present timestamp is validated, even if that
    sample is incomplete in another source. A complete sample's maximum pair
    skew is max(UTC timestamps)-min(UTC timestamps); no tolerance is added to the
    finite nonnegative caller-supplied limit. Incomplete samples have no skew or
    within-limit verdict. common_samples counts complete triples.

    The result is JSON-safe and preserves original timestamp strings verbatim.
    Conversion to UTC does not alter, reassign, or fill those declarations.
    """
    if isinstance(max_skew_seconds, bool) or not isinstance(max_skew_seconds, Real):
        raise ValueError("max_skew_seconds must be a finite nonnegative real number, not boolean")
    try:
        limit = float(max_skew_seconds)
    except (ValueError, OverflowError) as exc:
        raise ValueError("max_skew_seconds must be a finite nonnegative real number") from exc
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("max_skew_seconds must be a finite nonnegative real number")
    # Accepted timestamps have integer microseconds. Compare that exact count
    # against the declared numeric limit's decimal spelling: total_seconds()
    # alone loses microseconds for very large date separations. No limit is
    # rounded up to a timestamp tick or enlarged by an implicit tolerance.
    limit_microseconds = Decimal(str(limit)) * 1_000_000
    sources = {name: _source(value, name) for name, value in
               (("camera", camera), ("gauge", gauge), ("reference", reference))}
    sample_ids = list(dict.fromkeys(sample_id for originals, _ in sources.values() for sample_id in originals))
    missing = {name: [] for name in sources}
    samples, exceeded, observed = [], [], []
    for sample_id in sample_ids:
        original_timestamps = {name: originals.get(sample_id) for name, (originals, _) in sources.items()}
        timestamps = {name: parsed.get(sample_id) for name, (_, parsed) in sources.items()}
        for name, timestamp in timestamps.items():
            if timestamp is None:
                missing[name].append(sample_id)
        if all(timestamp is not None for timestamp in timestamps.values()):
            delta = max(timestamps.values()) - min(timestamps.values())
            microseconds = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
            skew = microseconds / 1_000_000
            within = bool(microseconds <= limit_microseconds)
            observed.append(microseconds)
            if not within:
                exceeded.append(sample_id)
        else:
            skew, within = None, None
        samples.append({"sample_id": sample_id, "original_timestamps": original_timestamps,
                        "maximum_skew_seconds": skew, "within_declared_limit": within})
    return {
        "max_skew_seconds": limit,
        "common_samples": len(observed),
        "missing_ids_by_source": missing,
        "maximum_observed_skew_seconds": max(observed) / 1_000_000 if observed else None,
        "exceeded_sample_ids": exceeded,
        "samples": samples,
        "note": ("This checks differences between declared capture timestamps for identical sample IDs, "
                 "not true simultaneity, clock accuracy, timestamp semantics, or timing uncertainty. "
                 "No timestamps are reassigned, copied between sources, rounded to agree, or resampled."),
    }
