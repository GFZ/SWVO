<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences

SPDX-License-Identifier: Apache-2.0
-->

# Reviewer report: complete OMNI variable support

## Purpose

The existing OMNI implementations intentionally reduced NASA data to nine
high-resolution columns and three hourly columns. This change makes every
published field accessible while preserving those defaults and the focused
reader APIs.

## Design decisions

- A typed registry is the single source of truth for canonical names, NASA
  request IDs, units, descriptions, fill values, cadence restrictions, and
  aliases.
- `variables=None` preserves existing output; `"all"` and explicit selections
  opt into the expanded interface.
- Processed cache filenames are unchanged. Complete files can serve any later
  selection, while partial legacy files remain usable for legacy reads.
- Schema upgrades are atomic and occur only when selected fields are missing
  and downloading is authorized.
- The hourly parser detects both NASA layouts: historic 55-word records and the
  current 57-word record with Lyman-alpha and proton quasi-invariant values.
- The dedicated Kp, Dst, F10.7, solar-wind, and SYM-H readers keep their
  established return schemas.

## Reviewer entry points

- `swvo/io/omni/variables.py`: authoritative metadata and selection rules.
- `swvo/io/omni/omni_high_res.py`: complete OMNIWeb requests, parsing, and
  monthly cache upgrades.
- `swvo/io/omni/omni_low_res.py`: version-tolerant hourly parsing and yearly
  cache upgrades.
- `docs/omni_variables.rst`: public API, complete variable tables, examples,
  cache behavior, and source acknowledgements.

## Compatibility

- Existing method calls remain valid because `variables` is appended as an
  optional argument.
- Default column names and ordering are unchanged.
- Existing processed files do not require an eager migration.
- Path handling uses `pathlib` and `tempfile`; no platform-specific commands or
  separators are introduced.

## Verification strategy

Deterministic tests cover both cadences, all NASA request IDs, response order,
fill conversion, aliases, validation, both hourly record widths, partial-cache
upgrades, UTC indices, provenance, atomic writes, and all OMNI-derived readers.
The full variable metadata tables are also checked for their expected 42, 45,
and 54 row counts.

The expanded edge-case matrix additionally covers registry uniqueness,
generators and invalid selection types, equal/reversed/mixed-timezone ranges,
mission-start clipping, month/year neighbor selection, malformed HTML and
corrected-range retry loops, absent headers, incorrect and mixed record widths,
nonnumeric values, invalid time words, corrupt-cache recovery and failed
recovery, fill-boundary behavior, Kp thirds, OS temporary-directory cleanup,
atomic replacement, and parallel worker failure propagation.

Large mandatory live downloads were removed from unit tests. An opt-in one-day
OMNIWeb request remains as a pre-delivery smoke check; run it with
`SWVO_RUN_LIVE_TESTS=1` so external service availability cannot make the normal
test suite nondeterministic.

## Local verification record

Checks were run on 2026-07-17 on macOS with Python 3.12.12:

- Ruff lint and formatting: passed for the complete repository.
- Deterministic OMNI and derived-reader tests: 118 passed, 2 opt-in live
  checks skipped.
- Opt-in NASA OMNIWeb smoke checks: 2 passed, covering all 42 one-minute
  fields and all 45 five-minute fields for one day.
- Production-size NASA check: a complete one-minute month produced 44,640
  rows and 42 fields, while a complete five-minute month produced 8,928 rows
  and 45 fields (including all three cadence-specific proton-flux fields).
  Both had UTC timestamps and left no residual temporary file.
- Live hourly check: NASA's 2024 55-word file produced 8,784 rows and all 54
  output fields; the two unavailable historic fields were `NaN`.
- Full repository test run: 586 passed and 12 skipped. The only failure was the
  unrelated live SuperMAG download test; repeat runs alternated malformed JSON
  between its two requested days. All 120 OMNI and OMNI-derived tests passed.
- A second complete repository run, deselecting only that exact external
  SuperMAG request, finished with 586 passed, 12 skipped, 1 deselected, and no
  failures. This separates the reproducible local result from the availability
  of an unrelated third-party endpoint.
- Ty: no new diagnostics. The same two RBMDataSet diagnostics occur on an
  untouched archive of `upstream/main` and on this branch.
- Sphinx: the standalone OMNI guide passed with warnings treated as errors.
  An integrated build with the repository configuration succeeded after
  excluding the existing notebook/changelog/API inputs; its remaining warnings
  are pre-existing offline intersphinx and `_static` configuration warnings.

GitHub Actions results will be recorded after the verified branch is pushed to
the fork. No pull request or upstream notification has been created.

## Known boundaries

- Integral proton fluxes above 10, 30, and 60 MeV are unavailable at one-minute
  cadence and are rejected before download.
- Historic hourly source records have no Lyman-alpha or proton quasi-invariant
  words; the corresponding output columns are `NaN`.
- NASA can revise the upstream format. Future fields should be added to the
  registry and record-width tests together.
