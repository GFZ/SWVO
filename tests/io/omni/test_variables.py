# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from swvo.io.omni.variables import (
    HIGH_RES_DEFAULT_VARIABLES,
    HIGH_RES_VARIABLES,
    LOW_RES_DEFAULT_VARIABLES,
    LOW_RES_VARIABLES,
    resolve_variable_names,
)


@pytest.mark.parametrize(
    "registry,expected_count",
    [(HIGH_RES_VARIABLES, 45), (LOW_RES_VARIABLES, 54)],
)
def test_registry_names_and_aliases_are_unambiguous(registry, expected_count):
    names = [variable.name for variable in registry]
    aliases = [alias for variable in registry for alias in variable.aliases]

    assert len(registry) == expected_count
    assert len(names) == len(set(names))
    assert len(aliases) == len(set(aliases))
    assert set(names).isdisjoint(aliases)
    assert all(variable.description for variable in registry)
    assert all(variable.unit is not None for variable in registry)


def test_high_resolution_registry_ids_and_cadences_are_complete():
    assert [variable.nasa_id for variable in HIGH_RES_VARIABLES] == list(range(4, 49))
    assert sum(1 in variable.cadences for variable in HIGH_RES_VARIABLES) == 42
    assert sum(5 in variable.cadences for variable in HIGH_RES_VARIABLES) == 45
    assert all(set(variable.cadences).issubset({1, 5}) for variable in HIGH_RES_VARIABLES)


@pytest.mark.parametrize(
    "registry,defaults,cadence",
    [
        (HIGH_RES_VARIABLES, HIGH_RES_DEFAULT_VARIABLES, 1),
        (LOW_RES_VARIABLES, LOW_RES_DEFAULT_VARIABLES, None),
    ],
)
def test_default_selection_is_stable_and_returns_a_new_list(registry, defaults, cadence):
    first = resolve_variable_names(registry, None, defaults, cadence)
    second = resolve_variable_names(registry, None, defaults, cadence)

    assert first == list(defaults)
    assert second == list(defaults)
    assert first is not second


def test_all_selection_respects_cadence_and_registry_order():
    one_minute = resolve_variable_names(HIGH_RES_VARIABLES, "all", HIGH_RES_DEFAULT_VARIABLES, 1)
    five_minute = resolve_variable_names(HIGH_RES_VARIABLES, "all", HIGH_RES_DEFAULT_VARIABLES, 5)

    assert one_minute == [variable.name for variable in HIGH_RES_VARIABLES if 1 in variable.cadences]
    assert five_minute == [variable.name for variable in HIGH_RES_VARIABLES]


def test_generator_aliases_and_duplicates_preserve_first_canonical_occurrence():
    requested = (name for name in ["sym_h", "speed", "sym-h", "flow_pressure", "pdyn"])

    result = resolve_variable_names(HIGH_RES_VARIABLES, requested, HIGH_RES_DEFAULT_VARIABLES, 1)

    assert result == ["sym-h", "speed", "pdyn"]


@pytest.mark.parametrize("selection", [[], (), iter(())])
def test_empty_iterables_are_rejected(selection):
    with pytest.raises(ValueError, match="at least one"):
        resolve_variable_names(LOW_RES_VARIABLES, selection, LOW_RES_DEFAULT_VARIABLES)


@pytest.mark.parametrize("selection", [42, object(), ["speed", 3], [None]])
def test_non_string_or_non_iterable_selections_are_rejected_clearly(selection):
    with pytest.raises(ValueError, match="variable name|names must be strings"):
        resolve_variable_names(HIGH_RES_VARIABLES, selection, HIGH_RES_DEFAULT_VARIABLES, 1)


def test_unknown_error_lists_only_cadence_valid_names():
    with pytest.raises(ValueError) as error:
        resolve_variable_names(HIGH_RES_VARIABLES, "unknown", HIGH_RES_DEFAULT_VARIABLES, 1)

    message = str(error.value)
    assert "bavg" in message
    assert "proton_flux_10_mev" not in message


def test_cadence_error_preserves_requested_alias_in_message():
    with pytest.raises(ValueError, match="proton_flux_10_mev"):
        resolve_variable_names(HIGH_RES_VARIABLES, ["speed", "proton_flux_10_mev"], HIGH_RES_DEFAULT_VARIABLES, 1)
