# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

"""Metadata for the substorm-onset catalogues distributed by SuperMAG."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SuperMAGSubstormCatalog:
    """Description of one SuperMAG substorm-onset catalogue."""

    name: str
    label: str
    aliases: tuple[str, ...]
    method: str
    coverage: str
    continuously_updated: bool
    location_description: str
    reference: str


SUPERMAG_SUBSTORM_CATALOGS: tuple[SuperMAGSubstormCatalog, ...] = (
    SuperMAGSubstormCatalog(
        name="newell",
        label="Newell and Gjerloev (2011)",
        aliases=("newell_gjerloev",),
        method="Automated onset identification from one-minute SML changes",
        coverage="1969-current",
        continuously_updated=True,
        location_description="Location of the station contributing to SML at onset",
        reference="https://doi.org/10.1029/2011JA016779",
    ),
    SuperMAGSubstormCatalog(
        name="forsyth",
        label="Forsyth et al. (2015), SOPHIE",
        aliases=("sophie",),
        method="SOPHIE expansion-phase onset identification from filtered SML",
        coverage="1969-current",
        continuously_updated=True,
        location_description="Location of the station contributing to SML at onset",
        reference="https://doi.org/10.1002/2015JA021343",
    ),
    SuperMAGSubstormCatalog(
        name="ohtani",
        label="Ohtani and Gjerloev (2020)",
        aliases=("ohtani_gjerloev",),
        method="High-confidence isolated-substorm identification from SML",
        coverage="1969-current",
        continuously_updated=True,
        location_description="Location of the station contributing to SML at onset",
        reference="https://doi.org/10.1029/2020JA027902",
    ),
    SuperMAGSubstormCatalog(
        name="frey",
        label="Frey et al. (2004 and 2006)",
        aliases=("frey_mende",),
        method="Auroral-onset identification from IMAGE-FUV observations",
        coverage="2000-05-19 to 2002-12-31",
        continuously_updated=False,
        location_description="Auroral brightening location derived from IMAGE-FUV observations",
        reference="https://doi.org/10.1029/2004JA010607",
    ),
    SuperMAGSubstormCatalog(
        name="liou",
        label="Liou (2010)",
        aliases=(),
        method="Auroral-breakup identification from Polar UVI observations",
        coverage="Parts of 1996-2000 and 2007",
        continuously_updated=False,
        location_description="Auroral breakup location derived from Polar UVI observations",
        reference="https://doi.org/10.1029/2010JA015578",
    ),
)


def available_catalogs() -> pd.DataFrame:
    """Return metadata for every SuperMAG substorm-onset catalogue.

    Returns
    -------
    pandas.DataFrame
        A new table in registry order with canonical names, labels, aliases,
        methods, coverage, update behavior, location interpretation, and
        references.
    """

    return pd.DataFrame(
        [
            {
                "name": catalog.name,
                "label": catalog.label,
                "aliases": catalog.aliases,
                "method": catalog.method,
                "coverage": catalog.coverage,
                "continuously_updated": catalog.continuously_updated,
                "location_description": catalog.location_description,
                "reference": catalog.reference,
            }
            for catalog in SUPERMAG_SUBSTORM_CATALOGS
        ]
    )
