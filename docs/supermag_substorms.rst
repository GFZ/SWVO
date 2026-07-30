.. SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
.. SPDX-FileContributor: Simon Mischel
..
.. SPDX-License-Identifier: Apache-2.0

SuperMAG substorm-onset catalogues
==================================

The :class:`swvo.io.substorms.SubstormsSuperMAG` reader downloads and reads
the five substorm-onset catalogues currently distributed through the SuperMAG
Products service. These data are sparse event records rather than a continuous
time series: every row represents an identified onset.

The catalogues implement different scientific definitions. They should be
selected and compared deliberately rather than merged into a single
``"all"`` result.

Quick start
-----------

Register on the `SuperMAG website <https://supermag.jhuapl.edu/>`_ and supply
the username at runtime:

.. code-block:: python

   import os
   from datetime import datetime, timezone
   from pathlib import Path

   from swvo.io.substorms import SubstormsSuperMAG

   reader = SubstormsSuperMAG(
       username=os.environ["SUPERMAG_USERNAME"],
       data_dir=Path("./supermag_data"),
   )

   start = datetime(2001, 1, 1, tzinfo=timezone.utc)
   end = datetime(2001, 1, 2, tzinfo=timezone.utc)

   events = reader.read(start, end, download=True)
   # The default catalogue is "newell".

Use ``catalog`` to select another onset list:

.. code-block:: python

   sophie_onsets = reader.read(
       start,
       end,
       download=True,
       catalog="sophie",
   )
   # "sophie" is normalized to SuperMAG's canonical "forsyth" identifier.

Only one catalogue is returned by each call. This prevents events derived by
different techniques from being combined or de-duplicated without an explicit
scientific decision by the caller.

Available catalogues
--------------------

The registry can be inspected without accessing the network:

.. code-block:: python

   metadata = reader.available_catalogs()

.. list-table::
   :header-rows: 1
   :widths: 12 20 30 18 20

   * - Name
     - Reference
     - Method
     - SuperMAG coverage
     - Update behavior
   * - ``newell``
     - Newell and Gjerloev (2011)
     - Automated onset identification from changes in one-minute SML
     - 1969-current
     - Continuously revised
   * - ``forsyth``
     - Forsyth et al. (2015), SOPHIE
     - SOPHIE expansion-phase onset identification from filtered SML
     - 1969-current
     - Continuously revised
   * - ``ohtani``
     - Ohtani and Gjerloev (2020)
     - Strict SML criteria for high-confidence isolated substorms
     - 1969-current
     - Continuously revised
   * - ``frey``
     - Frey et al. (2004 and 2006)
     - Auroral onsets identified in IMAGE-FUV observations
     - 19 May 2000-31 December 2002
     - Final, with observing gaps
   * - ``liou``
     - Liou (2010)
     - Auroral breakups identified in Polar UVI observations
     - Parts of 1996-2000 and 2007
     - Final, with observing gaps

Canonical names are case-insensitive. Documented aliases include ``sophie``
for ``forsyth``, ``newell_gjerloev`` for ``newell``,
``ohtani_gjerloev`` for ``ohtani``, and ``frey_mende`` for ``frey``.
Spaces and hyphens in aliases are normalized to underscores. Empty,
non-string, and unknown catalogue selections raise a descriptive exception
before any file or network access.

Output schema
-------------

The returned :class:`pandas.DataFrame` has a timezone-aware UTC
``DatetimeIndex`` named ``onset`` and the following columns:

.. list-table::
   :header-rows: 1
   :widths: 16 18 66

   * - Column
     - Unit/type
     - Meaning
   * - ``mlt``
     - hours
     - Magnetic local time associated with the onset. SuperMAG serves the
       Liou values in degrees despite labelling the field ``MLT``; SWVO
       normalizes them to hours using 15 degrees per hour.
   * - ``mlat``
     - degrees
     - Magnetic latitude associated with the onset
   * - ``glon``
     - degrees
     - Geographic longitude associated with the onset
   * - ``glat``
     - degrees
     - Geographic latitude associated with the onset
   * - ``catalog``
     - string
     - Canonical catalogue identifier
   * - ``file_name``
     - path
     - Local source-file provenance

Both interval boundaries are inclusive. A valid interval with no identified
events returns an empty DataFrame with the same columns and a UTC index. It is
not treated as a download failure.

Location interpretation
-----------------------

For the ``newell``, ``forsyth``, and ``ohtani`` lists, SuperMAG states that
the onset location is the location of the station contributing to SML at that
time. It is therefore a station-based location proxy and should not
automatically be interpreted as the physical auroral breakup location.

The ``frey`` and ``liou`` coordinates are instead derived from auroral image
observations. Their interpretation, coverage, and selection effects differ
from the index-derived catalogues.

The original Liou files distributed through SuperMAG store their
``<mlt>`` values in degrees. This is evident from values such as ``337.9`` and
is consistent with the approximately 22.6-hour mean reported by Liou (2010)
after division by 15 degrees per hour. SWVO performs that conversion in the
returned DataFrame while retaining the unmodified self-documented source file
in the cache.

SOPHIE onset list versus complete phase classification
------------------------------------------------------

The SuperMAG ``forsyth`` product exposed by this reader contains onset time
and location records with the common five-field event schema. It does not
contain the complete minute-by-minute SOPHIE growth, expansion, and recovery
phase classification described by Forsyth et al. (2015).

Implementing the complete SOPHIE method would be a separate derived-data
feature requiring its filtering, percentile thresholds, phase corrections,
and enhanced-convection checks to be reproduced and scientifically validated.

Cache layout and revisions
--------------------------

The reader stores one self-documented ASCII response per catalogue and year:

.. code-block:: text

   <SUPERMAG_STREAM_DIR>/
     substorms/
       newell/
         SuperMAG_SUBSTORMS_NEWELL_2001.txt
       forsyth/
         SuperMAG_SUBSTORMS_FORSYTH_2001.txt

Annual files minimize requests for these sparse data. The exact downloaded
response is retained after validation, preserving SuperMAG's data revision,
database generation time, download time, caveats, acknowledgement text, and
references. ``file_name`` connects each returned event to that provenance.

The three index-derived catalogues are revised when SuperMAG's underlying
holdings change. ``download=True`` retrieves missing files but preserves
existing ones for reproducibility. Explicitly refresh a cached interval when
the latest catalogue revision is required:

.. code-block:: python

   reader.download_and_process(
       start,
       end,
       reprocess_files=True,
       catalog="newell",
   )

Every response is parsed and validated before it atomically replaces the
target file. A request, validation, or write failure leaves an existing cache
untouched and removes the temporary file. A corrupt cache produces remediation
guidance when downloads are disabled and is replaced when ``download=True``.

Authentication and privacy
--------------------------

The SuperMAG Products page asks users to log on before downloading. SWVO sends
the registered username as an encoded request parameter. The username is not
stored in a cache, included in logs, or exposed through the reader's ``url``
provenance property. Keep it in a local environment variable and never commit
it to source control.

Reliability and error handling
------------------------------

Zero-byte responses, connection failures, timeouts, HTTP ``429``, and HTTP
``5xx`` responses are retried up to four times with bounded exponential
backoff. A well-formed response containing no event rows is valid and is
cached without retrying.

An explicit SuperMAG ``ERROR`` response, non-retryable HTTP error, malformed
table, non-numeric field, or invalid onset time is reported clearly. Historical
batch downloads continue to later years after exhausting retries for a
transiently failing year and emit a warning listing the failed years. An
on-demand :meth:`~swvo.io.substorms.SubstormsSuperMAG.read` remains strict
when its required download cannot be completed.

Sources, limitations, and acknowledgement
-----------------------------------------

SuperMAG emphasizes that all onset techniques have limitations and that users
must understand the assumptions of the selected list. Before publication,
consult the current:

* `SuperMAG Products overview and catalogue acknowledgements
  <https://supermag.jhuapl.edu/products/?tab=about>`_;
* `derivation descriptions
  <https://supermag.jhuapl.edu/products/?tab=description>`_;
* `download rules of the road
  <https://supermag.jhuapl.edu/products/?tab=download>`_.

SuperMAG states that its data and derived products are subject to fair-use
restrictions and must not be redistributed. It requests the catalogue-specific
acknowledgement and reference and, when an onset list is central to a study,
an offer of co-authorship to the authors of the selected technique.

The principal catalogue references are:

* `Newell and Gjerloev (2011)
  <https://doi.org/10.1029/2011JA016779>`_;
* `Forsyth et al. (2015)
  <https://doi.org/10.1002/2015JA021343>`_;
* `Ohtani and Gjerloev (2020)
  <https://doi.org/10.1029/2020JA027902>`_;
* `Frey et al. (2004)
  <https://doi.org/10.1029/2004JA010607>`_;
* `Liou (2010)
  <https://doi.org/10.1029/2010JA015578>`_.
