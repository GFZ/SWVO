.. SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
.. SPDX-FileContributor: Simon Mischel
..
.. SPDX-License-Identifier: Apache-2.0

SuperMAG electrojet indices
===========================

The :class:`swvo.io.sme.SMESuperMAG` reader provides the standard SuperMAG
auroral electrojet indices at one-minute cadence. It handles authenticated
requests, UTC conversion, fill-value masking, daily local caching, and source
file provenance. The existing class name and import path are retained for
backward compatibility.

Available variables
-------------------

All three variables are reported in nanotesla (nT):

.. list-table::
   :header-rows: 1
   :widths: 14 18 68

   * - Name
     - SuperMAG field
     - Description
   * - ``sme``
     - ``SME``
     - SuperMAG electrojet range, defined as ``SMU - SML``. This remains the
       default output.
   * - ``sml``
     - ``SML``
     - Lower envelope of the baseline-corrected magnetic northward component
       across the contributing auroral stations.
   * - ``smu``
     - ``SMU``
     - Upper envelope of the baseline-corrected magnetic northward component
       across the contributing auroral stations.

SuperMAG fill values greater than or equal to ``999998`` are represented as
``NaN`` for every variable.

Authentication and setup
------------------------

Register for data access on the `SuperMAG website
<https://supermag.jhuapl.edu/>`_. Supply the registered username at runtime;
it is not stored by SWVO. The cache directory may be passed explicitly or set
through ``SUPERMAG_STREAM_DIR``:

.. code-block:: python

   import os
   from pathlib import Path

   from swvo.io.sme import SMESuperMAG

   reader = SMESuperMAG(
       username=os.environ["SUPERMAG_USERNAME"],
       data_dir=Path("./supermag_data"),
   )

The username is sent as an encoded request parameter and is intentionally
excluded from download logs. Keep credentials in local environment variables;
do not commit them to source control.

Selecting variables
-------------------

``variables=None`` preserves the original schema and column order:

.. code-block:: python

   default = reader.read(start, end, download=True)
   # columns: sme, file_name

Use ``"all"`` for the complete electrojet trio:

.. code-block:: python

   complete = reader.read(start, end, download=True, variables="all")
   # columns: sme, sml, smu, file_name

A single name or iterable returns only those variables. Caller order is
preserved and duplicate names are removed after their first occurrence:

.. code-block:: python

   sml = reader.read(start, end, variables="sml")
   envelopes = reader.read(start, end, variables=["smu", "sml", "smu"])
   # envelope columns: smu, sml, file_name

Valid canonical names are ``sme``, ``sml``, and ``smu``. Empty selections,
non-string entries, and unknown names raise a descriptive exception before
files or the network are accessed.

Cache layout and compatibility
------------------------------

Processed data use the established daily file name
``SuperMAG_SME_YYYYMMDD.csv``. Newly downloaded files contain ``timestamp``,
``sme``, ``sml``, and ``smu``. Reads add ``file_name`` provenance for rows
that contain at least one requested value.

Existing SME-only files remain valid for a default read. When SML or SMU is
requested from such a file:

* With ``download=False``, the reader leaves the cache untouched and raises an
  error naming the absent fields and explaining how to upgrade it.
* With ``download=True``, the reader downloads all three fields and atomically
  replaces that daily file. A failed request, response parse, or write leaves
  the existing cache unchanged and removes the temporary file.

A corrupt cache follows the same policy: it produces remediation guidance when
downloads are disabled and is atomically replaced when downloads are enabled.
Complete existing caches are never downloaded again unless
``download_and_process(..., reprocess_files=True)`` is requested.

Time coverage and errors
------------------------

The returned index is timezone-aware UTC and follows the reader's established
inclusive one-minute interval behavior. Exact historical availability and
station participation are controlled by SuperMAG and can vary over time.
Source fill values are returned as ``NaN``.

Permanent HTTP and SuperMAG ``ERROR`` responses, malformed JSON, missing
response fields, and invalid timestamps are reported to the caller. The
transient conditions described below are retried first. Missing local files
still produce the established warning when ``download=False``.

Transient download failures
---------------------------

SuperMAG can occasionally return an empty HTTP response for a valid historical
day and then succeed when the same request is repeated. To make long downloads
resilient, SWVO treats the following conditions as transient:

* an empty response body or an empty JSON record array;
* a request timeout or connection failure;
* HTTP status ``429`` or a ``5xx`` server response.

Each affected day is attempted at most four times, with bounded exponential
backoff of one, two, and four seconds between attempts. Permanent failures,
including an explicit SuperMAG ``ERROR`` response such as an invalid username
and other HTTP ``4xx`` responses, are not retried.

The failure policy depends on how the download is invoked:

* :meth:`~swvo.io.sme.SMESuperMAG.download_and_process` continues with later
  days after all attempts for a transiently failing day are exhausted. At the
  end it emits a warning listing every failed date and recommends re-running
  the same range.
* :meth:`~swvo.io.sme.SMESuperMAG.read` with ``download=True`` remains strict.
  If the day needed for that read still fails after all attempts, the final
  exception is raised to the caller.

No failed attempt replaces an existing cache. Temporary files are removed
before retrying, and a later successful attempt is installed atomically.

Sources and acknowledgement
---------------------------

Data are retrieved from the `SuperMAG indices service
<https://supermag.jhuapl.edu/indices/>`_. Publications using these data should
follow SuperMAG's current `acknowledgement guidance
<https://supermag.jhuapl.edu/info/?page=acknowledgement>`_ and cite the relevant
SuperMAG collaborators and index literature. The SME/SMU/SML construction is
described by `Newell and Gjerloev (2011)
<https://doi.org/10.1029/2011JA016936>`_.
