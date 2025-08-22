# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys


sys.path.insert(0, os.path.abspath(".."))

import swvo

project = "swvo"
copyright = "2024, GFZ"
author = "Bernhard Haas, Sahil Jhawar"

version = swvo.__version__
release = version

master_doc = "index"
extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.imgconverter",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
autosummary_generate = True

autodoc_default_options = {
    "members": ("member-order,inherited-members, show-inheritance"),
    "member-order": "bysource",
    # "private-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "exclude-members": "__weakref__, __dict__, __module__",
}

SWVO_CHANGELOG_TOKEN = os.getenv("SWVO_CHANGELOG_TOKEN")
plot_html_show_source_link = False
plot_html_show_formats = False
plot_working_directory = os.path.abspath("..")
plot_rcparams = {"figure.figsize": (10, 5)}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": (
        "https://docs.scipy.org/doc/numpy/",
        (None, "http://data.astropy.org/intersphinx/numpy.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        (None, "http://data.astropy.org/intersphinx/scipy.inv"),
    ),
    "pandas": (
        "https://pandas.pydata.org/pandas-docs/stable/",
        (None, "http://data.astropy.org/intersphinx/pandas.inv"),
    ),
}

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css"]


def setup(app):
    app.add_css_file("custom.css")
