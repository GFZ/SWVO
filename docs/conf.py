# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from docutils import nodes
from sphinx import addnodes
from sphinx.domains.changeset import VersionChange, versionlabel_classes
from sphinx.locale import _

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
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.imgconverter",
    "sphinx_github_changelog",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

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
nbsphinx_execute = "auto"

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


class DeprecatedNoVersion(VersionChange):
    """Like the built-in ``deprecated::`` directive, but the version argument is optional.

    Renders "Deprecated: ..." instead of "Deprecated since version ...:" when no
    version is given, for cases where the deprecation isn't tied to a specific
    version we want to advertise.
    """

    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = True

    def run(self):
        name = "deprecated"
        node = addnodes.versionmodified()
        node.document = self.state.document
        self.set_source_info(node)
        node["type"] = name

        version = self.arguments[0] if self.arguments else ""
        node["version"] = version
        text = _("Deprecated since version %s") % version if version else _("Deprecated")

        messages = []
        if len(self.arguments) == 2:
            inodes, messages = self.parse_inline(self.arguments[1], lineno=self.lineno + 1)
            para = nodes.paragraph(self.arguments[1], "", *inodes, translatable=False)
            self.set_source_info(para)
            node.append(para)
        if self.content:
            node += self.parse_content_to_nodes()

        classes = ["versionmodified", versionlabel_classes[name]]
        if len(node) > 0 and isinstance(node[0], nodes.paragraph):
            if node[0].rawsource:
                content = nodes.inline(node[0].rawsource, translatable=True)
                content.source = node[0].source
                content.line = node[0].line
                content += node[0].children
                node[0].replace_self(nodes.paragraph("", "", content, translatable=False))
            para = node[0]
            para.insert(0, nodes.inline("", "%s: " % text, classes=classes))
        elif len(node) > 0:
            para = nodes.paragraph("", "", nodes.inline("", "%s: " % text, classes=classes), translatable=False)
            node.insert(0, para)
        else:
            para = nodes.paragraph("", "", nodes.inline("", "%s." % text, classes=classes), translatable=False)
            node.append(para)

        self.env.domains.changeset_domain.note_changeset(node)

        return [node, *messages]


def setup(app):
    app.add_css_file("custom.css")
    app.add_directive("deprecated", DeprecatedNoVersion, override=True)
