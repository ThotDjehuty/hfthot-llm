# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../../python"))

# -- Project information -----------------------------------------------------
project = "thotbook-AmentI"
copyright = "2026, HFThot Research Lab"
author = "HFThot Research Lab"
release = "0.2.0"
version = "0.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "thotbook-AmentI Documentation"
html_short_title = "thotbook-AmentI"
