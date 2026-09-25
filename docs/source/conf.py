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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "thotbook-AmentI Documentation"
html_short_title = "thotbook-AmentI"
html_logo = "_static/logo_thot_transparent.png"
html_favicon = "_static/logo_thot_transparent.png"

# Announcement bar (top panel) - Furo supports this via html_theme_options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#c2410c",
        "color-brand-content": "#c2410c",
        "color-api-pre-background": "#fef3e2",
        "color-background-primary": "#fefefe",
        "color-background-secondary": "#fef3e2",
        "color-foreground-primary": "#2d2d3a",
        "color-foreground-secondary": "#6b6b7a",
        "color-foreground-muted": "#9a9ab0",
        "color-border": "#fde68a",
        "color-announcement-background": "#c2410c",
        "color-announcement-text": "#ffffff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fb923c",
        "color-brand-content": "#fb923c",
        "color-api-pre-background": "#1a1a2e",
        "color-background-primary": "#141424",
        "color-background-secondary": "#1a1a2e",
        "color-foreground-primary": "#e8e8f0",
        "color-foreground-secondary": "#a0a0b8",
        "color-foreground-muted": "#707088",
        "color-border": "#3a250a",
        "color-announcement-background": "#fb923c",
        "color-announcement-text": "#ffffff",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/ThotDjehuty/hfthot-llm",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "announcement": "🚀 <strong>thotbook-AmentI v0.2.0</strong> — llm-auto, llm-ingest &amp; llm-salviers • <a href=\"https://hfthot-lab.eu/thotbook-amenti.html\">HFThot Research Lab</a>",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ThotDjehuty/hfthot-llm",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">
                    <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
