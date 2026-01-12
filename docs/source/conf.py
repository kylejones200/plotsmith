"""Sphinx configuration for PlotSmith documentation."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "PlotSmith"
copyright = "2024, Kyle T. Jones"
author = "Kyle T. Jones"
release = "0.0.1"
version = "0.0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_mock_imports = []
autosummary_generate = True

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Type hints
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# HTML theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

# HTML static files
html_static_path = ["_static"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Master document
master_doc = "index"



