# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("../../"))

from olmo_core.version import VERSION, VERSION_SHORT  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Olmo-core"
copyright = f"{datetime.today().year}, Allen Institute for Artificial Intelligence"
author = "Allen Institute for Artificial Intelligence"
version = VERSION_SHORT
release = VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_inline_tabs",
]

# Tell myst-parser to assign header anchors for h1-h3.
myst_heading_anchors = 3

suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "docker": ("https://docker-py.readthedocs.io/en/stable/", None),
    "requests": ("https://requests.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "safetensors": ("https://huggingface.co/docs/safetensors/main/en/", None),
    #  "beaker": ("https://beaker-py.readthedocs.io/en/latest/", None),
}

# By default, sort documented members by type within classes and modules.
autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": True}

# Include default values when documenting parameter types.
typehints_defaults = "comma"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_title = f"Olmo-core v{VERSION}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "light_logo": "olmo-full-color.png",
    "dark_logo": "olmo-full-color.png",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/allenai/Olmo-core",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
}

# -- Hack to get rid of stupid warnings from sphinx_autodoc_typehints --------


class ShutupSphinxAutodocTypehintsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if "Cannot resolve forward reference" in record.msg:
            return False
        if "Failed guarded type import" in record.msg:
            return False
        return True


logging.getLogger("sphinx.sphinx_autodoc_typehints").addFilter(ShutupSphinxAutodocTypehintsFilter())


def autodoc_skip_member(app, what, name, obj, skip, options):
    import inspect

    del app, name, options

    module = inspect.getmodule(obj)
    module_name = None if module is None else module.__name__

    if what == "class" and module_name is not None:
        # Skip documenting callback subclass methods.
        if (
            module_name.startswith("olmo_core.train.callbacks.")
            and module_name != "olmo_core.train.callbacks.callback"
            and (inspect.isfunction(obj) or inspect.ismethod(obj))
        ):
            return True

        # Skip documenting train module subclass methods.
        if (
            module_name.startswith("olmo_core.train.train_module.")
            and module_name != "olmo_core.train.train_module.train_module"
            and (inspect.isfunction(obj) or inspect.ismethod(obj) or isinstance(obj, property))
        ):
            return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
