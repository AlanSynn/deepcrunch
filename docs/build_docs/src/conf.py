# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../deepcrunch/"))
import version as ver

version = ver.__version__
release = version

with open("version.txt", "w") as f:
    f.write(version)

repo_url = "https://github.com/AlanSynn/deepcrunch/blob/v{}".format(version)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DeepCrunch"
copyright = "2023, Intel® Neural Compressor, Intel"
author = "Alan Synn"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx_md",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
]

autoapi_dirs = ["../../deepcrunch/"]
autoapi_root = "autoapi"
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autosummary_generate = True
autoapi_options = ["members", "show-module-summary"]
autoapi_ignore = []

templates_path = ["_templates"]

source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata"

html_static_path = ["_static"]


def skip_util_classes(app, what, name, obj, skip, options):
    if what == "property" or what == "method":
        skip = True
    return skip


def setup(app):
    app.add_css_file("custom.css")
    app.connect("autoapi-skip-member", skip_util_classes)


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "{}/{}.py".format(repo_url, filename)
