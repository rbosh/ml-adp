# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#sys.path.insert(0, os.path.abspath('../src/ml_adp'))
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'ml-adp'
copyright = '2021, Ruben Wiedemann'
author = 'Ruben Wiedemann'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

add_module_names = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    #'autoclasstoc',
    'sphinx.ext.napoleon',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'nbsphinx'
]

mathjax3_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}
autosummary_generate = True

add_module_names = False

autodoc_default_options = {
    'members': True,
    'special-members': True,
    'private-members': False,
    'inherited-members': False,
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'add_module_names': False,
    'autodoc_unqualified_typehints': True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Custom Options -----------------------------------------------------------
