# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GaMAC'
copyright = '2025, Sergey Muravyov, Nikita Kulin, Ivan Usov, Olga Muravyova, Simar Muratov'
author = 'Sergey Muravyov, Nikita Kulin, Ivan Usov, Olga Muravyova, Simar Muratov'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

# Mock imports for modules that require special dependencies
autodoc_mock_imports = [
    'cupy',
    'cupy.typing',
    'transformers',
    'umap',
    'pylibraft',
    'pylibraft.config',
    'xgboost'
]

templates_path = ['_templates']
exclude_patterns = []

language = 'ru'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
