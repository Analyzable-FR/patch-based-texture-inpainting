import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'patch-based-inpainting'
copyright = '2022, Analyzable'
author = 'Analyzable'
release = '0.0.0'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
