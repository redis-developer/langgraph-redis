# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from langgraph.checkpoint.redis.version import __version__

# -- Project information -----------------------------------------------------

project = "langgraph-checkpoint-redis"
copyright = "2024, Redis Inc"
author = "Redis Applied AI"
version = __version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
    "_extension.gallery_directive",
    "myst_nb",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

pygments_style = "friendly"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = "langgraph-checkpoint-redis"
html_logo = "_static/Redis_Favicon_32x32_Red.png"
html_favicon = "_static/Redis_Favicon_32x32_Red.png"

html_context = {
    "github_user": "redis-developer",
    "github_repo": "langgraph-redis",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

html_theme_options = {
    "repository_url": "https://github.com/redis-developer/langgraph-redis",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "navigation_depth": 4,
    "show_toc_level": 3,
    "home_page_in_toc": True,
    "logo": {
        "text": "langgraph-checkpoint-redis",
        "image_light": "_static/Redis_Favicon_32x32_Red.png",
        "image_dark": "_static/Redis_Favicon_32x32_Red.png",
    },
}

autoclass_content = "both"
add_module_names = False

nb_execution_mode = "off"

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# -- favicon options ---------------------------------------------------------

favicons = [
    "Redis_Favicon_32x32_Red.png",
    "Redis_Favicon_16x16_Red.png",
    "Redis_Favicon_144x144_Red.png",
]

# -- Sidebar with version switcher ------------------------------------------
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
        "versioning.html",
    ],
}
