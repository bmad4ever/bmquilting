# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import shutil
import dataclasses
import importlib
from pathlib import Path


# -- Project Root Setup -------------------------------------------------------
# Path is now relative to sphinx_src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = 'bmquilting'
copyright = '2026, BMad4ever'
author = 'BMad4ever'

# -- Import Markdown Docs from ../docs ----------------------------------------
# Keeps .md files in /docs for repo-level reading while Sphinx processes them
# from a mirrored subdirectory inside sphinx_src.

current_dir = Path(__file__).parent
docs_dir    = current_dir.parent / 'docs'
target_dir  = current_dir / 'docs_md'

def setup_md_docs():
    if target_dir.exists():
        if target_dir.is_symlink():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
    try:
        os.symlink(docs_dir, target_dir, target_is_directory=True)
        print(f"INFO: Created symlink {docs_dir} → {target_dir}")
    except (OSError, NotImplementedError) as e:
        print(f"WARNING: Symlink failed ({e}). Falling back to file copy.")
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in docs_dir.glob('*.md'):
            shutil.copy(f, target_dir)
        if (docs_dir / 'imgs').exists():
            shutil.copytree(docs_dir / 'imgs', target_dir / 'imgs', dirs_exist_ok=True)

def cleanup_md_docs(app, exception):
    """Remove the temporary docs_md directory after the build finishes."""
    if target_dir.exists():
        if target_dir.is_symlink():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
        print(f"INFO: Cleaned up {target_dir}")

setup_md_docs()

# -- HTML FIXES & CUSTOM -------------------------------------------------------

myst_heading_anchors = 6     # generate anchors for all heading levels (h1–h6)


# -- General configuration ----------------------------------------------------

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_github_alerts",
    "sphinx.ext.githubpages",
]

myst_enable_extensions = [
    "html_image",
    "dollarmath",
    "amsmath",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autodoc_mock_imports = ["numpy", "pyastar2d", "cv2", "sklearn", "joblib"]

# -- Autodoc settings ---------------------------------------------------------

autodoc_default_options = {
    'members':          True,
    'undoc-members':    True,
    'show-inheritance': True,
    'member-order':     'bysource',
    # '__init__' is intentionally omitted: for dataclasses its parameters are
    # identical to the :ivar: fields in the class docstring, so documenting it
    # separately would always produce a duplicate "Parameters" section.
}

# 'signature' keeps types visible in the function/class signature line but does
# NOT inject :param: entries into the description.  That injection is what caused
# Napoleon to generate the spurious "Parameters" section alongside "Variables".
autodoc_typehints = 'signature'

# Only use the class-level docstring (not __init__'s) for class documentation.
autoclass_content = 'class'


# -- Member filtering ---------------------------------------------------------

def skip_member(app, what, name, obj, skip, options):
    # Always skip private / dunder names.
    if name.startswith('_'):
        return True

    current_module = (
        app.env.temp_data.get('autodoc:module')
        or app.env.ref_context.get('py:module')
    )
    current_class = app.env.temp_data.get('autodoc:class')

    # ------------------------------------------------------------------ #
    # PATH A — we are inside a class                                       #
    # ------------------------------------------------------------------ #
    # Class context and module context are kept strictly separate so that  #
    # a failed module import can never silently drop class members.        #
    if current_class:
        # Dataclass fields are documented via :ivar: in the class docstring;
        # skip the redundant per-attribute autodoc entries.
        if what == 'attribute' and current_module:
            try:
                mod = sys.modules.get(current_module) or importlib.import_module(current_module)
                cls = getattr(mod, current_class, None)
                if cls and dataclasses.is_dataclass(cls):
                    if name in {f.name for f in dataclasses.fields(cls)}:
                        return True
            except Exception:
                pass

        # Every other class member (methods, classmethods, properties, …)
        # is always shown — __all__ governs module-level visibility only.
        return False

    # ------------------------------------------------------------------ #
    # PATH B — we are at module level                                      #
    # ------------------------------------------------------------------ #
    # If the module declares __all__, only expose what it explicitly       #
    # exports.  Fall back to the _internal guard when __all__ is absent.  #
    if current_module:
        try:
            mod = sys.modules.get(current_module) or importlib.import_module(current_module)
            all_members = getattr(mod, '__all__', None)
            if all_members is not None:
                return name not in all_members
        except Exception:
            pass

    # No __all__ available: suppress anything that originates from an
    # _internal submodule and was not explicitly re-exported.
    if '_internal' in getattr(obj, '__module__', ''):
        return True

    return skip


def setup(app):
    app.connect('autodoc-skip-member', skip_member)
    app.connect('build-finished', cleanup_md_docs)


templates_path    = ['_templates']
exclude_patterns  = ['docs_md/README.md']


# -- HTML output --------------------------------------------------------------

html_theme       = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name":    False,
    "navigation_with_keys": True,
}
