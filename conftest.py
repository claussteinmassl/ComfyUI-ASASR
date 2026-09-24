"""Pytest configuration.

The repository root doubles as the ComfyUI package directory, so it has an
__init__.py with a relative import that only works when ComfyUI imports the
repo as a package. If pytest treats this directory as a Package (any
directory containing __init__.py), Package.setup() unconditionally imports
that __init__.py to look for setUpModule/tearDownModule hooks -- regardless
of collect_ignore -- which raises the same relative-import error outside a
package context. Register a plugin that forces the repository root to
collect as a plain directory instead; it must be a genuinely global plugin
(not an auto-discovered conftest hookimpl) because the decision for how to
collect a given directory is made using the *parent* directory's hook scope,
which does not include this file's own hooks. Subdirectories such as tests/
are unaffected and keep normal Package handling.
"""
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).parent


class _RootAsPlainDir:
    def pytest_collect_directory(self, path, parent):
        if path == _ROOT:
            return pytest.Dir.from_parent(parent, path=path)
        return None


def pytest_configure(config):
    config.pluginmanager.register(_RootAsPlainDir(), "asasr_root_as_plain_dir")


import os
import sys

_COMFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dev", "ComfyUI")
if os.path.isdir(_COMFY_DIR) and _COMFY_DIR not in sys.path:
    sys.path.insert(0, _COMFY_DIR)
