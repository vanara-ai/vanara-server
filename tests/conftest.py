"""Shared pytest fixtures and test environment setup.

This runs before any test module is imported. We neutralize two sources
of env pollution so tests see a clean, predictable environment:

1. Direct env vars from the shell (``SUPABASE_URL``, ``SUPABASE_KEY``).
2. ``app.env.setup_environment()`` calling ``load_dotenv(override=True)``,
   which would otherwise re-populate those vars from the project root
   ``.env`` file during ``app.main`` import.

We monkey-patch ``dotenv.load_dotenv`` to a no-op at the module level so
that any ``from dotenv import load_dotenv`` performed *after* this runs
picks up our stub. Tests needing persistence-on mode can set env vars
explicitly and import ``app.main`` under a fresh ``sys.modules``.
"""

import os
import sys

import dotenv


def _noop_load_dotenv(*args, **kwargs):
    return False


# Purge Supabase vars inherited from the shell / CI.
for _key in ("SUPABASE_URL", "SUPABASE_KEY"):
    os.environ.pop(_key, None)

# Neutralize dotenv so setup_environment() cannot re-populate from .env.
dotenv.load_dotenv = _noop_load_dotenv

# If app.* was already imported (e.g. by a prior pytest run in the same
# interpreter), drop it so the next import observes our patched dotenv.
for _mod in list(sys.modules):
    if _mod == "app" or _mod.startswith("app."):
        del sys.modules[_mod]
