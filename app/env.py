"""Environment setup for local development.

In deployed environments, vars come from the platform (Render, Vercel, etc.)
and this module is a no-op beyond loading .env for convenience.
"""

import os
import sys

from dotenv import load_dotenv


def setup_environment():
    load_dotenv(override=True)

    if sys.platform == "darwin":
        existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        homebrew_lib = "/opt/homebrew/lib"
        if homebrew_lib not in existing.split(":"):
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{existing}:{homebrew_lib}" if existing else homebrew_lib
