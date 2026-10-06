"""Shared fixtures.

Equivalence tests compare the refactored code with the original research tree.
Point ``NTPP_ORIG`` at a checkout of that tree to run them; they are skipped
otherwise.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def ntpp_orig() -> Path:
    root = os.environ.get("NTPP_ORIG")
    if not root or not Path(root).is_dir():
        pytest.skip("NTPP_ORIG is not set; skipping equivalence test against the original tree")
    return Path(root)
