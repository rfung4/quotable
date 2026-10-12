from importlib.metadata import version

import quotable


def test_version_matches_installed_metadata():
    assert quotable.__version__ == version("quotable")
