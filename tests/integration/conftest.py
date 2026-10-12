import pytest


@pytest.fixture(autouse=True)
def tiny_cpu_env(monkeypatch, tmp_path):
    """Every integration test runs the real ``tiny`` model on CPU with a throwaway cache.

    Set here, not on the command line, so a bare ``pytest -m integration`` (the end-to-end
    check, CI) runs exactly what the per-task Verify lines run. Subprocesses inherit it.
    """
    monkeypatch.setenv("QUOTABLE_MODEL", "tiny")
    monkeypatch.setenv("QUOTABLE_DEVICE", "cpu")
    monkeypatch.setenv("QUOTABLE_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("QUOTABLE_OUTPUT_DIR", str(tmp_path / "out"))
