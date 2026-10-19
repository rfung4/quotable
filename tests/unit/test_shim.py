import os
import stat
import subprocess
import sys
import tempfile

import pytest

from quotable import shim


@pytest.fixture
def code(tmp_path, monkeypatch):
    """A fake package dir and site-packages dir, so mtimes can be moved."""
    package = tmp_path / "pkg"
    site = tmp_path / "site"
    package.mkdir()
    site.mkdir()
    (package / "server.py").write_text("")
    (package / "notes.txt").write_text("")
    monkeypatch.setattr(shim, "PACKAGE_DIR", package)
    monkeypatch.setattr(shim, "_site_packages", lambda: str(site))
    return package, site


def _bump(path, seconds=10):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def test_key_is_16_hex_and_stable(code):
    key = shim.socket_key({})
    assert len(key) == 16
    int(key, 16)
    assert shim.socket_key({}) == key


@pytest.mark.parametrize(
    "name", ["QUOTABLE_MODEL", "QUOTABLE_DEVICE", "QUOTABLE_MODEL_TTL", "HF_TOKEN", "XDG_DATA_HOME"]
)
def test_key_changes_with_settings(code, name):
    assert shim.socket_key({name: "a"}) != shim.socket_key({name: "b"})


@pytest.mark.parametrize("name", ["QUOTABLE_DAEMON", "PATH", "XDG_RUNTIME_DIR"])
def test_key_ignores_unrelated_env(code, name):
    assert shim.socket_key({name: "a"}) == shim.socket_key({})


def test_key_ignores_env_order(code):
    a = {"QUOTABLE_MODEL": "tiny", "QUOTABLE_DEVICE": "cpu"}
    b = {"QUOTABLE_DEVICE": "cpu", "QUOTABLE_MODEL": "tiny"}
    assert shim.socket_key(a) == shim.socket_key(b)


def test_key_changes_when_source_is_touched(code):
    package, _ = code
    before = shim.socket_key({})
    _bump(package / "server.py")
    assert shim.socket_key({}) != before


def test_key_ignores_non_python_files(code):
    package, _ = code
    before = shim.socket_key({})
    _bump(package / "notes.txt")
    assert shim.socket_key({}) == before


def test_key_changes_when_dependencies_change(code):
    _, site = code
    before = shim.socket_key({})
    _bump(site)
    assert shim.socket_key({}) != before


def test_key_changes_with_version_and_prefix(code, monkeypatch):
    before = shim.socket_key({})
    monkeypatch.setattr(shim, "_version", lambda: "9.9.9")
    after_version = shim.socket_key({})
    monkeypatch.setattr(shim.sys, "prefix", "/elsewhere")
    assert len({before, after_version, shim.socket_key({})}) == 3


def test_key_on_the_real_install():
    key = shim.socket_key({})
    assert len(key) == 16 and key == shim.socket_key({})


def test_runtime_dir_under_xdg(tmp_path):
    path = shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})
    assert path == tmp_path / "quotable"
    assert stat.S_IMODE(path.lstat().st_mode) == 0o700


def test_runtime_dir_falls_back_to_tempdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = shim.runtime_dir({})
    assert path == tmp_path / f"quotable-{os.getuid()}"
    assert stat.S_IMODE(path.lstat().st_mode) == 0o700


def test_runtime_dir_is_reused(tmp_path):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    assert shim.runtime_dir(env) == shim.runtime_dir(env)


def test_runtime_dir_refuses_a_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "quotable").symlink_to(real)
    with pytest.raises(shim.ShimError, match="not a directory"):
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


def test_runtime_dir_refuses_a_file(tmp_path):
    (tmp_path / "quotable").write_text("")
    with pytest.raises(shim.ShimError, match="not a directory"):
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o770])
def test_runtime_dir_refuses_shared_access(tmp_path, mode):
    path = tmp_path / "quotable"
    path.mkdir()
    path.chmod(mode)
    with pytest.raises(shim.ShimError, match="accessible to other users") as err:
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})
    assert "chmod 700" in err.value.fix


def test_runtime_dir_refuses_another_owner(tmp_path, monkeypatch):
    uid = os.getuid()
    monkeypatch.setattr(shim.os, "getuid", lambda: uid + 1)
    (tmp_path / "quotable").mkdir(mode=0o700)
    with pytest.raises(shim.ShimError, match="owned by another user"):
        shim.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


def test_key_names_the_socket(tmp_path, code):
    env = {"XDG_RUNTIME_DIR": str(tmp_path)}
    path = shim.socket_path(env)
    assert path == tmp_path / "quotable" / f"{shim.socket_key(env)}.sock"


def test_runtime_dir_longest_socket_path_fits_macos(tmp_path, code, monkeypatch):
    # macOS: TMPDIR is /var/folders/<2>/<28>/T/; sun_path holds 104 bytes including the NUL.
    macos_tmp = "/var/folders/zz/zyxvpxvq6csfxvn_n0000000000000/T"
    monkeypatch.setattr(shim.os, "getuid", lambda: 4294967294)
    monkeypatch.setattr(tempfile, "tempdir", macos_tmp)
    monkeypatch.setattr(shim, "runtime_dir", lambda env: shim._runtime_dir_path(env))
    path = shim.socket_path({})
    assert str(path).startswith(macos_tmp + "/quotable-4294967294/")
    assert len(os.fsencode(path)) < 104


def test_shim_imports_only_the_stdlib():
    probe = (
        "import sys; before = set(sys.modules); import quotable.shim; "
        "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
        "print(sorted(new - set(sys.stdlib_module_names) - {'quotable'}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"
