import hashlib
import os
import subprocess
import sys
import json
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_release as release  # noqa: E402
from build_release import ReleaseError, build_release  # noqa: E402


def test_bundled_runtime_precedes_an_installed_tokenpulse(tmp_path: Path) -> None:
    other = tmp_path / "installed" / "token_pulse"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("raise RuntimeError('wrong runtime')\n")
    vendor = Path(__file__).resolve().parents[1] / "plugins/codex-speed/skills/codex-speed/_vendor"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(vendor), str(other.parent)))}
    result = subprocess.run(
        [sys.executable, "-S", "-c", "from token_pulse.recent import query_recent; assert callable(query_recent)"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def _release_tree(root: Path, version: str = "7.6.5") -> Path:
    plugin = root / "plugins" / "codex-speed"
    skill = plugin / "skills" / "codex-speed"
    vendor = skill / "_vendor" / "token_pulse"
    files = {
        ".codex-plugin/plugin.json": json.dumps(
            {"name": "codex-speed", "version": version, "skills": "./skills/"}
        ),
        "README.md": "plugin readme\n",
        "skills/codex-speed/SKILL.md": "skill instructions\n",
        "skills/codex-speed/LICENSE": "MIT\n",
        "skills/codex-speed/THIRD_PARTY.md": "upstream notice\n",
        "skills/codex-speed/scripts/codex_speed.py": "print('ready')\n",
        "skills/codex-speed/_vendor/token_pulse/domain.py": "VERSION = 1\n",
        "skills/codex-speed/_vendor/token_pulse/recent.py": "def recent(): return []\n",
    }
    source = {
        "plugin": "codex-speed",
        "plugin_version": "0.2.0",
        "source_project": "TokenPulse",
        "source_repository": "https://github.com/zzccchen/token-pulse",
        "source_base_commit": "0123456789abcdef",
        "source_overlay_sha256": "old-overlay-digest",
        "source_version": "0.1.1",
        "bundled_modules": ["token_pulse.domain"],
    }
    files["skills/codex-speed/SOURCE.json"] = json.dumps(source, indent=2)
    for relative, content in files.items():
        destination = plugin / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    # These represent local history, tests, and cached data that must not ship.
    (plugin / "benchmarks.sqlite3").write_bytes(b"private history")
    (plugin / "probe.json").write_text("{}", encoding="utf-8")
    (vendor / "__pycache__").mkdir()
    (vendor / "__pycache__" / "domain.pyc").write_bytes(b"cache")
    return plugin


def test_archive_uses_manifest_version_whitelist_and_current_provenance(tmp_path):
    plugin = _release_tree(tmp_path, version="4.2.0")
    original_source = (plugin / "skills/codex-speed/SOURCE.json").read_bytes()
    archive_path, checksum_path = build_release(tmp_path, Path("dist"))

    assert archive_path.name == "codex-speed-4.2.0.zip"
    assert checksum_path.read_text(encoding="ascii") == (
        f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n"
    )
    assert (plugin / "skills/codex-speed/SOURCE.json").read_bytes() == original_source

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert names == {
            ".codex-plugin/plugin.json",
            "README.md",
            "skills/codex-speed/SKILL.md",
            "skills/codex-speed/LICENSE",
            "skills/codex-speed/THIRD_PARTY.md",
            "skills/codex-speed/SOURCE.json",
            "skills/codex-speed/scripts/codex_speed.py",
            "skills/codex-speed/_vendor/token_pulse/domain.py",
            "skills/codex-speed/_vendor/token_pulse/recent.py",
        }
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        source = json.loads(archive.read("skills/codex-speed/SOURCE.json"))
        assert source["plugin_version"] == "4.2.0"
        assert source["source_repository"] == "https://github.com/tawanerguo-cn/codex-speed"
        assert source["upstream_repository"] == "https://github.com/zzccchen/token-pulse"
        assert source["upstream_base_commit"] == "0123456789abcdef"
        assert source["upstream_source_version"] == "0.1.1"
        assert source["upstream_overlay_sha256"] == "old-overlay-digest"
        runtime_files = {
            "skills/codex-speed/_vendor/token_pulse/domain.py",
            "skills/codex-speed/_vendor/token_pulse/recent.py",
        }
        assert source["runtime_files_sha256"] == {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in runtime_files
        }
        assert archive.read("skills/codex-speed/_vendor/token_pulse/domain.py") == b"VERSION = 1\n"
        assert "benchmarks.sqlite3" not in names
        assert "probe.json" not in names
        assert all("__pycache__" not in name for name in names)


def test_build_is_byte_reproducible_and_accepts_identical_existing_output(tmp_path):
    _release_tree(tmp_path)
    first_zip, first_sum = build_release(tmp_path, Path("out-a"))
    second_zip, second_sum = build_release(tmp_path, Path("out-b"))

    assert first_zip.read_bytes() == second_zip.read_bytes()
    assert first_sum.read_bytes() == second_sum.read_bytes()
    same_zip, same_sum = build_release(tmp_path, Path("out-a"))
    assert same_zip == first_zip
    assert same_sum == first_sum


def test_conflicting_output_is_rejected_without_replacing_it(tmp_path):
    _release_tree(tmp_path)
    output = tmp_path / "dist"
    output.mkdir()
    archive = output / "codex-speed-7.6.5.zip"
    archive.write_bytes(b"another artifact")

    with pytest.raises(ReleaseError, match="refusing to overwrite"):
        build_release(tmp_path, Path("dist"))
    assert archive.read_bytes() == b"another artifact"
    assert not (output / "codex-speed-7.6.5.zip.sha256").exists()


def test_linked_release_input_is_rejected(tmp_path):
    plugin = _release_tree(tmp_path)
    readme = plugin / "README.md"
    external = tmp_path / "outside.md"
    external.write_text("outside", encoding="utf-8")
    readme.unlink()
    try:
        readme.symlink_to(external)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ReleaseError, match="symlink/reparse"):
        build_release(tmp_path, Path("dist"))


def test_reparse_input_flag_is_rejected(tmp_path, monkeypatch):
    _release_tree(tmp_path)
    is_link = release._is_reparse_or_symlink
    monkeypatch.setattr(
        release,
        "_is_reparse_or_symlink",
        lambda path: path.name == "README.md" or is_link(path),
    )

    with pytest.raises(ReleaseError, match="symlink/reparse"):
        build_release(tmp_path, Path("dist"))


def test_output_cannot_be_written_inside_plugin_source(tmp_path):
    plugin = _release_tree(tmp_path)
    with pytest.raises(ReleaseError, match="outside the plugin source tree"):
        build_release(tmp_path, plugin / "release")
