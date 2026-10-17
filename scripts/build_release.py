"""Build a reproducible, source-only Codex Speed plugin archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from pathlib import Path
from typing import Sequence


SOURCE_REPOSITORY = "https://github.com/tawanerguo-cn/codex-speed"
PLUGIN_PATH = Path("plugins/codex-speed")
SKILL_PATH = Path("skills/codex-speed")
VENDOR_PATH = SKILL_PATH / "_vendor" / "token_pulse"
REQUIRED_FILES = (
    Path(".codex-plugin/plugin.json"),
    Path("README.md"),
    SKILL_PATH / "SKILL.md",
    SKILL_PATH / "LICENSE",
    SKILL_PATH / "THIRD_PARTY.md",
    SKILL_PATH / "SOURCE.json",
    SKILL_PATH / "scripts/codex_speed.py",
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]*$")


class ReleaseError(RuntimeError):
    """The release inputs are invalid or would overwrite different output."""


def _is_reparse_or_symlink(path: Path) -> bool:
    info = path.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _checked_source_path(root: Path, relative: Path, *, directory: bool = False) -> Path:
    """Check every source path component without following links."""
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ReleaseError(f"required release input is missing: {relative.as_posix()}") from exc
        if _is_reparse_or_symlink(current):
            raise ReleaseError(f"release input contains a symlink/reparse point: {current}")
        if current != root / relative and not stat.S_ISDIR(info.st_mode):
            raise ReleaseError(f"release input parent is not a directory: {current}")
    if directory:
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise ReleaseError(f"release input is not a directory: {relative.as_posix()}")
    elif not stat.S_ISREG(current.lstat().st_mode):
        raise ReleaseError(f"release input is not a regular file: {relative.as_posix()}")
    return current


def _source_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for relative in REQUIRED_FILES:
        source = _checked_source_path(root, PLUGIN_PATH / relative)
        files.append((relative.as_posix(), source))

    vendor_relative = PLUGIN_PATH / VENDOR_PATH
    vendor = _checked_source_path(root, vendor_relative, directory=True)
    modules: list[tuple[str, Path]] = []
    with os.scandir(vendor) as entries:
        for entry in entries:
            if not entry.name.endswith(".py"):
                continue
            path = vendor / entry.name
            if _is_reparse_or_symlink(path):
                raise ReleaseError(f"release input contains a symlink/reparse point: {path}")
            if not entry.is_file(follow_symlinks=False):
                raise ReleaseError(f"vendored module is not a regular file: {path}")
            relative = VENDOR_PATH / entry.name
            checked = _checked_source_path(root, PLUGIN_PATH / relative)
            modules.append((relative.as_posix(), checked))
    if not modules:
        raise ReleaseError("no vendored token_pulse Python modules found")
    files.extend(sorted(modules))

    names = [name for name, _ in files]
    if len(names) != len(set(names)):
        raise ReleaseError("release whitelist contains duplicate paths")
    return sorted(files)


def _read_source(path: Path) -> bytes:
    """Normalize text checkout line endings for cross-platform archive bytes."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def _load_metadata(files: list[tuple[str, Path]]) -> tuple[str, bytes]:
    manifest_path = next(path for name, path in files if name == ".codex-plugin/plugin.json")
    source_path = next(path for name, path in files if name == (SKILL_PATH / "SOURCE.json").as_posix())
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"invalid release JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        raise ReleaseError("plugin manifest and SOURCE.json must be JSON objects")
    version = manifest.get("version")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ReleaseError("plugin manifest must contain a valid version string")

    old_repository = metadata.get("upstream_repository") or metadata.get("source_repository")
    if not isinstance(old_repository, str) or not old_repository:
        raise ReleaseError("SOURCE.json must identify the upstream source repository")
    upstream_base = metadata.pop("upstream_base_commit", None) or metadata.pop("source_base_commit", None)
    if upstream_base:
        metadata["upstream_base_commit"] = upstream_base
    upstream_version = metadata.get("upstream_source_version") or metadata.get("source_version")
    if upstream_version:
        metadata["upstream_source_version"] = upstream_version
    old_overlay = metadata.pop("upstream_overlay_sha256", None) or metadata.pop("source_overlay_sha256", None)
    if old_overlay:
        metadata["upstream_overlay_sha256"] = old_overlay

    vendor_prefix = (SKILL_PATH / "_vendor" / "token_pulse").as_posix() + "/"
    runtime_hashes = {
        vendor_prefix + Path(name).name: hashlib.sha256(_read_source(path)).hexdigest()
        for name, path in files
        if name.startswith((SKILL_PATH / "_vendor" / "token_pulse").as_posix() + "/")
    }
    metadata["plugin"] = manifest.get("name", "codex-speed")
    metadata["plugin_version"] = version
    metadata["source_repository"] = SOURCE_REPOSITORY
    metadata["upstream_repository"] = old_repository
    metadata["bundled_modules"] = [
        "token_pulse." + Path(name).stem
        for name, _ in files
        if name.startswith((SKILL_PATH / "_vendor" / "token_pulse").as_posix() + "/")
    ]
    metadata["runtime_files_sha256"] = dict(sorted(runtime_hashes.items()))
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return version, encoded


def _zip_bytes(files: list[tuple[str, Path]], source_json: bytes) -> bytes:
    from io import BytesIO

    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in files:
            data = source_json if name == (SKILL_PATH / "SOURCE.json").as_posix() else _read_source(source)
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def _check_nonconflicting(path: Path, data: bytes) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if _is_reparse_or_symlink(path) or not stat.S_ISREG(info.st_mode) or path.read_bytes() != data:
        raise ReleaseError(f"refusing to overwrite conflicting output: {path}")


def _write_if_nonconflicting(path: Path, data: bytes) -> None:
    _check_nonconflicting(path, data)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError:
        _check_nonconflicting(path, data)
        return
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except OSError:
        path.unlink(missing_ok=True)
        raise


def build_release(repo_root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = Path(repo_root).resolve()
    files = _source_files(root)
    version, source_json = _load_metadata(files)
    archive_bytes = _zip_bytes(files, source_json)
    digest = hashlib.sha256(archive_bytes).hexdigest()
    filename = f"codex-speed-{version}.zip"
    requested_output = Path(output_dir)
    destination_dir = (root / requested_output).resolve() if not requested_output.is_absolute() else requested_output.resolve()
    plugin_root = (root / PLUGIN_PATH).resolve()
    if destination_dir == plugin_root or plugin_root in destination_dir.parents:
        raise ReleaseError("output directory must be outside the plugin source tree")
    archive_path = destination_dir / filename
    checksum_path = destination_dir / (filename + ".sha256")
    checksum = f"{digest}  {filename}\n".encode("ascii")
    _check_nonconflicting(archive_path, archive_bytes)
    _check_nonconflicting(checksum_path, checksum)
    _write_if_nonconflicting(archive_path, archive_bytes)
    _write_if_nonconflicting(checksum_path, checksum)
    return archive_path, checksum_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the reproducible Codex Speed plugin ZIP.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        output_dir = args.output_dir if args.output_dir is not None else repo_root / "dist"
        archive_path, checksum_path = build_release(repo_root, output_dir)
    except (OSError, ReleaseError) as exc:
        print(f"release build failed: {exc}", file=sys.stderr)
        return 2
    print(f"{archive_path} ({hashlib.sha256(archive_path.read_bytes()).hexdigest()})")
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
