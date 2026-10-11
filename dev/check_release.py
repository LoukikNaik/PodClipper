"""Reject incomplete or oversized distribution archives before publishing."""

from __future__ import annotations

import ast
from email.parser import BytesParser
import os
from pathlib import Path
import sys
import tarfile
import zipfile


def check_release(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1 or len(list(directory.iterdir())) != 2:
        raise ValueError("Expected exactly one wheel and one source distribution")
    expected = os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PODCLIPPER")
    versions = []
    for archive in (wheels[0], sources[0]):
        if archive.stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large archive: {archive.name}")
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as bundle:
                files = {name: bundle.read(name) for name in bundle.namelist() if not name.endswith("/")}
            metadata_path = next(name for name in files if name.endswith(".dist-info/METADATA"))
        else:
            with tarfile.open(archive) as bundle:
                files = {item.name: bundle.extractfile(item).read() for item in bundle.getmembers() if item.isfile()}
            metadata_path = min((name for name in files if name.endswith("/PKG-INFO")), key=len)
        metadata = BytesParser().parsebytes(files[metadata_path])
        if metadata["Name"] != "podclipper" or metadata["License-Expression"] != "AGPL-3.0-only":
            raise ValueError(f"Incorrect package name or license: {archive.name}")
        version = metadata["Version"]
        if not version or (expected and version != expected):
            raise ValueError(f"Version {version!r} does not match release {expected!r}")
        versions.append(version)
        for required in ("LICENSE", "NOTICE", "podclipper/config/default.yaml", "podclipper/prompts/reel_detector.txt"):
            if not any(name == required or name.endswith("/" + required) for name in files):
                raise ValueError(f"Missing {required}: {archive.name}")
        generated = next(data for name, data in files.items() if name.endswith("podclipper/_version.py"))
        tree = ast.parse(generated)
        literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
        if version not in literals:
            raise ValueError(f"Runtime version differs from metadata: {archive.name}")
        for name in files:
            parts = Path(name).parts
            if any(part in {"landing", "samples", "outputs", "transcripts", "corpus", ".cache"} or part.startswith(".env") for part in parts):
                raise ValueError(f"Unexpected release file: {name}")
            if Path(name).suffix.lower() in {".mp4", ".mov", ".wav", ".mp3", ".pt", ".pth", ".onnx", ".task", ".tflite"}:
                raise ValueError(f"Media/model file in release: {name}")
        print(f"Validated {archive.name}: version={version}, license=AGPL-3.0-only")
    if versions[0] != versions[1]:
        raise ValueError("Wheel and source versions differ")


if __name__ == "__main__":
    check_release(Path(sys.argv[1]))
