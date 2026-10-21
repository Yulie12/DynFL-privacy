"""Verify cleanup recovery archives, manifest completeness and active imports."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify():
    manifest = (ROOT / "docs/CLEANUP_MANIFEST.md").read_text(encoding="utf-8")
    removed = re.findall(r"^- `(.+)`$", manifest, re.M)
    archives = {}
    names = set()
    for name in ("source_before_cleanup.zip", "unused_seal_vendor.zip"):
        path = ROOT / "out/before_paper_cleanup_20261021" / name
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"corrupt recovery archive: {name}: {bad}")
            entries = [entry.replace("\\", "/") for entry in archive.namelist()]
        if name == "source_before_cleanup.zip":
            names = set(entries)
        archives[name] = {"entries": len(entries), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "crc": "passed"}
    missing = [name for name in removed if name not in names]
    surviving = [name for name in removed if (ROOT / name).exists()]
    if missing or surviving:
        raise RuntimeError(f"cleanup mismatch: not backed up={missing}, still active={surviving}")
    imports = []
    for path in sorted((ROOT / "dynfed").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8-sig"))
        module = f"dynfed.{path.stem}"
        importlib.import_module(module)
        imports.append(module)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    return {"branch": branch, "manifest_removed_files": len(removed),
            "all_manifest_files_archived": True, "all_manifest_files_removed": True,
            "active_module_imports": imports, "archives": archives}


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2))
    (ROOT / "docs/CHECKPOINT_AUDIT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
