"""
Provenance and self-containment.

1. Every file copied from the SimStudio Final Project is byte-identical to what the
   provenance manifest recorded (SHA-256), so the vendored code and reference
   data are exactly the published ones (docs/SIMSTUDIO_PROVENANCE.json).
2. No release code imports the parent project (``simstudio``, ``anomaly_model``),
   adds paths outside the release to ``sys.path``, or references ``../`` paths.
3. The vendored packages import without the parent project being installed.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = {"simstudio", "anomaly_model"}


def _sha256(path: Path, gz: bool = False) -> str:
    h = hashlib.sha256()
    opener = gzip.open if gz else open
    with opener(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_hashes_match():
    manifest = json.loads((ROOT / "docs" / "SIMSTUDIO_PROVENANCE.json").read_text())
    assert manifest["n_files"] == len(manifest["files"]) > 300
    for entry in manifest["files"]:
        p = ROOT / entry["dest"]
        assert p.exists(), entry["dest"]
        gz = entry["encoding"].startswith("gzip")
        assert _sha256(p, gz=gz) == entry["sha256"], f"hash mismatch for {entry['dest']}"


def test_vendored_code_is_in_manifest():
    manifest = json.loads((ROOT / "docs" / "SIMSTUDIO_PROVENANCE.json").read_text())
    dests = {e["dest"] for e in manifest["files"] if e["kind"] == "code"}
    for pkg in ("src/deepdas/fp", "src/deepdas/fp_anomaly_model"):
        for py in (ROOT / pkg).glob("*.py"):
            rel = str(py.relative_to(ROOT))
            if py.name == "__init__.py" and pkg.endswith("fp_anomaly_model"):
                continue  # the only new file in a vendored package (docstring only)
            assert rel in dests, f"{rel} is not a recorded verbatim copy"


def _python_files():
    for sub in ("src", "scripts", "tests"):
        yield from (ROOT / sub).rglob("*.py")
    yield from (ROOT / "research").glob("*/code/*.py")


def test_no_imports_of_parent_project():
    for py in _python_files():
        if "/fp/" in str(py) or "/fp_anomaly_model/" in str(py):
            continue  # vendored files are checked separately (relative imports only)
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN_MODULES, f"{py}: imports {n}"


def test_vendored_files_use_only_relative_or_stdlib_imports_at_module_level():
    """Module-level imports of the vendored files are relative or third-party.

    The only parent-project import in the vendored files is the *function-level*
    ``from anomaly_model.feature_extractor import ScenarioLoader`` inside
    ``scenario_scorer.features_from_export`` — a CLI helper the release never calls
    (checked by ``test_dl_code_never_calls_features_from_export``).
    """
    for pkg in ("src/deepdas/fp", "src/deepdas/fp_anomaly_model"):
        for py in (ROOT / pkg).glob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    assert top not in FORBIDDEN_MODULES, f"{py}: absolute import of {node.module}"
                if isinstance(node, ast.Import):
                    for a in node.names:
                        assert a.name.split(".")[0] not in FORBIDDEN_MODULES, f"{py}: import {a.name}"


def test_release_code_never_calls_features_from_export():
    for py in _python_files():
        if py.name == "scenario_scorer.py" or py.name == Path(__file__).name:
            continue
        assert "features_from_export" not in py.read_text(encoding="utf-8"), str(py)


def test_no_machine_paths_and_no_parent_paths():
    pat_parent = re.compile(r"""['"]\.\./""")
    for py in _python_files():
        if py.name in ("scenario_scorer.py",):
            continue  # vendored verbatim; its sys.path line resolves inside src/ (checked below)
        text = py.read_text(encoding="utf-8")
        if py.name not in (Path(__file__).name, "verify_release.py"):  # these two contain the patterns themselves
            for bad in ("/Users/", "/home/", "/sessions/", "C:\\"):
                assert bad not in text, f"{py}: machine-specific path {bad!r}"
        for m in pat_parent.finditer(text):
            pytest.fail(f"{py}: parent-relative path near offset {m.start()}")


def test_vendored_scorer_sys_path_stays_inside_release():
    """scenario_scorer.py inserts Path(__file__).parents[2] into sys.path at import time.

    Vendored verbatim, that expression resolves to src/ of the release."""
    import deepdas.fp_anomaly_model.scenario_scorer as sc
    assert Path(sc.REPO_ROOT).resolve() == (ROOT / "src").resolve()


def test_vendored_packages_import_without_parent_project():
    for mod in ("deepdas.fp.models", "deepdas.fp.sim_core", "deepdas.fp.tracking", "deepdas.fp.kalman",
                "deepdas.fp.audit", "deepdas.fp.project_io", "deepdas.fp_anomaly_model.feature_extractor",
                "deepdas.fp_anomaly_model.scenario_scorer"):
        importlib.import_module(mod)
    assert not any(m.split(".")[0] in FORBIDDEN_MODULES for m in sys.modules)
