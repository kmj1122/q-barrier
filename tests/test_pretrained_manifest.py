"""Checkpoint allowlisting and fail-closed integrity checks; no model execution."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_release import ALLOWED_PRETRAINED, checkpoint_findings, scan


def fixture_bundle(tmp_path):
    entries = []
    for name in sorted(ALLOWED_PRETRAINED):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately not a torch checkpoint: this tests bytes, not deserialization.
        content = f"integrity fixture: {name}".encode()
        path.write_bytes(content)
        entries.append({"path": name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = tmp_path / "pretrained/manifest.json"
    manifest.write_text(json.dumps({"format": 1, "files": entries}))
    return manifest


def test_curated_weights_and_manifest_are_accepted(tmp_path):
    fixture_bundle(tmp_path)
    assert not scan(tmp_path, require_layout=False)


def test_changed_checkpoint_fails_integrity(tmp_path):
    fixture_bundle(tmp_path)
    (tmp_path / "pretrained/darkroom.pt").write_bytes(b"tampered")
    assert ("pretrained/darkroom.pt", "bundled checkpoint size/SHA256 mismatch") in checkpoint_findings(tmp_path)


def test_unknown_checkpoint_is_not_allowlisted(tmp_path):
    fixture_bundle(tmp_path)
    (tmp_path / "pretrained/unreviewed.pt").write_bytes(b"fixture")
    assert ("pretrained/unreviewed.pt", "weights or experimental data") in scan(tmp_path, False)


def test_missing_manifest_rejects_present_weights(tmp_path):
    fixture_bundle(tmp_path).unlink()
    assert ("pretrained/manifest.json", "checkpoint manifest missing or symlink") in checkpoint_findings(tmp_path)


@pytest.mark.parametrize("mutation", ["duplicate", "traversal"])
def test_duplicate_or_path_traversal_manifest_is_rejected(tmp_path, mutation):
    path = fixture_bundle(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["files"][-1]["path"] = (
        manifest["files"][0]["path"] if mutation == "duplicate" else "../outside.pt"
    )
    path.write_text(json.dumps(manifest))
    assert ("pretrained/manifest.json", "invalid checkpoint manifest") in checkpoint_findings(tmp_path)


def test_checkpoint_symlink_is_not_trusted(tmp_path):
    fixture_bundle(tmp_path)
    path = tmp_path / "pretrained/darkroom.pt"
    path.unlink()
    path.symlink_to(tmp_path / "pretrained/point.pt")
    assert ("pretrained/darkroom.pt", "bundled checkpoint missing or symlink") in checkpoint_findings(tmp_path)


def test_actual_bundle_matches_integrity_manifest():
    root = Path(__file__).resolve().parents[1]
    assert not checkpoint_findings(root)
