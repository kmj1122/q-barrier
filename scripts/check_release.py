"""Bounded, offline release hygiene check; not a substitute for secret review.

Checks only this source tree. No network, training, evaluation, or file writes.
Reports finding types and relative filenames, never matched credential values.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


FORBIDDEN_DIRS = {
    "checkpoints", "best_checkpoints", "buffer", "wandb", "slurm",
    "overleaf-paper", "SCARED_data", "camera_ready_context_results",
    "camera_ready_horizon_results",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".pkl", ".pickle"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".cff", ".in", ".csv"}
ALLOWED_FIGURE_CSVS = {
    "data/figures/RQ1_aggregated_plot_values.csv",
    "data/figures/RQ2_aggregated_plot_values.csv",
}
ALLOWED_PRETRAINED = {
    f"pretrained/{name}.pt" for name in ("darkroom", "point", "car", "ant", "cheetah")
}
SUSPICIOUS_TEXT = {
    "private absolute path": re.compile(r"/(?:p|u|home|scratch|sw)/[^\s\"'<>]+"),
    "GitHub token pattern": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "AWS key pattern": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential-bearing URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    "reviewer-internal marker": re.compile(
        r"\b(?:" + "|".join(("CT" + "BC", "dZ" + "zt", "wp" + "Ke")) + r")(?:\b|-Q\d)"
    ),
}
REQUIRED = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.txt", "requirements-mujoco.txt",
    "amago/LICENSE", "amago/amago/shielded_agent.py", "vendor/safety-gymnasium/LICENSE",
    "crl2_darkroom.py", "crl2_mujoco.py", "test_crl2_darkroom.py", "test_crl2_mujoco.py",
    "pretrained/manifest.json", "pretrained/README.md",
    *sorted(ALLOWED_PRETRAINED),
)


def findings_for_file(relative: Path, path: Path) -> list[str]:
    issues = []
    if path.is_symlink():
        return ["symlink requires explicit publication review"]
    if set(relative.parts) & FORBIDDEN_DIRS:
        issues.append("private/generated output directory")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES and relative.as_posix() not in ALLOWED_PRETRAINED:
        issues.append("weights or experimental data")
    if path.suffix.lower() == ".csv" and relative.as_posix() not in ALLOWED_FIGURE_CSVS:
        issues.append("unreviewed data export")
    if path.name.startswith((".env", "scratch", "rebuttal")) or path.name in {".netrc", "id_rsa", "id_ed25519"}:
        issues.append("private note or credential filename")
    if path.stat().st_size > 90 * 1024 * 1024:
        issues.append("file exceeds source-release size allowance")
    if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            return issues + ["unexpected non-UTF8 text file"]
        issues += [name for name, pattern in SUSPICIOUS_TEXT.items() if pattern.search(text)]
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError:
                issues.append("Python syntax error")
    return issues


def checkpoint_findings(root: Path) -> list[tuple[str, str]]:
    """Allow only the five curated policy files, with manifest integrity checks."""
    manifest_path = root / "pretrained/manifest.json"
    any_weights = any((root / name).exists() for name in ALLOWED_PRETRAINED)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return ([("pretrained/manifest.json", "checkpoint manifest missing or symlink")]
                if any_weights else [])
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        if (manifest.get("format") != 1 or not isinstance(files, list)
                or len(files) != len(ALLOWED_PRETRAINED)
                or {entry["path"] for entry in files} != ALLOWED_PRETRAINED):
            raise ValueError("invalid checkpoint manifest paths")
        findings = []
        for entry in files:
            relative = entry["path"]
            path = root / relative
            if not path.is_file() or path.is_symlink():
                findings.append((relative, "bundled checkpoint missing or symlink"))
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                findings.append((relative, "bundled checkpoint size/SHA256 mismatch"))
        return findings
    except (ValueError, KeyError, TypeError, OSError):
        return [("pretrained/manifest.json", "invalid checkpoint manifest")]


def scan(root: Path, require_layout: bool = True) -> list[tuple[str, str]]:
    findings = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".git" or any(p in {"__pycache__", ".pytest_cache", ".venv"} for p in relative.parts):
            continue
        if path.is_symlink() or path.is_file():
            findings += [(relative.as_posix(), issue) for issue in findings_for_file(relative, path)]
    if require_layout:
        findings += [(name, "required source/license/document missing")
                     for name in REQUIRED if not (root / name).is_file()]
    findings += checkpoint_findings(root)
    return findings


def selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="qbarrier-release-check-") as temporary:
        root = Path(temporary)
        (root / "good.py").write_text("def double(x):\n    return 2 * x\n")
        assert not scan(root, require_layout=False)
        (root / "weights.pt").write_bytes(b"not a real checkpoint")
        assert any(issue == "weights or experimental data" for _, issue in scan(root, False))
        (root / ".env").write_text("placeholder=not-a-secret\n")
        assert any(issue == "private note or credential filename" for _, issue in scan(root, False))
        (root / "bad.py").write_text("def broken(:\n")
        assert any(issue == "Python syntax error" for _, issue in scan(root, False))
        (root / "token.txt").write_text("gh" + "p_" + "A" * 36)
        assert any(issue == "GitHub token pattern" for _, issue in scan(root, False))
        (root / "path.txt").write_text("/" + "home/private-user/project")
        assert any(issue == "private absolute path" for _, issue in scan(root, False))
        (root / "link.py").symlink_to(root / "good.py")
        assert any(issue == "symlink requires explicit publication review" for _, issue in scan(root, False))
        (root / "raw.csv").write_text("value\n1\n")
        assert any(issue == "unreviewed data export" for _, issue in scan(root, False))
        relative = Path(sorted(ALLOWED_FIGURE_CSVS)[0])
        approved = root / relative
        approved.parent.mkdir(parents=True)
        approved.write_text("mean,std\n1.0,0.1\n")
        assert not findings_for_file(relative, approved)
        approved.write_text("source\n/" + "home/private-user/figure.csv\n")
        assert "private absolute path" in findings_for_file(relative, approved)
    print("Release checker self-test: 10/10 passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    for path, issue in findings:
        print(f"FAIL {path}: {issue}")
    if findings:
        return 1
    print("Release source/layout scan passed (bounded heuristic; manual review still required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
