from pathlib import Path
import re

from packaging.requirements import Requirement
import pytest

from scripts.check_release import scan, selftest


def test_failure_mode_fixtures():
    selftest()


def test_source_tree_has_no_disallowed_release_artifacts():
    assert not scan(Path(__file__).resolve().parents[1])


def test_single_requirements_covers_full_install():
    root = Path(__file__).resolve().parents[1]
    assert sorted(path.name for path in root.glob("requirements*.txt")) == ["requirements.txt"]
    lines = [line.strip() for line in (root / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    editables = [line.removeprefix("-e ") for line in lines if line.startswith("-e ")]
    assert set(editables) == {"./amago", "./vendor/safety-gymnasium"}
    assert all((root / path).is_dir() for path in editables)
    packages = [Requirement(line) for line in lines if not line.startswith("-e ")]
    by_name = {package.name: package for package in packages}
    assert len(by_name) == len(packages), "Merge duplicate package constraints instead of dropping one"
    assert {"torch", "numpy", "matplotlib", "pytest", "mujoco", "gymnasium-robotics",
            "pygame", "glfw", "xmltodict", "PyYAML"} <= by_name.keys()
    assert "1.23.5" in by_name["numpy"].specifier
    assert "1.24" not in by_name["numpy"].specifier
    assert str(by_name["matplotlib"].specifier) == "==3.9.2"
    assert str(by_name["torch"].specifier) == "==2.10.0"


@pytest.mark.parametrize("readme", ["README.md", "pretrained/README.md", "data/figures/README.md"])
def test_project_readme_local_links_resolve(readme):
    root = Path(__file__).resolve().parents[1]
    path = root / readme
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        relative = target.split("#", 1)[0].strip("<>")
        assert (path.parent / relative).exists(), f"Broken link in {readme}: {target}"
