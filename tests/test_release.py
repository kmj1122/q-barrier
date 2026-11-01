from pathlib import Path

from scripts.check_release import scan, selftest


def test_failure_mode_fixtures():
    selftest()


def test_source_tree_has_no_disallowed_release_artifacts():
    assert not scan(Path(__file__).resolve().parents[1])
