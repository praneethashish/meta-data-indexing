import os
import time

import bookextractor.tasks as tasks_module
from bookextractor.config import settings
from bookextractor.tasks import cleanup_stale_uploads


def test_cleanup_stale_uploads_removes_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))

    stale_file = tmp_path / "stale_file.txt"
    stale_file.write_text("old data")
    fresh_file = tmp_path / "fresh_file.txt"
    fresh_file.write_text("new data")

    old_time = time.time() - (48 * 3600)
    os.utime(str(stale_file), (old_time, old_time))

    cleanup_stale_uploads()

    assert not os.path.exists(str(stale_file))
    assert os.path.exists(str(fresh_file))


def test_cleanup_stale_uploads_skips_fresh_files(tmp_path, monkeypatch):
    monkeypatch.setattr("bookextractor.config.settings.UPLOAD_DIR", str(tmp_path))

    fresh_file = tmp_path / "fresh_file.txt"
    fresh_file.write_text("new data")

    cleanup_stale_uploads()

    assert os.path.exists(str(fresh_file))


def test_cleanup_stale_uploads_handles_missing_dir(tmp_path, monkeypatch):
    nonexistent = str(tmp_path / "does_not_exist")
    monkeypatch.setattr(settings, "UPLOAD_DIR", nonexistent)

    cleanup_stale_uploads()


def test_cleanup_stale_uploads_skips_in_flight_files(tmp_path, monkeypatch):
    monkeypatch.setattr("bookextractor.config.settings.UPLOAD_DIR", str(tmp_path))

    stale_file = tmp_path / "in_flight_stale.txt"
    stale_file.write_text("old data")

    old_time = time.time() - (48 * 3600)
    os.utime(str(stale_file), (old_time, old_time))

    # Register the file as in-flight
    tasks_module._register_in_flight(str(stale_file))

    try:
        cleanup_stale_uploads()
        # File should NOT be removed because it's in-flight
        assert os.path.exists(str(stale_file))
    finally:
        tasks_module._unregister_in_flight(str(stale_file))


def test_in_flight_register_and_unregister():
    from bookextractor.tasks import _in_flight_files, _register_in_flight, _unregister_in_flight

    _register_in_flight("/tmp/test_file.pdf")
    assert "/tmp/test_file.pdf" in _in_flight_files

    _unregister_in_flight("/tmp/test_file.pdf")
    assert "/tmp/test_file.pdf" not in _in_flight_files
