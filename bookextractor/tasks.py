import logging
import os
import threading
import time

from celery import Celery
from celery.signals import worker_ready

from .pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery("bookextractor")
celery_app.config_from_object("bookextractor.celery_config")

# Singleton pipeline instances
_pipeline_with_vlm = None
_pipeline_no_vlm = None

UPLOAD_DIR = os.getenv("BOOKEXTRACTOR_UPLOAD_DIR", "uploads")
STALE_FILE_THRESHOLD = 86400  # 24 hours in seconds

# In-flight file tracking to prevent cleanup from deleting active files
_in_flight_files: set[str] = set()
_in_flight_lock = threading.Lock()


def _register_in_flight(filepath: str) -> None:
    with _in_flight_lock:
        _in_flight_files.add(filepath)


def _unregister_in_flight(filepath: str) -> None:
    with _in_flight_lock:
        _in_flight_files.discard(filepath)


@worker_ready.connect
def cleanup_stale_uploads(**_kwargs):
    now = time.time()
    if not os.path.isdir(UPLOAD_DIR):
        return
    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        try:
            with _in_flight_lock:
                if filepath in _in_flight_files:
                    continue
            if os.path.isfile(filepath) and (now - os.path.getmtime(filepath)) > STALE_FILE_THRESHOLD:
                os.remove(filepath)
                logger.info(f"Removed stale upload: {filepath}")
        except Exception:
            logger.exception(f"Failed to remove stale file: {filepath}")


def get_pipeline(load_vlm: bool = True):
    """Lazy-load the extraction pipeline within the worker process."""
    global _pipeline_with_vlm, _pipeline_no_vlm
    if load_vlm:
        if _pipeline_with_vlm is None:
            _pipeline_with_vlm = ExtractionPipeline(load_vlm=True)
        return _pipeline_with_vlm
    else:
        if _pipeline_no_vlm is None:
            _pipeline_no_vlm = ExtractionPipeline(load_vlm=False)
        return _pipeline_no_vlm


@celery_app.task(name="bookextractor.extract_pdf", bind=True)
def extract_pdf_task(self, pdf_path: str, lang: str = "en", benchmark: bool = False):  # noqa: ARG001
    """Celery task for PDF extraction."""
    _register_in_flight(pdf_path)
    try:
        p = get_pipeline(load_vlm=True)
        return p.process_pdf_sync(pdf_path, benchmark=benchmark, lang=lang)
    finally:
        _unregister_in_flight(pdf_path)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@celery_app.task(name="bookextractor.extract_image", bind=True)
def extract_image_task(self, image_path: str, benchmark: bool = False, use_vlm: bool = False):  # noqa: ARG001
    """Celery task for image extraction."""
    _register_in_flight(image_path)
    try:
        p = get_pipeline(load_vlm=use_vlm)
        return p.process_image_sync(image_path, benchmark=benchmark)
    finally:
        _unregister_in_flight(image_path)
        if os.path.exists(image_path):
            os.remove(image_path)


@celery_app.task(name="bookextractor.extract_text", bind=True)
def extract_text_task(self, file_path: str, benchmark: bool = False, lang: str = "en"):  # noqa: ARG001
    """Celery task for text/json file extraction."""
    _register_in_flight(file_path)
    try:
        p = get_pipeline(load_vlm=True)
        return p.process_text_file_sync(file_path, benchmark=benchmark, lang=lang)
    finally:
        _unregister_in_flight(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)