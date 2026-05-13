import asyncio
import os

from celery import Celery

from .pipeline import ExtractionPipeline

# Initialize Celery app
celery_app = Celery("bookextractor")
celery_app.config_from_object("bookextractor.celery_config")

# Singleton pipeline instances
_pipeline_with_llm = None
_pipeline_no_llm = None


def get_pipeline(load_llm: bool = True):
    """Lazy-load the extraction pipeline within the worker process."""
    global _pipeline_with_llm, _pipeline_no_llm
    if load_llm:
        if _pipeline_with_llm is None:
            _pipeline_with_llm = ExtractionPipeline(load_llm=True)
        return _pipeline_with_llm
    else:
        if _pipeline_no_llm is None:
            _pipeline_no_llm = ExtractionPipeline(load_llm=False)
        return _pipeline_no_llm


@celery_app.task(name="bookextractor.extract_pdf", bind=True)
def extract_pdf_task(self, pdf_path: str, lang: str = "en", benchmark: bool = False):  # noqa: ARG001
    """Celery task for PDF extraction."""
    try:
        # Note: currently process_pdf calls extract_from_text (which needs LLM)
        p = get_pipeline(load_llm=True)
        return asyncio.run(p.process_pdf(pdf_path, benchmark=benchmark, lang=lang))
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@celery_app.task(name="bookextractor.extract_image", bind=True)
def extract_image_task(self, image_path: str, benchmark: bool = False):  # noqa: ARG001
    """Celery task for image extraction."""
    try:
        # EXIF extraction doesn't need LLM
        p = get_pipeline(load_llm=False)
        return asyncio.run(p.process_image(image_path, benchmark=benchmark))
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@celery_app.task(name="bookextractor.extract_text", bind=True)
def extract_text_task(self, file_path: str, benchmark: bool = False, lang: str = "en"):  # noqa: ARG001
    """Celery task for text/json file extraction."""

    try:
        # Needs LLM for semantic extraction
        p = get_pipeline(load_llm=True)
        return asyncio.run(p.process_text_file(file_path, benchmark=benchmark, lang=lang))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
