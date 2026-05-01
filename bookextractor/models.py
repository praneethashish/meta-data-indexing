from pydantic import BaseModel, Field


class ConfidenceScores(BaseModel):
    title: float = Field(0.0, ge=0.0, le=1.0)
    author: float = Field(0.0, ge=0.0, le=1.0)
    publisher: float = Field(0.0, ge=0.0, le=1.0)
    isbn: float = Field(0.0, ge=0.0, le=1.0)
    published_date: float = Field(0.0, ge=0.0, le=1.0)

class BookMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    publisher: str | None = None
    isbn: str | None = None
    published_date: str | None = None
    confidence: ConfidenceScores

class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    color_space: str | None = None
    bit_depth: int | None = None
    exif_camera_make: str | None = None
    exif_camera_model: str | None = None
    exif_date_taken: str | None = None
    exif_gps_latitude: float | None = None
    exif_gps_longitude: float | None = None
    exif_lens: str | None = None
    dpi_horizontal: float | None = None
    dpi_vertical: float | None = None

class ImageVLMMetadata(BaseModel):
    description: str | None = None
    text_content: str | None = None
    language: str | None = None
    scene_classification: str | None = None
    entities: list[dict] = []

class ExtractionResult(BaseModel):
    book_metadata: BookMetadata | None = None
    image_metadata: ImageMetadata | None = None
    image_vlm_metadata: ImageVLMMetadata | None = None

class BenchmarkResult(BaseModel):
    result: ExtractionResult
    debug: dict = Field(default_factory=dict)
