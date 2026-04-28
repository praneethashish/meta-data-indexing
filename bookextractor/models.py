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


class BenchmarkResult(BaseModel):
    result: BookMetadata
    debug: dict = Field(default_factory=dict)
