from typing import Optional
from pydantic import BaseModel, Field

class ConfidenceScores(BaseModel):
    title: float = Field(0.0, ge=0.0, le=1.0)
    author: float = Field(0.0, ge=0.0, le=1.0)
    publisher: float = Field(0.0, ge=0.0, le=1.0)
    isbn: float = Field(0.0, ge=0.0, le=1.0)
    published_date: float = Field(0.0, ge=0.0, le=1.0)

class BookMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    published_date: Optional[str] = None
    confidence: ConfidenceScores

class BenchmarkResult(BaseModel):
    result: BookMetadata
    debug: dict = Field(default_factory=dict)
