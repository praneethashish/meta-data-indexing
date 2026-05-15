class BookExtractorError(Exception):
    """Base exception for all BookExtractor errors."""


class ModelNotAvailableError(BookExtractorError):
    """Raised when the LLM/VLM model is not loaded or not available."""

    def __init__(self, message: str = "No LLM model is loaded. Ensure load_vlm=True when creating the pipeline."):
        self.message = message
        super().__init__(self.message)


class PromptFailedError(BookExtractorError):
    """Raised when LLM prompting fails to produce a valid result."""

    def __init__(self, message: str = "LLM prompt did not produce a valid result."):
        self.message = message
        super().__init__(self.message)


class ParsingError(BookExtractorError):
    """Raised when LLM output cannot be parsed as expected."""

    def __init__(self, message: str = "Failed to parse LLM output.", raw_output: str | None = None):
        self.message = message
        self.raw_output = raw_output
        super().__init__(self.message)