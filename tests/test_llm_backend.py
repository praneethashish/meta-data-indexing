import pytest

from bookextractor.llm_backend import TextLLMBackend, VisionLLMBackend


class MockOutput:
    def __init__(self, text):
        self.text = text


class MockRequestOutput:
    def __init__(self, text):
        self.outputs = [MockOutput(text)]


class MockLLMEngine:
    def __init__(self, outputs=None):
        self._outputs = outputs or []

    def generate(self, prompts, sampling_params=None):
        return self._outputs


@pytest.mark.skipif(
    True,
    reason="Requires vLLM SamplingParams; tested via mocks in test_vlm_client.py",
)
def test_text_backend_with_real_vllm():
    pass


def test_text_backend_is_ready_when_model_loaded():
    mock_model = MockLLMEngine()
    backend = TextLLMBackend(model=mock_model)
    assert backend.is_ready() is True


def test_text_backend_not_ready_without_model():
    backend = TextLLMBackend(model=None)
    assert backend.is_ready() is False


def test_text_backend_generate_raises_without_model():
    backend = TextLLMBackend(model=None)
    with pytest.raises(RuntimeError, match="model is not loaded"):
        backend.generate(["test prompt"])


def test_text_backend_extract_json_from_output():
    backend = TextLLMBackend(model=None)
    result = backend.extract_json_from_output('Some text {"key": "value"} more text')
    assert result == {"key": "value"}


def test_text_backend_extract_json_from_output_empty():
    backend = TextLLMBackend(model=None)
    result = backend.extract_json_from_output("no json here")
    assert result is None


def test_text_backend_extract_json_from_output_invalid():
    backend = TextLLMBackend(model=None)
    result = backend.extract_json_from_output('{"invalid": json}')
    assert result is None


def test_vision_backend_is_ready_when_model_loaded():
    mock_model = MockLLMEngine()
    backend = VisionLLMBackend(model=mock_model)
    assert backend.is_ready() is True


def test_vision_backend_not_ready_without_model():
    backend = VisionLLMBackend(model=None)
    assert backend.is_ready() is False


def test_vision_backend_generate_raises_without_model():
    backend = VisionLLMBackend(model=None)
    with pytest.raises(RuntimeError, match="model is not loaded"):
        backend.generate(["test prompt"])


def test_text_backend_generate_and_extract():
    mock_output = MockRequestOutput('{"title": "Test Book", "author": "Test Author"}')
    mock_model = MockLLMEngine(outputs=[mock_output])
    backend = TextLLMBackend(model=mock_model)
    result = backend.generate_and_extract("Extract metadata from this text")
    assert result is not None
    assert result["title"] == "Test Book"
    assert result["author"] == "Test Author"


def test_text_backend_generate_and_extract_failure():
    mock_model = MockLLMEngine(outputs=[])
    backend = TextLLMBackend(model=mock_model)
    result = backend.generate_and_extract("Extract metadata")
    assert result is None