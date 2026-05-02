# PRD: Phase 2 - Audio, Video & Level 2 Image Extraction

## Problem Statement

The current extraction system supports text, PDFs, and level 1 image metadata (EXIF/dimensions) but cannot process audio and video media files. Users need to extract key technical metadata from audio and video files (duration, codec, bitrate, etc.) through the same unified `/extract` endpoint. Additionally, we need to unlock "Level 2" visual extraction for images by allowing our unified LLM (Gemma-4 via vLLM) to actually "see" and describe the images. The API title also needs a slight adjustment to reflect its broader multimedia capabilities without breaking the Python package structure.

## Solution

1.  **Audio Pipeline**: Add `ffprobe` support to extract metadata from `.mp3`, `.wav`, and `.m4a` files into an `AudioMetadata` model (with stream warnings).
2.  **Video Pipeline**: Add `ffprobe` support to extract metadata from `.mp4` and `.mkv` files into a `VideoMetadata` model.
3.  **Image Level 2 (VLM)**: Update the image pipeline to pass the image through Gemma-4 (via vLLM) to generate a descriptive summary, which will be appended to the output alongside EXIF data.
4.  **Text LLM with vLLM**: Switch from llama-cpp to vLLM + PyTorch for Gemma-4 inference, providing GPU-accelerated semantic extraction for both text and vision.
5.  **Rename API**: The FastAPI application title and documentation will be renamed to `metadata-extractor`.

## User Stories

1. As an API user, I want to upload a `.mp4` or `.mkv` video file to the `/extract` route, so that I can get its metadata (duration, width, height, codec, framerate, bitrate, format, size).
2. As an API user, I want to upload an `.mp3`, `.wav`, or `.m4a` audio file to the `/extract` route, so that I can get its metadata (duration, bitrate, sample rate, channels, codec, format, container, size).
3. As an API user, I want to see a warning in the `AudioMetadata` output if my uploaded audio file contains multiple audio streams, so that I am aware of potentially ambiguous data.
4. As an API user, I want to upload an image and receive an AI-generated description of the image contents (Level 2 extraction) alongside its technical EXIF data.
5. As a system administrator, I want `ffprobe` to be explicitly defined as a system requirement and included in the Dockerfile so the environment is easy to reproduce.
6. As a system administrator, I want the system to use vLLM for GPU-accelerated inference of Gemma-4, so that semantic extraction is fast and efficient.
7. As a CLI user, I want to pass an audio or video file path to the extraction command, so that I receive structured JSON metadata.

## Implementation Decisions

- **Format Router**: Update `main.py` to route `.mp4`, `.mkv` to `process_video()` and `.mp3`, `.wav`, `.m4a` to `process_audio()`.
- **Media Utilities (`media_utils.py`)**: Create `extract_audio_metadata(file_path)` and `extract_video_metadata(file_path)` executing `ffprobe -v quiet -print_format json -show_format -show_streams <file_path>` via `subprocess.run()`.
- **Image Pipeline (Level 2)**:
  - Update `process_image` in `pipeline.py` to prompt Gemma-4 vLLM for an image description.
  - Same unified model handles both text and vision processing.
- **Text LLM with vLLM**:
  - Switch from llama-cpp to vLLM + PyTorch for Gemma-4 inference.
  - Use model ID `google/gemma-4-E4B-it` from HuggingFace.
  - vLLM provides GPU-accelerated inference with automatic batching.
- **Data Models (`models.py`)**:
  - Create `VideoMetadata` and `AudioMetadata` models.
  - Update `ImageMetadata` to include a new optional string field: `description`.
  - Update `ExtractionResult` to include optional `video_metadata` and `audio_metadata`.
- **FastAPI Renaming**: Change `title` in `FastAPI(title="metadata-extractor")`.
- **Dependencies**: Update `Dockerfile` to install `ffmpeg` and vLLM.

## Testing Decisions

- **Unit Tests**:
  - Mock `subprocess.run` to simulate `ffprobe` JSON outputs for standard/multi-stream media.
  - Verify that the `warnings` list behaves correctly for Audio.
- **Integration Tests**: Mock VLM response for Image Level 2 extraction to verify the description field populates correctly.

## Out of Scope

- Changing the underlying Python package name (`bookextractor`).
- Audio transcription (Speech-to-Text).
- Video frame-by-frame VLM analysis.

## Further Notes

- `ffprobe` will be invoked with `-of json` or `-print_format json`.
- Video framerate parsing from rational strings (`"30000/1001"`) to float might be required.
- **vLLM Engine**: Gemma-4 (text + vision) inference uses vLLM + PyTorch for GPU-accelerated processing. Requires A100 or similar GPU with CUDA support. Single unified model for both semantic extraction and image understanding.

---

## Updated Architecture & Workflow

### Extraction Workflow

1. **Input:** File submitted via `/extract` API or CLI.
2. **Routing:** The Format Router checks the extension.
3. **Processing:**
   - **Text & PDF:** Passes through the text core (vParse -> LLM for semantic fields -> ISBN Regex -> OpenLibrary).
   - **Image:** Processed via PIL/Piexif for EXIF metadata. Additionally passed to the initialized Multimodal LLM (with `mmproj` projector) to generate a visual description.
   - **Audio & Video:** `ffprobe` is executed via subprocess. JSON output is parsed into metadata models.
4. **Output:** A structured JSON `ExtractionResult`.

### Architecture Diagram

```text
       [ USER / CLIENT ]
                |
                v
      .-----------------------.
      |     Format Router     |
      '-----------------------'
       /      |      |      \      \
      /       |      |       \      \
( .pdf )  (.md)   (.jpg)   (.mp3)  (.mp4)
   |         |      |         |       |
   v         |      v         v       v
.------.     |   .------. .-------. .-------.
|vParse|     |   | PIL/ | |ffprobe| |ffprobe|
|(OCR) |     |   |EXIF  | | Audio | | Video |
'------'     |   '------' '-------' '-------'
    \        /      |         |       |
     v      v       v         |       |
   .----------.  .------.     |       |
   |Text Core |  | VLM  |     |       |
   | Gemma-4 |  |(Desc)|     |       |
   |  vLLM   |  |Gemma-4|     |       |
   | OpenLib  |  vLLM  |     |       |
   '----------' '------'     |       |
        |           |         |       |
        v           v         v       v
   [        Structured JSON Result        ]
```
