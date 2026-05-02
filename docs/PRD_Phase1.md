# PRD: Multi-Format Metadata Extraction Pipeline (Phase 1)

## Problem Statement

The current system only handles PDFs and relies on an internal OCR pipeline. Users need a versatile system that can extract metadata from various file types (.pdf, .json, .md, .png, .jpg) using a unified endpoint and CLI. Additionally, the OCR capability needs to be delegated to an external mineru-dots (vParse) dockerized pipeline for better accuracy, and a new image metadata extraction pipeline must be introduced to capture EXIF and basic file properties.

## Solution

Create a unified extraction endpoint (`/extract`) and CLI that routes files to specialized pipelines based on their extensions:

1.  **Text Pipeline (`.json`, `.md`)**: Content is passed directly to the multimodal LLM for structured metadata extraction.
2.  **PDF Pipeline (`.pdf`)**: Delegates OCR to a dockerized vParse (mineru-dots) API. The resulting JSON output is passed to the LLM for structured metadata extraction. If an ISBN is found, it is validated and enriched via OpenLibrary.
3.  **Image Pipeline (`.jpg`, `.png`, etc.)**: Extracts metadata (dimensions, camera info, GPS, DPI) using EXIF/PIL, normalizing GPS coordinates to decimal degrees.

## User Stories

1. As an API user, I want to upload a `.pdf` file to the `/extract` route, so that I can get its book metadata using the vParse OCR and LLM.
2. As a CLI user, I want to pass a `.md` or `.json` file to the extract command, so that I can get structured metadata directly from the LLM.
3. As an API user, I want to upload a `.jpg` or `.png` image to the `/extract` route, so that I can receive detailed EXIF and image metadata.
4. As a system administrator, I want the PDF extraction to delegate to the mineru-dots dockerized service, so that I can independently scale the OCR engine.
5. As an API user, I want the system to automatically infer the correct extraction pipeline based on my file extension.
6. As a reader, I want extracted ISBNs to be validated and enriched with OpenLibrary data.

## Implementation Decisions

- **Format Router**: A central routing module in FastAPI/Typer to inspect file extensions and delegate to specific pipeline processors.
- **PDF Pipeline**: Calls external vParse OCR API instead of internal logic. Results are processed by the LLM.
- **Text Pipeline**: Reads contents of `.md` or `.json` and sends raw text to the LLM with a specialized prompt.
- **Image Pipeline**: Uses `piexif`/`exif` and `Pillow`. Extracts: `width`, `height`, `format`, `color_space`, `bit_depth`, `exif_camera_make`, `exif_camera_model`, `exif_date_taken`, `exif_gps_latitude`, `exif_gps_longitude`, `exif_lens`, `dpi_horizontal`, `dpi_vertical`.
- **GPS Normalization**: Dedicated utility to convert degrees/minutes/seconds to decimal degrees.

## Testing Decisions

- **Mocking**: Mock external APIs (vParse, OpenLibrary) and the LLM for unit/integration testing.
- **GPS Verification**: Specific tests for GPS normalization from various EXIF formats.
- **Router Logic**: Ensure all supported extensions route to the correct internal pipeline.

## Out of Scope

- Celery/Redis asynchronous task management (Phase 2).
- Automatic deployment of the mineru-dots container.
