# Metadata Extraction Pipeline - Wiki

Welcome to the documentation for the multi-phase implementation of the metadata extraction pipeline.

## Overview

**Target Scale:** 130,000+ documents with audio/video content

**Infrastructure:** A100 GPU (80GB VRAM) for deployment, CPU for development

**Purpose:** Extract structured metadata, perform OCR, and generate AI-powered descriptions from documents and multimedia files.

---

## Phase Roadmap

| Phase                                  | Status         | Description        | Key Deliverables                                                            |
| -------------------------------------- | -------------- | ------------------ | --------------------------------------------------------------------------- |
| [Phase 1](Phase-1-Foundation)          | ✅ COMPLETE    | Foundation         | Format Router, PDF/Text/Image pipelines, Docker                             |
| [Phase 2](Phase-2-Multimedia-VLM)      | 🚧 IN PROGRESS | Multimedia & VLM   | Audio/Video, Pre-OCR regex, Gemma-4 vLLM VLM, OCR format, Celery foundation |
| [Phase 3](Phase-3-Async-Processing)    | 📋 FUTURE      | Async Processing   | Queue activation, distributed workers                                       |
| [Phase 4](Phase-4-Scale-Optimization)  | 📋 FUTURE      | Scale Optimization | VLM batching, caching, Kubernetes                                           |
| [Phase N](Phase-N-Future-Enhancements) | 📋 FUTURE      | Enhancements       | Streaming OCR, multi-tenant, RAG pipeline                                   |

---

## Quick Links

### Documentation

- [Architecture](Architecture) - System architecture and component overview
- [Timeline](Timeline) - Gantt chart and implementation timeline
- [Glossary](Glossary) - Terms and definitions

### Phase Details

- [Phase 1: Foundation](Phase-1-Foundation) - Completed foundation features
- [Phase 2: Multimedia & VLM](Phase-2-Multimedia-VLM) - Current implementation phase
- [Phase 3: Async Processing](Phase-3-Async-Processing) - Future async architecture
- [Phase 4: Scale Optimization](Phase-4-Scale-Optimization) - Future optimizations
- [Phase N: Future Enhancements](Phase-N-Future-Enhancements) - Planned features

### Technical Docs

- [VParse Integration](https://github.com/suryamanoj4/mineru-dots) - External VParse repository
- [Gemma-4 Model](https://huggingface.co/google/gemma-4-E4B-it) - Unified LLM/VLM on HuggingFace

---

## Technology Stack

| Component            | Technology                      | Purpose                               |
| -------------------- | ------------------------------- | ------------------------------------- |
| **API Framework**    | FastAPI                         | REST API endpoints                    |
| **OCR Engine**       | VParse (mineru-dots)            | PDF and document OCR                  |
| **LLM/VLM**          | Gemma-4 (google/gemma-4-E4B-it) | Text semantic + Image understanding   |
| **Inference Engine** | vLLM + PyTorch                  | GPU-accelerated unified LLM inference |
| **ISBN Lookup**      | OpenLibrary API                 | Book metadata enrichment              |
| **Audio/Video**      | ffprobe                         | Media metadata extraction             |
| **Task Queue**       | Celery + Redis                  | Async processing foundation           |
| **Container**        | Docker + Docker Compose         | Deployment                            |

---

## File Processing Capabilities

| File Type                | Extraction Method                    | LLM Used? |
| ------------------------ | ------------------------------------ | --------- |
| PDF                      | VParse OCR (pipeline) + Gemma-4 vLLM | ✅        |
| Text (.md, .json)        | Direct LLM processing                | ✅        |
| Image (.jpg, .png, etc.) | EXIF + **Gemma-4 vLLM**              | ✅ Yes    |
| Audio (.mp3, .wav, .m4a) | ffprobe                              | ❌        |
| Video (.mp4, .mkv)       | ffprobe                              | ❌        |

---

## Getting Started

1. **For Development:** See [Phase 1](Phase-1-Foundation) for current implementation
2. **For Implementation:** See [Phase 2](Phase-2-Multimedia-VLM) for current work
3. **For Architecture:** See [Architecture](Architecture) for system design

---

## Contributing

When implementing new phases:

1. Update the relevant Phase page
2. Update this home page with status changes
3. Update the [Timeline](Timeline)
4. Ensure tests are added/updated

---

_Last updated: Phase 2 implementation in progress_
