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
| [Phase 2](Phase-2-Multimedia-VLM)      | ✅ COMPLETE    | Multimedia & VLM   | vLLM engine, hardware detection, model management, magazine extraction      |
| [Phase 3](Phase-3-Async-Processing)    | ✅ COMPLETE    | Async Processing   | Celery + Redis, dual-queue workers, async API endpoints, safe cleanup     |
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
- [Phase 2: Multimedia & VLM](Phase-2-Multimedia-VLM) - vLLM engine, hardware detection
- [Phase 3: Async Processing](Phase-3-Async-Processing) - Celery + Redis workers
- [Phase 4: Scale Optimization](Phase-4-Scale-Optimization) - Future optimizations
- [Phase N: Future Enhancements](Phase-N-Future-Enhancements) - Planned features

### Technical Docs

- [VParse Integration](https://github.com/suryamanoj4/mineru-dots) - External VParse repository
- [Supported Models](../../README.md#supported-models) - Qwen, Gemma models on HuggingFace

---

## Technology Stack

| Component            | Technology                      | Purpose                               |
| -------------------- | ------------------------------- | ------------------------------------- |
| **API Framework**    | FastAPI                         | REST API endpoints                    |
| **CLI**              | Typer                           | Command-line interface                |
| **OCR Engine**       | VParse (mineru-dots)            | PDF and document OCR                  |
| **LLM/VLM**          | Qwen2.5-VL, Qwen3-VL, Gemma 4   | Text semantic extraction              |
| **Inference Engine** | vLLM + PyTorch                  | GPU-accelerated LLM inference         |
| **ISBN Lookup**      | OpenLibrary API                 | Book metadata enrichment              |
| **Task Queue**       | Celery + Redis                  | Async background processing           |
| **Container**        | Docker + Docker Compose         | Deployment                            |

---

## File Processing Capabilities

| File Type                | Extraction Method                    | LLM Used? |
| ------------------------ | ------------------------------------ | --------- |
| PDF                      | VParse OCR + LLM extraction          | ✅        |
| Text (.md, .json, .txt)  | Direct LLM processing                | ✅        |
| Image (.jpg, .png, etc.) | EXIF + PIL metadata extraction (+ optional VLM) | ❌/✅ |

---

## Getting Started

1. **For Development:** See [Phase 1](Phase-1-Foundation) for implemented features
2. **For Architecture:** See [Architecture](Architecture) for system design
3. **For Docker:** See [README](../../README.md#docker) for deployment guide

---

## Contributing

When implementing new phases:

1. Update the relevant Phase page
2. Update this home page with status changes
3. Update the [Timeline](Timeline)
4. Ensure tests are added/updated

---

_Last updated: Phase 3 complete — production hardening refactor (lifecycle management, retry logic, upload limits, dead code removal)_
