# Glossary

## Terms and Definitions

### A

**API** (Application Programming Interface)
> A set of protocols and tools for building software applications and enabling communication between different systems.

**Async** (Asynchronous Processing)
> Processing that doesn't block the caller. The system can handle other requests while waiting for a task to complete.

**A100** (NVIDIA A100)
> A high-performance GPU (Graphics Processing Unit) with 80GB VRAM, designed for AI and HPC workloads.

---

### B

**Batch Processing**
> Processing multiple files or items together as a group, rather than one at a time.

**Benchmark**
> A standardized test used to measure the performance of hardware or software components.

---

### C

**Celery**
> An open-source Python library for asynchronous task queue management, commonly used for distributed task processing.

**Confidence Score**
> A numerical value (0.0 to 1.0) indicating the reliability or accuracy of an extracted metadata field.

**Container** (Docker)
> A lightweight, standalone package that includes everything needed to run a piece of software.

**CPU-bound**
> A computational task limited primarily by the speed of the central processing unit rather than memory or I/O.

---

### D

**Gemma-4 vLLM**
> Google's Gemma-4 4B parameter instruction-tuned model via vLLM inference engine. A unified multimodal model that handles both text semantic extraction and image vision understanding on GPU.

**dots.mocr**
> The default VLM (Vision Language Model) used by VParse for image description and document understanding. A 3B parameter model supporting 109+ languages.

**DMS to Decimal**
> Conversion of GPS coordinates from Degrees/Minutes/Seconds format to decimal degrees format.

**Docker**
> A platform for developing, shipping, and running applications in containers.

**Docker Compose**
> A tool for defining and running multi-container Docker applications.

---

### E

**EXIF** (Exchangeable Image File Format)
> Standard format for storing metadata in digital images, including camera settings, date taken, and GPS coordinates.

**Extraction Pipeline**
> A series of processing steps that transform raw input files into structured metadata output.

---

### F

**FastAPI**
> A modern, fast Python web framework for building APIs with automatic Swagger documentation.

**ffprobe**
> A multimedia stream analyzer tool used to extract metadata from audio and video files.

**Format Router**
> A component that directs input files to the appropriate processing pipeline based on file extension.

---

### G

**Gemma-4**
> Google's open-source text-only LLM (Large Language Model), used in this project for semantic metadata extraction via vLLM + PyTorch.

**GGUF** (GPT-Generated Unified Format)
> A file format for storing large language models, optimized for efficient loading and inference. (Note: Project uses HuggingFace format with vLLM instead of GGUF)

**GPU** (Graphics Processing Unit)
> A specialized processor designed for parallel processing, essential for running VLM and other AI models efficiently.

**GPS Normalization**
> Converting GPS coordinates from DMS (Degrees/Minutes/Seconds) format to decimal degrees for standardized storage and comparison.

---

### I

**ISBN** (International Standard Book Number)
> A unique numeric identifier for books. ISBN-10 has 10 digits; ISBN-13 has 13 digits.

---

### L

**Level 1 Extraction**
> Basic metadata extraction, such as EXIF data from images (dimensions, camera info, GPS).

**Level 2 Extraction**
> AI-powered content understanding, such as VLM-generated descriptions of image content.

**LLM** (Large Language Model)
> A type of AI model trained on vast amounts of text data, capable of understanding and generating human language.

**llama-cpp**
> A C/C++ implementation of LLaMA for efficient inference, supporting GGUF model files. (Note: Project uses vLLM instead)

**vLLM**
> High-throughput, GPU-accelerated LLM inference engine developed by UC Berkeley. Used in this project for Gemma-4 inference with automatic batching and CUDA optimization.

---

### M

**Metadata**
> Data that describes other data. In this project: structured information extracted from files (title, author, ISBN, etc.).

**mmproj.gguf**
> A multimodal projector file used to connect vision encoders to language models in VLM architectures.

**Multimodal**
> AI systems that can process multiple types of input, such as text, images, and audio.

---

### O

**OCR** (Optical Character Recognition)
> Technology that converts images of text (scanned documents, photos) into machine-readable text.

**OpenLibrary**
> A free, open online library project providing book metadata through a public API.

---

### P

**PDF Metadata**
> Information embedded in PDF files, including author, title, creation date, and more.

**Phases**
> Implementation stages for the metadata extraction pipeline:
> - Phase 1: Foundation
> - Phase 2: Multimedia & VLM
> - Phase 3: Async Processing
> - Phase 4: Scale Optimization
> - Phase N: Future Enhancements

**Piexif**
> A Python library for extracting and manipulating EXIF data from images.

**PIL** (Python Imaging Library)
> A Python library for opening, manipulating, and saving many different image file formats.

**Pipeline**
> A series of processing steps that data flows through, from raw input to structured output.

**Pre-OCR**
> Metadata extraction performed before the OCR process, using quick techniques like regex on raw text.

**Pydantic**
> A Python library for data validation using Python type annotations.

---

### Q

**Queue**
> A waiting area for tasks, managed by Celery and typically backed by Redis.

---

### R

**RAG** (Retrieval-Augmented Generation)
> A technique that enhances LLM responses by retrieving relevant information from a knowledge base.

**Redis**
> An in-memory data store used as a message broker (Celery) and caching layer.

**REST API** (Representational State Transfer)
> An architectural style for web services, providing interoperability between computer systems.

---

### S

**Scale**
> The ability to handle increasing volumes of data or users without performance degradation.

**Soft Delete**
> Marking data as deleted without physically removing it from storage.

---

### T

**Task**
> A unit of work submitted to Celery for asynchronous execution.

**Task Always Eager**
> A Celery setting where tasks execute synchronously in the current process instead of being sent to a queue.

**Throughput**
> The number of items processed per unit of time.

---

### V

**VLM** (Vision Language Model)
> A type of AI model capable of understanding both images and text, generating descriptions or answering questions about images.

**VParse** (formerly MinerU)
> A document parsing and OCR toolkit that converts PDFs and images into machine-readable formats, with VLM support.

**VParse API**
> The HTTP API provided by VParse for document parsing and OCR operations.

---

### W

**Worker**
> A process that executes Celery tasks from the queue. Can be CPU workers or GPU workers.

---

## Acronyms

| Acronym | Full Form |
|---------|-----------|
| API | Application Programming Interface |
| CPU | Central Processing Unit |
| DMS | Degrees Minutes Seconds |
| EXIF | Exchangeable Image File Format |
| GPS | Global Positioning System |
| GPU | Graphics Processing Unit |
| HTTP | Hypertext Transfer Protocol |
| ISBN | International Standard Book Number |
| LLM | Large Language Model |
| OCR | Optical Character Recognition |
| PDF | Portable Document Format |
| RAG | Retrieval-Augmented Generation |
| REST | Representational State Transfer |
| VLM | Vision Language Model |
| VRAM | Video Random Access Memory |

---

## File Extensions

| Extension | File Type |
|-----------|-----------|
| `.pdf` | Portable Document Format |
| `.md` | Markdown (text) |
| `.json` | JavaScript Object Notation |
| `.jpg`, `.jpeg` | JPEG Image |
| `.png` | Portable Network Graphics |
| `.tiff` | Tagged Image File Format |
| `.webp` | WebP Image |
| `.mp3` | MP3 Audio |
| `.wav` | WAV Audio |
| `.m4a` | MPEG-4 Audio |
| `.mp4` | MPEG-4 Video |
| `.mkv` | Matroska Video |

---

## Related Documentation

- [Home](Home) - Wiki home page
- [Architecture](Architecture) - System architecture
- [Phase 1: Foundation](Phase-1-Foundation) - Completed features
- [Phase 2: Multimedia & VLM](Phase-2-Multimedia-VLM) - Current implementation
- [Phase 3: Async Processing](Phase-3-Async-Processing) - Future async architecture
- [Phase 4: Scale Optimization](Phase-4-Scale-Optimization) - Future optimizations
- [Timeline](Timeline) - Implementation timeline
