BOOK_EXTRACTION_PROMPT = """You are an expert book metadata extractor. Your task is to identify and extract
structured metadata from noisy OCR text of scanned book pages.

The text below was extracted via OCR from scanned book pages and may contain:
- OCR errors, garbled characters, or misread words
- Mixed languages (English, Telugu, Hindi)
- Publishing information like edition details, print runs, and pricing
- Copyright notices with author names
- Publisher addresses and contact information

Extract the following fields and return ONLY a valid JSON object:
- "title": The book's title (look for prominent text, large headings, or text on the title page)
- "author": The author's full name (look near "By", "©", "Written by", or Telugu/Hindi equivalents)
- "publisher": The publisher's name (look near "Published by", "ప్రచురణ", "प्रकाशक", or publishing house names)
- "published_date": The earliest publication date (look for years like 1996, 2004 near
  "First Edition", "ముద్రణ", "संस्करण")

Rules:
- Return STRICT JSON only — no explanation, no markdown, no extra text.
- If a field cannot be confidently determined, set its value to null.
- Do NOT fabricate or guess values. Only extract what is clearly present.
- Do NOT include ISBN (it is extracted separately).
- Prefer the original/first edition date over reprint dates.

OCR Text:
{text}

JSON:
"""

MAGAZINE_EXTRACTION_PROMPT = """Extract magazine metadata from the OCR text below.
The text is primarily in {lang_label}.

Return ONLY a valid JSON object with these fields (use null for unknown fields):
- "magazine_name": Name of the magazine
- "editor": Editor name
- "publisher": Publisher name
- "issue_date": Issue month and year
- "issue_number": Issue/volume number
- "price": Price

Example:
{{"magazine_name": "చందమామ", "editor": "చక్రపాతి", "issue_date": "August 1948", "issue_number": "2"}}

OCR Text:
{text}

JSON:
"""

IMAGE_ANALYSIS_PROMPT = (
    "Analyze this image. Return ONLY a valid JSON object with the following keys: "
    "'description' (string, visual description), 'text_content' (string, any visible text, or null), "
    "'language' (string, detected language, or null), "
    "'scene_classification' (string, e.g., 'document', 'nature', 'diagram'), "
    "'entities' (list of dicts with 'name' and 'type')."
)

LANGUAGE_LABELS = {
    "te": "Telugu (తెలుగు)",
    "en": "English",
    "hi": "Hindi (हिन्दी)",
}

DEFAULT_LANGUAGE_LABEL = "Telugu (తెలుగు) and English"

BOOK_EXTRACTION_MAX_TEXT_LENGTH = 15000
MAGAZINE_EXTRACTION_MAX_TEXT_LENGTH = 3000