from .config import settings
from .models import ConfidenceScores, MagazineConfidenceScores


def calculate_book_confidence(final: dict, candidates: dict[str, list], has_isbn: bool) -> ConfidenceScores:
    scores = {}
    for field in settings.BOOK_CONFIDENCE_FIELDS:
        if not final.get(field):
            scores[field] = 0.0
        else:
            vals = [v for v in candidates[field] if v]
            consistency = vals.count(final[field]) / len(vals) if vals else 0.5
            scores[field] = min(1.0, 0.5 + 0.5 * consistency)

    scores["isbn"] = 1.0 if has_isbn else 0.0
    return ConfidenceScores(**scores)


def calculate_magazine_confidence(llm_result: dict) -> MagazineConfidenceScores:
    return MagazineConfidenceScores(
        **{
            field: threshold if llm_result.get(field) else 0.0
            for field, threshold in settings.MAGAZINE_CONFIDENCE_THRESHOLDS.items()
        }
    )
