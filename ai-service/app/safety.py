"""Deterministic guardrails that run before and after the LLM.

Rules here are intentionally simple and auditable: they do not depend on the model behaving.
"""

import re

# Red-flag phrases that should short-circuit to emergency guidance instead of booking.
_EMERGENCY = re.compile(
    r"\b(chest pain|can'?t breathe|cannot breathe|difficulty breathing|trouble breathing|short(ness)? of breath|"
    r"anaphyla\w*|throat (is )?(closing|swelling)|swollen (throat|tongue)|unconscious|passed out|fainted|"
    r"seizure|stroke|face drooping|severe bleeding|bleeding heavily|suicid\w*|kill myself|overdose|"
    r"heart attack)\b",
    re.IGNORECASE,
)

# Typical prompt-injection phrasing. Matching messages are still answered, but the model is told
# to treat the text as data and the event is recorded in the trace.
_INJECTION = re.compile(
    r"(ignore (all |the )?(previous|above|prior) (instructions|rules)|system prompt|you are now|"
    r"developer mode|reveal your (prompt|instructions)|act as (an? )?(admin|developer)|"
    r"book (it )?for (patient )?P0*[2-9]\d*|another patient)",
    re.IGNORECASE,
)

_CITATION = re.compile(r"\[([CDT]\d{3,})\]")

MEDICAL_DISCLAIMER = "This is general information, not a diagnosis; please follow your doctor's advice."


def is_emergency(message: str) -> bool:
    return bool(_EMERGENCY.search(message))


def looks_like_injection(message: str) -> bool:
    return bool(_INJECTION.search(message))


def check_citations(text: str, allowed_ids: set[str]) -> tuple[str, list[str], list[str]]:
    """Keeps citations that point at retrieved sources and removes invented ones.

    Returns (clean_text, valid_ids, removed_ids).
    """
    valid: list[str] = []
    removed: list[str] = []

    def _sub(m: re.Match) -> str:
        cid = m.group(1)
        if cid in allowed_ids:
            if cid not in valid:
                valid.append(cid)
            return m.group(0)
        removed.append(cid)
        return ""

    clean = _CITATION.sub(_sub, text)
    clean = re.sub(r"[ \t]+([,.;:)])", r"\1", clean)  # no stray space left where a citation was removed
    return re.sub(r"[ \t]{2,}", " ", clean).strip(), valid, removed
