import re

INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*prompt",
    r"forget\s+(everything|all)",
    r"act\s+as\s+(if\s+you\s+are|a)\s",
    r"disregard\s+(your|previous)",
    r"new\s+instructions?:",
    r"print\s+(your\s+)?(system|initial)\s+prompt",
    r"reveal\s+(your\s+)?(instructions|prompt)",
]


def check_injection(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in INJECTION_PATTERNS)
