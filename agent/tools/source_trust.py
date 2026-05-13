import re
from urllib.parse import urlparse

LOW_TRUST_PATTERNS = [
    r"pinterest\.com",
    r"quora\.com",
    r"reddit\.com/r/\w+/comments",
    r"\.xyz$",
    r"click\.",
    r"tracking\.",
    r"ads\.",
]

HIGH_TRUST_DOMAINS = [
    "arxiv.org",
    "github.com",
    "docs.",
    "developer.",
    "research.",
    "scholar.google",
    "nature.com",
    "ieee.org",
    "acm.org",
    "openai.com",
    "anthropic.com",
    "huggingface.co",
    "langchain.com",
    "langgraph",
    "python.org",
    "pytorch.org",
    "tensorflow.org",
    ".edu",
    ".gov",
]


def score_trust(url: str) -> float:
    domain = urlparse(url).netloc.lower()

    for pattern in LOW_TRUST_PATTERNS:
        if re.search(pattern, domain):
            return 0.2

    for trusted in HIGH_TRUST_DOMAINS:
        if trusted in domain:
            return 0.9

    return 0.5
