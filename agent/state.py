from typing import TypedDict


class ResearchState(TypedDict):
    question: str
    search_queries: list[str]
    search_results: list[dict]
    pages_read: list[dict]
    scored_sources: list[dict]
    depth: int
    max_depth: int
    token_budget: int
    tokens_used: int
    should_continue: bool
    final_report: dict | None
    guardrail_passed: bool
    retry_count: int
    error: str | None
