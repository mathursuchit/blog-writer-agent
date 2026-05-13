from typing import TypedDict


class BlogState(TypedDict):
    # User inputs
    topic: str
    tone: str              # "technical" | "beginner" | "conversational"
    target_audience: str
    target_word_count: int

    # Set by plan_blog — used by reused research nodes
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

    # Blog-specific outputs
    blog_outline: dict | None   # {title, intro_hook, sections: [{heading, key_points}]}
    blog_post: str | None       # final markdown post

    # Guardrails
    guardrail_passed: bool
    retry_count: int
    error: str | None
