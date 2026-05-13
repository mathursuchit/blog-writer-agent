import re
import structlog
from agent.blog_state import BlogState

logger = structlog.get_logger()

URL_PATTERN = re.compile(r"https?://[^\s\)\]\>\"']+")


async def run(state: BlogState) -> dict:
    blog_post = state.get("blog_post")
    retry_count = state.get("retry_count", 0)

    if not blog_post:
        logger.warning("blog_guardrail_failed", reason="no_post")
        return {"guardrail_passed": False, "retry_count": retry_count + 1}

    if len(blog_post.split()) < 200:
        logger.warning("blog_guardrail_failed", reason="too_short", words=len(blog_post.split()))
        return {"guardrail_passed": False, "retry_count": retry_count + 1}

    known_urls = {s["url"] for s in state.get("scored_sources", [])}
    found_urls = URL_PATTERN.findall(blog_post)
    hallucinated = [u for u in found_urls if u not in known_urls]

    if hallucinated:
        logger.warning("hallucinated_urls_removed", count=len(hallucinated))
        cleaned = blog_post
        for url in hallucinated:
            cleaned = cleaned.replace(url, "[source removed]")
        return {"blog_post": cleaned, "guardrail_passed": True}

    logger.info("blog_guardrail_passed", words=len(blog_post.split()))
    return {"guardrail_passed": True}
