import asyncio
import structlog
from agent.state import ResearchState
from agent.tools.fetch_page import fetch_page
from agent.tools.source_trust import score_trust

logger = structlog.get_logger()

MAX_PAGES_PER_DEPTH = 5


async def run(state: ResearchState) -> dict:
    if state.get("error"):
        return {"pages_read": state.get("pages_read", [])}

    results = state.get("search_results", [])
    already_read = {p["url"] for p in state.get("pages_read", [])}
    to_read = [r for r in results if r["url"] not in already_read][:MAX_PAGES_PER_DEPTH]

    if not to_read:
        return {"pages_read": state.get("pages_read", [])}

    pages = await asyncio.gather(*[fetch_page(r["url"]) for r in to_read])

    enriched = []
    for page, result in zip(pages, to_read):
        if page.get("error") or not page.get("content"):
            continue
        enriched.append({
            **page,
            "title": result.get("title", ""),
            "trust_score": score_trust(page["url"]),
            "search_snippet": result.get("content", ""),
        })

    all_pages = state.get("pages_read", []) + enriched
    logger.info("pages_read", new=len(enriched), total=len(all_pages))
    return {"pages_read": all_pages}
