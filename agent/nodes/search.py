import structlog
from agent.state import ResearchState
from agent.tools.tavily_search import search as tavily_search
from agent.tools.injection_guard import check_injection

logger = structlog.get_logger()


async def run(state: ResearchState) -> dict:
    question = state["question"]

    if check_injection(question):
        logger.warning("injection_detected", question=question[:100])
        return {"error": "Prompt injection detected in question.", "should_continue": False}

    queries = state.get("search_queries", [])
    query = queries[-1] if queries else question

    try:
        results = tavily_search(query, max_results=5)
        all_results = state.get("search_results", []) + results
        logger.info("search_complete", query=query[:80], new_results=len(results))
        return {
            "search_queries": queries + [query],
            "search_results": all_results,
            "error": None,
        }
    except Exception as e:
        logger.error("search_failed", error=str(e))
        return {"error": str(e), "should_continue": False}
