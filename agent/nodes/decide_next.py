import structlog
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from agent.state import ResearchState

logger = structlog.get_logger()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

TOKEN_BUDGET = 50_000
HIGH_RELEVANCE_THRESHOLD = 0.7
MIN_GOOD_SOURCES = 3


class NextAction(BaseModel):
    action: str          # "continue" | "synthesize"
    refined_query: str | None
    reasoning: str


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Decide whether to search for more information or synthesize a report now. "
        "Choose 'synthesize' if enough high-quality sources exist to answer the question well. "
        "Choose 'continue' only if critical information is clearly missing — and provide a refined query targeting that gap.",
    ),
    (
        "human",
        "Question: {question}\n\n"
        "Depth: {depth}/{max_depth}\n"
        "High-relevance sources found: {good_sources}\n"
        "Gap: {gap}",
    ),
])


async def run(state: ResearchState) -> dict:
    depth = state.get("depth", 0) + 1
    scored = state.get("scored_sources", [])
    good_sources = [s for s in scored if s.get("relevance_score", 0) >= HIGH_RELEVANCE_THRESHOLD]
    max_depth = state.get("max_depth", 3)

    if depth >= max_depth:
        logger.info("max_depth_reached", depth=depth)
        return {"depth": depth, "should_continue": False}

    if state.get("tokens_used", 0) >= TOKEN_BUDGET:
        logger.info("token_budget_reached", tokens_used=state["tokens_used"])
        return {"depth": depth, "should_continue": False}

    if len(good_sources) >= MIN_GOOD_SOURCES:
        logger.info("sufficient_sources", count=len(good_sources))
        return {"depth": depth, "should_continue": False}

    gap = f"Only {len(good_sources)} high-relevance sources found out of {len(scored)} total."

    try:
        structured_llm = llm.with_structured_output(NextAction)
        decision = await (PROMPT | structured_llm).ainvoke({
            "question": state["question"],
            "depth": depth,
            "max_depth": max_depth,
            "good_sources": len(good_sources),
            "gap": gap,
        })

        queries = list(state.get("search_queries", []))
        if decision.action == "continue" and decision.refined_query:
            queries.append(decision.refined_query)

        logger.info("decide_next", action=decision.action, depth=depth)
        return {"depth": depth, "should_continue": decision.action == "continue", "search_queries": queries}

    except Exception as e:
        logger.error("decide_failed", error=str(e))
        return {"depth": depth, "should_continue": False}
