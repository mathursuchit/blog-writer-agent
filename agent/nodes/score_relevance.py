import structlog
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from agent.state import ResearchState

logger = structlog.get_logger()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class SourceScore(BaseModel):
    url: str
    relevance: float = Field(ge=0.0, le=1.0)


class RelevanceScores(BaseModel):
    scores: list[SourceScore]


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Score each source's relevance to the research question from 0.0 to 1.0. "
        "Be strict — score above 0.7 only if the source directly addresses the question. "
        "Return one score per URL provided.",
    ),
    ("human", "Question: {question}\n\nSources:\n{sources}"),
])


async def run(state: ResearchState) -> dict:
    if state.get("error"):
        return {"scored_sources": state.get("scored_sources", [])}

    pages = state.get("pages_read", [])
    already_scored = {s["url"] for s in state.get("scored_sources", [])}
    new_pages = [p for p in pages if p["url"] not in already_scored]

    if not new_pages:
        return {"scored_sources": state.get("scored_sources", [])}

    sources_text = "\n\n".join(
        f"URL: {p['url']}\nTitle: {p.get('title', '')}\nExcerpt: {p['content'][:400]}"
        for p in new_pages
    )

    try:
        structured_llm = llm.with_structured_output(RelevanceScores)
        result = await (PROMPT | structured_llm).ainvoke({
            "question": state["question"],
            "sources": sources_text,
        })

        score_map = {s.url: s.relevance for s in result.scores}
        scored = [
            {**p, "relevance_score": score_map.get(p["url"], p.get("trust_score", 0.5))}
            for p in new_pages
        ]
    except Exception as e:
        logger.error("scoring_failed", error=str(e))
        scored = [{**p, "relevance_score": p.get("trust_score", 0.5)} for p in new_pages]

    all_scored = state.get("scored_sources", []) + scored
    high = sum(1 for s in scored if s.get("relevance_score", 0) >= 0.7)
    logger.info("relevance_scored", new=len(scored), high_relevance=high)
    return {"scored_sources": all_scored}
