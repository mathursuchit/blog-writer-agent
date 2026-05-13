import structlog
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from agent.blog_state import BlogState

logger = structlog.get_logger()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class BlogSection(BaseModel):
    heading: str
    key_points: list[str]


class BlogOutline(BaseModel):
    title: str
    intro_hook: str
    sections: list[BlogSection]
    search_query: str


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a blog strategist. Given a topic, tone, and target audience, create a clear blog outline "
        "with a compelling title, a one-sentence intro hook, and 4-6 sections. Each section should have "
        "2-3 key points to cover. Also generate the single best search query to research this topic.",
    ),
    (
        "human",
        "Topic: {topic}\nTone: {tone}\nTarget audience: {target_audience}\nTarget length: ~{word_count} words",
    ),
])


async def run(state: BlogState) -> dict:
    try:
        structured_llm = llm.with_structured_output(BlogOutline)
        outline = await (PROMPT | structured_llm).ainvoke({
            "topic": state["topic"],
            "tone": state["tone"],
            "target_audience": state["target_audience"],
            "word_count": state["target_word_count"],
        })
        logger.info("blog_planned", title=outline.title, sections=len(outline.sections))
        return {
            "blog_outline": outline.model_dump(),
            "question": state["topic"],
            "search_queries": [outline.search_query],
        }
    except Exception as e:
        logger.error("planning_failed", error=str(e))
        return {
            "blog_outline": None,
            "question": state["topic"],
            "search_queries": [state["topic"]],
            "error": str(e),
        }
