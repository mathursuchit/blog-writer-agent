import structlog
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from agent.blog_state import BlogState

logger = structlog.get_logger()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class BlogSection(BaseModel):
    heading: str
    key_points: list[str]  # must have at least 3 items


class BlogOutline(BaseModel):
    title: str
    intro_hook: str
    sections: list[BlogSection]  # must have at least 5 items
    search_query: str


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a blog strategist. Create a detailed blog outline with:\n"
        "- A compelling title\n"
        "- A one-sentence intro hook that grabs attention\n"
        "- EXACTLY 5-6 sections (never fewer than 5)\n"
        "- Each section must have EXACTLY 3-4 key points — be specific, not generic\n"
        "- The best single search query to research this topic\n\n"
        "Do not produce fewer than 5 sections. Do not produce fewer than 3 key points per section.",
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
