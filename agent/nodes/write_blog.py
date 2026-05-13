import structlog
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from agent.blog_state import BlogState

logger = structlog.get_logger()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert blog writer. Write a complete, well-structured blog post in Markdown.\n\n"
        "Rules:\n"
        "- Write EVERY section from the outline — do not skip or merge any\n"
        "- Each section must be at least 2-3 full paragraphs, not bullet points\n"
        "- Cover every key point listed under each section heading\n"
        "- Cite sources inline as [Source N] where relevant\n"
        "- End with a '## References' section listing each cited source as '- [Source N]: URL'\n"
        "- Only cite URLs from the provided sources — do not invent any\n"
        "- Hit the target word count — do not stop early\n\n"
        "Tone: {tone}. Target audience: {target_audience}.",
    ),
    (
        "human",
        "Outline:\n{outline}\n\n"
        "Target word count: ~{word_count} words\n\n"
        "Sources:\n{sources}",
    ),
])


async def run(state: BlogState) -> dict:
    if state.get("error"):
        return {"blog_post": None}

    outline = state.get("blog_outline")
    if not outline:
        return {"blog_post": None, "error": "No outline available."}

    scored = state.get("scored_sources", [])
    good_sources = [s for s in scored if s.get("relevance_score", 0) >= 0.4]

    if not good_sources:
        return {"blog_post": None, "error": "No relevant sources to write from."}

    outline_text = f"Title: {outline['title']}\n\nIntro hook: {outline['intro_hook']}\n\n"
    for section in outline["sections"]:
        outline_text += f"## {section['heading']}\n"
        for kp in section.get("key_points", []):
            outline_text += f"- {kp}\n"
        outline_text += "\n"

    sources_text = "\n\n".join(
        f"[Source {i + 1}] {s.get('title', '')}\nURL: {s['url']}\n{s['content'][:600]}"
        for i, s in enumerate(good_sources)
    )

    try:
        response = await (PROMPT | llm).ainvoke({
            "tone": state.get("tone", "technical"),
            "target_audience": state.get("target_audience", "developers"),
            "outline": outline_text,
            "word_count": state.get("target_word_count", 1000),
            "sources": sources_text,
        })
        post = response.content if hasattr(response, "content") else str(response)
        logger.info("blog_written", words=len(post.split()))
        return {"blog_post": post}
    except Exception as e:
        logger.error("writing_failed", error=str(e))
        return {"blog_post": None, "error": str(e)}
