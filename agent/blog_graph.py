from langgraph.graph import StateGraph, END
from agent.blog_state import BlogState
from agent.nodes import search, read_pages, score_relevance, decide_next
from agent.nodes import plan_blog, write_blog, blog_guardrails


def _route_after_plan(state: BlogState) -> str:
    if state.get("error"):
        return END
    return "search"


def _route_after_search(state: BlogState) -> str:
    if state.get("error"):
        return END
    return "read_pages"


def _route_after_decide(state: BlogState) -> str:
    if state.get("should_continue"):
        return "search"
    return "write_blog"


def _route_after_guardrails(state: BlogState) -> str:
    if state.get("guardrail_passed"):
        return END
    if state.get("retry_count", 0) >= 1:
        return END  # fail open
    return "write_blog"


def build_blog_graph() -> StateGraph:
    g = StateGraph(BlogState)

    g.add_node("plan_blog",        plan_blog.run)
    g.add_node("search",           search.run)
    g.add_node("read_pages",       read_pages.run)
    g.add_node("score_relevance",  score_relevance.run)
    g.add_node("decide_next",      decide_next.run)
    g.add_node("write_blog",       write_blog.run)
    g.add_node("blog_guardrails",  blog_guardrails.run)

    g.set_entry_point("plan_blog")
    g.add_conditional_edges("plan_blog",       _route_after_plan)
    g.add_conditional_edges("search",          _route_after_search)
    g.add_edge("read_pages",       "score_relevance")
    g.add_edge("score_relevance",  "decide_next")
    g.add_conditional_edges("decide_next",     _route_after_decide)
    g.add_edge("write_blog",       "blog_guardrails")
    g.add_conditional_edges("blog_guardrails", _route_after_guardrails)

    return g.compile()
