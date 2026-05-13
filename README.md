# Blog Writer Agent

Give it a topic — it plans an outline, researches the web, and writes a full cited blog post. Built with LangGraph.

## How it works

```
topic + settings
      |
  [plan_blog] ── llama-3.1-8b generates title, intro hook, section outline + search query
      |
  [search] ──── Tavily web search
      |
  [read_pages] ── fetch + clean top pages in parallel
      |
  [score_relevance] ── llama-3.1-8b scores each source 0-1
      |
  [decide_next] ── enough sources? write : search again (max_depth / token_budget hard stops)
      |
  [write_blog] ── llama-3.3-70b writes full Markdown post with inline citations
      |
  [blog_guardrails] ── drops any URL in the post not in actual scored sources
      |
  final .md post
```

## Production features

- **Prompt injection guard** — topic scanned before any search
- **Citation hallucination check** — every URL in the post verified against actual sources
- **Source trust scoring** — arxiv, GitHub, .edu, .gov scored higher; ad trackers scored lower
- **Token budget** — hard stop at 50K tokens, enforced before LLM calls
- **Model routing** — llama-3.1-8b-instant for fast planning/scoring, llama-3.3-70b-versatile for writing

## Run locally

```bash
cp .env.example .env
# fill in GROQ_API_KEY and TAVILY_API_KEY
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Streamlit Community Cloud → New app → `app.py` → add secrets:

```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```

## Stack

LangGraph, LangChain, Groq (Llama 3), Tavily, Streamlit, httpx, Pydantic v2, structlog
