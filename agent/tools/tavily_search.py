import os
from tavily import TavilyClient


def search(query: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query, max_results=max_results)
    return [
        {
            "url": r["url"],
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]
