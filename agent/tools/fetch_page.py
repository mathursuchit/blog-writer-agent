import re
import httpx

MAX_PAGE_CHARS = 8_000


async def fetch_page(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; BlogWriterAgent/1.0)"},
            )
            resp.raise_for_status()
            text = resp.text
    except Exception as e:
        return {"url": url, "content": "", "error": str(e)}

    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()

    return {"url": url, "content": clean[:MAX_PAGE_CHARS], "error": None}
