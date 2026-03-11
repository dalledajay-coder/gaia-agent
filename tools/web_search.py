"""Web search and browsing tools for the GAIA agent."""

import aiohttp
from bs4 import BeautifulSoup
from claude_agent_sdk import tool
from typing import Any


@tool(
    "web_search",
    "Search the web using a query string. Returns relevant search results with titles, URLs, and snippets.",
    {"query": str, "num_results": int},
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    num_results = args.get("num_results", 5)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        # Use DuckDuckGo HTML search
        url = f"https://html.duckduckgo.com/html/?q={query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        results = []

        for r in soup.select(".result")[:num_results]:
            title_el = r.select_one(".result__title a")
            snippet_el = r.select_one(".result__snippet")
            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": title_el.get("href", ""),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })

        if not results:
            return {"content": [{"type": "text", "text": f"No results found for: {query}"}]}

        text = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            text += f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"

        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Search error: {str(e)}"}]}


@tool(
    "web_browse",
    "Fetch and extract text content from a URL. Returns the main text content of the page.",
    {"url": str},
)
async def web_browse(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return {"content": [{"type": "text", "text": f"Non-text content type: {content_type}"}]}
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Truncate to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "\n... [truncated]"

        return {"content": [{"type": "text", "text": f"Content from {url}:\n\n{text}"}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Browse error for {url}: {str(e)}"}]}
