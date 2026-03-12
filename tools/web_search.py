"""Web search and browsing tools for the GAIA agent."""

import aiohttp
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from claude_agent_sdk import tool
from typing import Any


@tool(
    "web_search",
    "Search the web using a query string. Returns relevant search results with titles, URLs, and snippets. Tries multiple search engines for better coverage.",
    {"query": str, "num_results": int},
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    query_str = args["query"]
    num_results = args.get("num_results", 8)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    results = []

    # Try DuckDuckGo first
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query_str)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        for r in soup.select(".result")[:num_results]:
            title_el = r.select_one(".result__title a")
            snippet_el = r.select_one(".result__snippet")
            if title_el:
                href = title_el.get("href", "")
                # DuckDuckGo wraps URLs in redirects, extract actual URL
                if "uddg=" in href:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(href)
                    params = parse_qs(parsed.query)
                    href = params.get("uddg", [href])[0]
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": href,
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
    except Exception as e:
        pass  # Will try Google fallback

    # If DuckDuckGo returned few results, try Google
    if len(results) < 3:
        try:
            google_url = f"https://www.google.com/search?q={quote_plus(query_str)}&num={num_results}"
            async with aiohttp.ClientSession() as session:
                async with session.get(google_url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            for div in soup.select("div.g")[:num_results]:
                a_tag = div.select_one("a[href]")
                if a_tag and a_tag.get("href", "").startswith("http"):
                    title = a_tag.get_text(strip=True)
                    href = a_tag["href"]
                    # Skip if already in results
                    if any(r["url"] == href for r in results):
                        continue
                    snippet_div = div.select_one(".VwiC3b, .IsZvec, span.st")
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet,
                    })
        except Exception:
            pass

    if not results:
        return {"content": [{"type": "text", "text": f"No results found for: {query_str}. Try rephrasing your query or using wikipedia_search instead."}]}

    text = f"Search results for '{query_str}':\n\n"
    for i, r in enumerate(results[:num_results], 1):
        text += f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"

    return {"content": [{"type": "text", "text": text}]}


@tool(
    "web_browse",
    "Fetch and extract text content from a URL. Returns the main text content of the page. Use this to read full articles, documentation, or data from web pages found via search.",
    {"url": str, "extract_tables": bool},
)
async def web_browse(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    extract_tables = args.get("extract_tables", False)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), allow_redirects=True) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return {"content": [{"type": "text", "text": f"Non-text content type: {content_type}. URL: {url}"}]}
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/nav elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Extract tables if requested
        table_text = ""
        if extract_tables:
            tables = soup.find_all("table")
            for idx, table in enumerate(tables[:5]):
                rows = table.find_all("tr")
                table_text += f"\n[Table {idx+1}]\n"
                for row in rows:
                    cells = row.find_all(["th", "td"])
                    table_text += " | ".join(c.get_text(strip=True) for c in cells) + "\n"

        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Truncate smartly
        if len(text) > 12000:
            text = text[:12000] + "\n... [truncated]"

        result = f"Content from {url}:\n\n{text}"
        if table_text:
            result += f"\n\n--- Tables ---{table_text}"

        return {"content": [{"type": "text", "text": result}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Browse error for {url}: {str(e)}. Try a different URL or search query."}]}
