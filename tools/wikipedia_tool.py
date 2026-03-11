"""Wikipedia lookup tool for factual questions."""

import aiohttp
from claude_agent_sdk import tool
from typing import Any


@tool(
    "wikipedia_lookup",
    "Look up a Wikipedia article by title. Returns the article summary and key sections. Use this for factual questions about people, places, events, concepts.",
    {"title": str, "sentences": int},
)
async def wikipedia_lookup(args: dict[str, Any]) -> dict[str, Any]:
    title = args["title"]
    sentences = args.get("sentences", 10)

    try:
        # Use Wikipedia API for structured data
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    extract = data.get("extract", "")
                    title_found = data.get("title", title)

                    # Also get more detailed content
                    detail_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title.replace(' ', '_')}&prop=extracts&exintro=false&explaintext=true&exsectionformat=plain&format=json"
                    async with session.get(detail_url, timeout=aiohttp.ClientTimeout(total=10)) as detail_resp:
                        if detail_resp.status == 200:
                            detail_data = await detail_resp.json()
                            pages = detail_data.get("query", {}).get("pages", {})
                            for page_id, page in pages.items():
                                full_extract = page.get("extract", "")
                                if full_extract and len(full_extract) > len(extract):
                                    extract = full_extract

                    if len(extract) > 10000:
                        extract = extract[:10000] + "\n... [truncated]"

                    return {"content": [{"type": "text", "text": f"Wikipedia: {title_found}\n\n{extract}"}]}

                elif resp.status == 404:
                    # Try search
                    search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={title}&limit=5&format=json"
                    async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as search_resp:
                        if search_resp.status == 200:
                            search_data = await search_resp.json()
                            if len(search_data) > 1 and search_data[1]:
                                suggestions = ", ".join(search_data[1][:5])
                                return {"content": [{"type": "text", "text": f"Article '{title}' not found. Similar articles: {suggestions}"}]}
                    return {"content": [{"type": "text", "text": f"Wikipedia article not found: {title}"}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Wikipedia error: {str(e)}"}]}


@tool(
    "wikipedia_search",
    "Search Wikipedia for articles matching a query. Returns article titles and snippets.",
    {"query": str, "num_results": int},
)
async def wikipedia_search(args: dict[str, Any]) -> dict[str, Any]:
    query_str = args["query"]
    num_results = args.get("num_results", 5)

    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query_str}&srlimit={num_results}&format=json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("query", {}).get("search", [])

                    if not results:
                        return {"content": [{"type": "text", "text": f"No Wikipedia results for: {query_str}"}]}

                    text = f"Wikipedia search results for '{query_str}':\n\n"
                    for i, r in enumerate(results, 1):
                        import re
                        snippet = re.sub(r'<[^>]+>', '', r.get("snippet", ""))
                        text += f"{i}. {r['title']}\n   {snippet}\n\n"

                    return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Wikipedia search error: {str(e)}"}]}
