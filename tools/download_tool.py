"""File download tool for fetching remote resources."""

import os
import aiohttp
from claude_agent_sdk import tool
from typing import Any


@tool(
    "download_file",
    "Download a file from a URL and save it locally. Returns the local file path. Useful for downloading attachments, datasets, images, audio files.",
    {"url": str, "save_as": str},
)
async def download_file(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    save_as = args.get("save_as", "")

    if not save_as:
        # Generate filename from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        save_as = os.path.join("/tmp", os.path.basename(parsed.path) or "downloaded_file")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    return {"content": [{"type": "text", "text": f"Download failed: HTTP {resp.status}"}]}

                content = await resp.read()
                os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)
                with open(save_as, "wb") as f:
                    f.write(content)

                size = len(content)
                return {"content": [{"type": "text", "text": f"Downloaded {size} bytes to {save_as}"}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Download error: {str(e)}"}]}
