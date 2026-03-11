"""Vision tool for analyzing images using Claude's multimodal capabilities."""

import os
import base64
from claude_agent_sdk import tool
from typing import Any


@tool(
    "analyze_image",
    "Analyze an image file using Claude's vision capabilities. Describe what's in the image, read text/numbers, interpret charts/diagrams. Provide a specific question about what to look for.",
    {"file_path": str, "question": str},
)
async def analyze_image(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["file_path"]
    question = args.get("question", "Describe this image in detail. If there is text, transcribe it exactly.")

    if not os.path.exists(file_path):
        return {"content": [{"type": "text", "text": f"Image not found: {file_path}"}]}

    try:
        import anthropic

        # Read and encode image
        with open(file_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(file_path)[1].lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        media_type = media_type_map.get(ext, "image/png")

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text
        return {"content": [{"type": "text", "text": f"Image analysis of {file_path}:\n\n{response_text}"}]}

    except ImportError:
        return {"content": [{"type": "text", "text": "anthropic package not available for vision analysis"}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Vision analysis error: {str(e)}"}]}
