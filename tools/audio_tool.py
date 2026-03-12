"""Audio transcription tool using Whisper."""

import os
import subprocess
from claude_agent_sdk import tool
from typing import Any


@tool(
    "transcribe_audio",
    "Transcribe speech from an audio or video file using Whisper. Returns the full transcript text. Supports mp3, wav, m4a, mp4, webm, ogg, flac formats.",
    {"file_path": str, "language": str},
)
async def transcribe_audio(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["file_path"]
    language = args.get("language", "en")

    if not os.path.exists(file_path):
        return {"content": [{"type": "text", "text": f"File not found: {file_path}"}]}

    try:
        # Use whisper CLI for transcription
        result = subprocess.run(
            [
                "whisper", file_path,
                "--model", "base",
                "--language", language,
                "--output_format", "txt",
                "--output_dir", "/tmp/whisper_out",
            ],
            capture_output=True, text=True, timeout=120
        )

        # Read the output text file
        basename = os.path.splitext(os.path.basename(file_path))[0]
        txt_path = f"/tmp/whisper_out/{basename}.txt"
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                transcript = f.read().strip()
            return {"content": [{"type": "text", "text": f"Transcript of {file_path}:\n\n{transcript}"}]}

        # Fallback: check stdout
        if result.stdout:
            return {"content": [{"type": "text", "text": f"Transcript:\n{result.stdout}"}]}

        return {"content": [{"type": "text", "text": f"Transcription produced no output. stderr: {result.stderr[:500]}"}]}

    except subprocess.TimeoutExpired:
        return {"content": [{"type": "text", "text": "Transcription timed out after 120s. Try a shorter audio file."}]}
    except FileNotFoundError:
        # Whisper CLI not found, try Python API
        try:
            result = subprocess.run(
                ["python3", "-c", f"""
import whisper
model = whisper.load_model("base")
result = model.transcribe("{file_path}", language="{language}")
print(result["text"])
"""],
                capture_output=True, text=True, timeout=120
            )
            if result.stdout:
                return {"content": [{"type": "text", "text": f"Transcript of {file_path}:\n\n{result.stdout.strip()}"}]}
            return {"content": [{"type": "text", "text": f"Transcription error: {result.stderr[:500]}"}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Transcription error: {str(e)}"}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Transcription error: {str(e)}"}]}
