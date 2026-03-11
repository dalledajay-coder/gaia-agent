"""File handling tools for processing GAIA task attachments."""

import os
import base64
from claude_agent_sdk import tool
from typing import Any


@tool(
    "read_file",
    "Read the contents of a file. Supports text files, CSVs, and returns base64 for binary files.",
    {"file_path": str},
)
async def read_file(args: dict[str, Any]) -> dict[str, Any]:
    file_path = args["file_path"]

    if not os.path.exists(file_path):
        return {"content": [{"type": "text", "text": f"File not found: {file_path}"}]}

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in (".txt", ".csv", ".json", ".md", ".py", ".js", ".html", ".xml", ".yaml", ".yml", ".tsv", ".log"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 15000:
                content = content[:15000] + "\n... [truncated]"
            return {"content": [{"type": "text", "text": f"File: {file_path}\n\n{content}"}]}

        elif ext in (".xlsx", ".xls"):
            return await _read_excel(file_path)

        elif ext == ".pdf":
            return await _read_pdf(file_path)

        else:
            # Binary file - return info
            size = os.path.getsize(file_path)
            return {"content": [{"type": "text", "text": f"Binary file: {file_path} ({size} bytes, type: {ext})"}]}

    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error reading {file_path}: {str(e)}"}]}


async def _read_excel(file_path: str) -> dict[str, Any]:
    """Read Excel files using openpyxl or pandas."""
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-c", f"""
import csv, io, sys
try:
    import openpyxl
    wb = openpyxl.load_workbook('{file_path}', data_only=True)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"=== Sheet: {{sheet_name}} ===")
        for row in ws.iter_rows(values_only=True):
            print('\\t'.join(str(c) if c is not None else '' for c in row))
except ImportError:
    print("openpyxl not available")
"""],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout:
            return {"content": [{"type": "text", "text": f"Excel file: {file_path}\n\n{result.stdout[:15000]}"}]}
        return {"content": [{"type": "text", "text": f"Could not read Excel file: {result.stderr}"}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Excel read error: {str(e)}"}]}


async def _read_pdf(file_path: str) -> dict[str, Any]:
    """Read PDF files."""
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-c", f"""
try:
    import subprocess
    r = subprocess.run(['pdftotext', '{file_path}', '-'], capture_output=True, text=True, timeout=30)
    if r.stdout:
        print(r.stdout)
    else:
        # Try PyPDF2
        from PyPDF2 import PdfReader
        reader = PdfReader('{file_path}')
        for page in reader.pages:
            text = page.extract_text()
            if text:
                print(text)
except Exception as e:
    print(f"PDF read error: {{e}}")
"""],
            capture_output=True, text=True, timeout=30
        )
        text = result.stdout if result.stdout else f"Could not extract text: {result.stderr}"
        if len(text) > 15000:
            text = text[:15000] + "\n... [truncated]"
        return {"content": [{"type": "text", "text": f"PDF file: {file_path}\n\n{text}"}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"PDF read error: {str(e)}"}]}
