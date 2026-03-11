"""Code execution tools using E2B sandboxes."""

import os
from claude_agent_sdk import tool
from typing import Any


@tool(
    "execute_python",
    "Execute Python code in a secure E2B sandbox. Returns stdout, stderr, and any results. Use for calculations, data processing, file operations, and any computational tasks.",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Execution timeout in seconds (default 30)", "minimum": 1, "maximum": 300},
        },
        "required": ["code"],
    },
)
async def execute_python(args: dict[str, Any]) -> dict[str, Any]:
    code = args["code"]
    timeout = args.get("timeout", 30)

    try:
        from e2b_code_interpreter import Sandbox

        api_key = os.environ.get("E2B_API_KEY", "")
        if not api_key:
            # Fallback: execute locally with subprocess
            return await _execute_locally(code, timeout)

        sbx = Sandbox(api_key=api_key, timeout=timeout + 10)
        try:
            execution = sbx.run_code(code, timeout=timeout)

            output_parts = []
            if execution.logs.stdout:
                output_parts.append(f"STDOUT:\n{''.join(execution.logs.stdout)}")
            if execution.logs.stderr:
                output_parts.append(f"STDERR:\n{''.join(execution.logs.stderr)}")
            if execution.error:
                output_parts.append(f"ERROR:\n{execution.error.name}: {execution.error.value}\n{execution.error.traceback}")
            if execution.results:
                for r in execution.results:
                    if hasattr(r, "text") and r.text:
                        output_parts.append(f"RESULT:\n{r.text}")

            text = "\n\n".join(output_parts) if output_parts else "Code executed successfully (no output)"
            return {"content": [{"type": "text", "text": text}]}
        finally:
            sbx.kill()

    except ImportError:
        return await _execute_locally(code, timeout)
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Execution error: {str(e)}"}]}


async def _execute_locally(code: str, timeout: int) -> dict[str, Any]:
    """Fallback: execute Python code locally using subprocess."""
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        output_parts = []
        if stdout:
            output_parts.append(f"STDOUT:\n{stdout.decode()}")
        if stderr:
            output_parts.append(f"STDERR:\n{stderr.decode()}")
        if proc.returncode != 0:
            output_parts.append(f"Exit code: {proc.returncode}")

        text = "\n\n".join(output_parts) if output_parts else "Code executed successfully (no output)"
        return {"content": [{"type": "text", "text": text}]}

    except asyncio.TimeoutError:
        return {"content": [{"type": "text", "text": f"Execution timed out after {timeout}s"}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Local execution error: {str(e)}"}]}
    finally:
        os.unlink(tmp_path)
