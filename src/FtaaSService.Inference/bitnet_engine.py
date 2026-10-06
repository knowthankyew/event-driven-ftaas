"""
BitNet C++ Native Inference Engine Adapter for FtaaSService.Inference.

Provides non-blocking, native CPU inference for Microsoft BitNet b1.58 ternary models
by wrapping the compiled `llama-cli` / `bitnet.cpp` binary.
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("FtaaSService.Inference.BitNet")

# Default binary and model paths with environment variable overrides
DEFAULT_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-cli"
DEFAULT_MODEL_PATH = Path.home() / "Github" / "BitNet" / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"

DEFAULT_EMBED_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-embedding"
DEFAULT_EMBED_MODEL_PATH = Path.home() / "Github" / "BitNet" / "models" / "bitnet-embedding-270m" / "bitnet-embeddings-270m-bf16-i2_s.gguf"

BITNET_CLI_PATH = Path(os.getenv("BITNET_CLI_PATH", str(DEFAULT_CLI_PATH))).resolve()
BITNET_MODEL_PATH = Path(os.getenv("BITNET_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
BITNET_EMBED_CLI_PATH = Path(os.getenv("BITNET_EMBED_CLI_PATH", str(DEFAULT_EMBED_CLI_PATH))).resolve()
BITNET_EMBED_MODEL_PATH = Path(os.getenv("BITNET_EMBED_MODEL_PATH", str(DEFAULT_EMBED_MODEL_PATH))).resolve()
BITNET_THREADS = int(os.getenv("BITNET_THREADS", "4"))
BITNET_TIMEOUT_SECONDS = float(os.getenv("BITNET_TIMEOUT_SECONDS", "60.0"))


def is_bitnet_model(model_name: Optional[str]) -> bool:
    """Return True if the specified model identifier corresponds to a BitNet architecture."""
    if not model_name:
        return False
    return "bitnet" in model_name.lower()


def is_bitnet_available() -> bool:
    """Check whether the native bitnet binary and gguf weights are accessible on the host."""
    cli_ok = BITNET_CLI_PATH.is_file() and os.access(str(BITNET_CLI_PATH), os.X_OK)
    model_ok = BITNET_MODEL_PATH.is_file()
    return cli_ok and model_ok


def get_bitnet_status() -> dict:
    """Return diagnostic telemetry regarding BitNet binary and weight accessibility."""
    cli_exists = BITNET_CLI_PATH.is_file()
    cli_exec = cli_exists and os.access(str(BITNET_CLI_PATH), os.X_OK)
    model_exists = BITNET_MODEL_PATH.is_file()
    model_size_mb = round(BITNET_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if model_exists else 0.0

    embed_cli_exists = BITNET_EMBED_CLI_PATH.is_file()
    embed_cli_exec = embed_cli_exists and os.access(str(BITNET_EMBED_CLI_PATH), os.X_OK)
    embed_model_exists = BITNET_EMBED_MODEL_PATH.is_file()
    embed_model_size_mb = round(BITNET_EMBED_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if embed_model_exists else 0.0

    return {
        "available": cli_exec and model_exists,
        "cliPath": str(BITNET_CLI_PATH),
        "cliExecutable": cli_exec,
        "modelPath": str(BITNET_MODEL_PATH),
        "modelPresent": model_exists,
        "modelSizeMb": model_size_mb,
        "embedAvailable": embed_cli_exec and embed_model_exists,
        "embedCliPath": str(BITNET_EMBED_CLI_PATH),
        "embedCliExecutable": embed_cli_exec,
        "embedModelPath": str(BITNET_EMBED_MODEL_PATH),
        "embedModelPresent": embed_model_exists,
        "embedModelSizeMb": embed_model_size_mb,
        "threads": BITNET_THREADS
    }


def parse_llama_cli_output(raw_output: str, prompt: str) -> str:
    """
    Extract the clean model generation output from the raw llama-cli stdout stream.
    Strips out banners, prompt echoes, interactive prompts, and runtime footer telemetry.
    """
    text = raw_output

    # Strip runtime performance metrics footer
    if "[ Prompt:" in text:
        text = text.split("[ Prompt:")[0]
    elif "Exiting..." in text:
        text = text.split("Exiting...")[0]

    # Isolate assistant response from prompt echo
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[1]
    elif prompt in text:
        text = text.split(prompt, 1)[1]
    elif "> " in text:
        parts = text.split("> ")
        text = parts[-1]

    # Clean up whitespace and special markers
    text = text.strip()
    return text


async def generate_bitnet(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    adapter_path: Optional[str] = None
) -> Tuple[str, float]:
    """
    Execute asynchronous native BitNet C++ inference via llama-cli.

    Args:
        prompt: Raw input text from the user or Arena.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        adapter_path: Optional path to a GGUF LoRA adapter file.

    Returns:
        tuple[str, float]: (generated_text, elapsed_ms)
    """
    if not is_bitnet_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ native runtime is not available: "
            f"cli_exec={status['cliExecutable']} ({status['cliPath']}), "
            f"model_present={status['modelPresent']} ({status['modelPath']}). "
            f"Set BITNET_CLI_PATH and BITNET_MODEL_PATH environment variables."
        )

    # Format using BitNet prompt template (User: ...<|eot_id|>\nAssistant: )
    formatted_prompt = f"User: {prompt}<|eot_id|>\nAssistant:"

    cmd = [
        str(BITNET_CLI_PATH),
        "-m", str(BITNET_MODEL_PATH),
        "-p", formatted_prompt,
        "-n", str(max(1, max_tokens)),
        "-t", str(max(1, BITNET_THREADS)),
        "--temp", str(max(0.0, temperature)),
        "-ngl", "0",
        "-st",
        "--simple-io",
        "--no-display-prompt",
        "--log-disable"
    ]

    # Attach GGUF LoRA adapter if provided and accessible
    if adapter_path:
        p = Path(adapter_path)
        if p.is_file():
            cmd.extend(["--lora", str(p)])
        elif p.is_dir() and (p / "adapter.gguf").is_file():
            cmd.extend(["--lora", str(p / "adapter.gguf")])
        else:
            logger.warning(f"Requested adapter at '{adapter_path}' does not contain a GGUF file; running base BitNet")

    start_t = time.perf_counter()
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            ),
            timeout=BITNET_TIMEOUT_SECONDS
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=BITNET_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(f"BitNet inference timed out after {BITNET_TIMEOUT_SECONDS}s")
        try:
            proc.kill()
        except Exception:
            pass
        raise TimeoutError(f"BitNet inference process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace").strip()
        logger.error(f"BitNet inference process failed (exit {proc.returncode}): {err_msg}")
        raise RuntimeError(f"BitNet inference exited with code {proc.returncode}: {err_msg}")

    raw_output = stdout.decode("utf-8", errors="replace")
    completion_text = parse_llama_cli_output(raw_output, formatted_prompt)

    return completion_text, elapsed_ms


def generate_bitnet_sync(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    adapter_path: Optional[str] = None
) -> Tuple[str, float]:
    """
    Synchronous execution of native BitNet C++ inference via llama-cli.

    Args:
        prompt: Raw input text from the user or Arena.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        adapter_path: Optional path to a GGUF LoRA adapter file.

    Returns:
        tuple[str, float]: (generated_text, elapsed_ms)
    """
    import subprocess

    if not is_bitnet_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ native runtime is not available: "
            f"cli_exec={status['cliExecutable']} ({status['cliPath']}), "
            f"model_present={status['modelPresent']} ({status['modelPath']}). "
            f"Set BITNET_CLI_PATH and BITNET_MODEL_PATH environment variables."
        )

    formatted_prompt = f"User: {prompt}<|eot_id|>\nAssistant:"

    cmd = [
        str(BITNET_CLI_PATH),
        "-m", str(BITNET_MODEL_PATH),
        "-p", formatted_prompt,
        "-n", str(max(1, max_tokens)),
        "-t", str(max(1, BITNET_THREADS)),
        "--temp", str(max(0.0, temperature)),
        "-ngl", "0",
        "-st",
        "--simple-io",
        "--no-display-prompt",
        "--log-disable"
    ]

    if adapter_path:
        p = Path(adapter_path)
        if p.is_file():
            cmd.extend(["--lora", str(p)])
        elif p.is_dir() and (p / "adapter.gguf").is_file():
            cmd.extend(["--lora", str(p / "adapter.gguf")])
        else:
            logger.warning(f"Requested adapter at '{adapter_path}' does not contain a GGUF file; running base BitNet")

    start_t = time.perf_counter()
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BITNET_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        logger.error(f"BitNet inference timed out after {BITNET_TIMEOUT_SECONDS}s")
        raise TimeoutError(f"BitNet inference process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if res.returncode != 0:
        logger.error(f"BitNet inference process failed (exit {res.returncode}): {res.stderr}")
        raise RuntimeError(f"BitNet inference exited with code {res.returncode}: {res.stderr}")

    completion_text = parse_llama_cli_output(res.stdout, formatted_prompt)
    return completion_text, elapsed_ms



def is_bitnet_embed_available() -> bool:
    '''Check whether the native llama-embedding binary and gguf weights are accessible on the host.'''
    cli_ok = BITNET_EMBED_CLI_PATH.is_file() and os.access(str(BITNET_EMBED_CLI_PATH), os.X_OK)
    model_ok = BITNET_EMBED_MODEL_PATH.is_file()
    return cli_ok and model_ok

async def embed_bitnet(prompt: str) -> Tuple[list[float], float]:
    '''Asynchronous execution of native BitNet C++ embedding via llama-embedding.'''
    import tempfile
    
    if not is_bitnet_embed_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ embedding runtime is not available: "
            f"cli_exec={status['embedCliExecutable']} ({status['embedCliPath']}), "
            f"model_present={status['embedModelPresent']} ({status['embedModelPath']})."
        )

    with tempfile.NamedTemporaryFile("w+", delete=True) as tf:
        tf.write(prompt)
        tf.flush()
        
        cmd = [
            str(BITNET_EMBED_CLI_PATH),
            "-m", str(BITNET_EMBED_MODEL_PATH),
            "-t", str(max(1, BITNET_THREADS)),
            "-c", "4096",
            "--pooling", "last",
            "--embd-normalize", "2",
            "--embd-output-format", "array",
            "-ngl", "0",
            "-f", tf.name
        ]

        start_t = time.perf_counter()
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=BITNET_TIMEOUT_SECONDS
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=BITNET_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error(f"BitNet embedding timed out after {BITNET_TIMEOUT_SECONDS}s")
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutError(f"BitNet embedding process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
        stdout_str = stdout_bytes.decode('utf-8', errors='replace')

        if proc.returncode != 0:
            err_msg = stderr_bytes.decode('utf-8', errors='replace').strip()
            logger.error(f"BitNet embedding process failed (exit {proc.returncode}): {err_msg}")
            raise RuntimeError(f"BitNet embedding exited with code {proc.returncode}: {err_msg}")

    return _parse_embed_output(stdout_str, elapsed_ms)

def embed_bitnet_sync(prompt: str) -> Tuple[list[float], float]:
    '''Synchronous execution of native BitNet C++ embedding via llama-embedding.'''
    import subprocess
    import tempfile

    if not is_bitnet_embed_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ embedding runtime is not available: "
            f"cli_exec={status['embedCliExecutable']} ({status['embedCliPath']}), "
            f"model_present={status['embedModelPresent']} ({status['embedModelPath']})."
        )

    with tempfile.NamedTemporaryFile("w+", delete=True) as tf:
        tf.write(prompt)
        tf.flush()
        
        cmd = [
            str(BITNET_EMBED_CLI_PATH),
            "-m", str(BITNET_EMBED_MODEL_PATH),
            "-t", str(max(1, BITNET_THREADS)),
            "-c", "4096",
            "--pooling", "last",
            "--embd-normalize", "2",
            "--embd-output-format", "array",
            "-ngl", "0",
            "-f", tf.name
        ]

        start_t = time.perf_counter()
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BITNET_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            logger.error(f"BitNet embedding timed out after {BITNET_TIMEOUT_SECONDS}s")
            raise TimeoutError(f"BitNet embedding process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        if res.returncode != 0:
            logger.error(f"BitNet embedding process failed (exit {res.returncode}): {res.stderr}")
            raise RuntimeError(f"BitNet embedding exited with code {res.returncode}: {res.stderr}")

    return _parse_embed_output(res.stdout, elapsed_ms)

def _parse_embed_output(stdout_str: str, elapsed_ms: float) -> Tuple[list[float], float]:
    import json
    match = re.search(r'\[\s*\[\s*-?\d+\.?\d*', stdout_str)
    if not match:
        raise ValueError("No JSON array found in output.")
        
    start_idx = match.start()
    end_idx = stdout_str.rfind(']]')
    if end_idx == -1 or end_idx < start_idx:
        raise ValueError("Malformed JSON array in output.")
        
    array_str = stdout_str[start_idx:end_idx+2]
    
    try:
        embeddings = json.loads(array_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON array: {e}")
        
    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
        embeddings = embeddings[0]
        
    if len(embeddings) != 640:
        raise ValueError(f"Expected embedding dimension 640, got {len(embeddings)}")
        
    return embeddings, elapsed_ms
