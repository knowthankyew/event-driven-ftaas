"""
BitNet C++ Native Inference Engine Adapter for FtaaSService.Inference.

Provides non-blocking, native CPU inference for Microsoft BitNet b1.58 ternary models
by wrapping the compiled `llama-cli` / `bitnet.cpp` binary.
"""

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("FtaaSService.Inference.BitNet")

# Default binary and model paths with environment variable overrides
DEFAULT_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-cli"
DEFAULT_COMPLETION_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-completion"
DEFAULT_MODEL_PATH = Path.home() / "Github" / "BitNet" / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"

DEFAULT_EMBED_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-embedding"
DEFAULT_EMBED_MODEL_PATH = Path.home() / "Github" / "BitNet" / "models" / "bitnet-embedding-270m" / "bitnet-embeddings-270m-bf16-i2_s.gguf"
DEFAULT_TOKENIZE_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-tokenize"

BITNET_CLI_PATH = Path(os.getenv("BITNET_CLI_PATH", str(DEFAULT_CLI_PATH))).resolve()
BITNET_COMPLETION_CLI_PATH = Path(os.getenv("BITNET_COMPLETION_CLI_PATH", str(DEFAULT_COMPLETION_CLI_PATH))).resolve()
BITNET_MODEL_PATH = Path(os.getenv("BITNET_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
BITNET_EMBED_CLI_PATH = Path(os.getenv("BITNET_EMBED_CLI_PATH", str(DEFAULT_EMBED_CLI_PATH))).resolve()
BITNET_EMBED_MODEL_PATH = Path(os.getenv("BITNET_EMBED_MODEL_PATH", str(DEFAULT_EMBED_MODEL_PATH))).resolve()
BITNET_TOKENIZE_CLI_PATH = Path(os.getenv("BITNET_TOKENIZE_CLI_PATH", str(DEFAULT_TOKENIZE_CLI_PATH))).resolve()
BITNET_THREADS = int(os.getenv("BITNET_THREADS", "4"))
BITNET_TIMEOUT_SECONDS = float(os.getenv("BITNET_TIMEOUT_SECONDS", "60.0"))
DEV_STDIN = Path("/dev/stdin")

# Maximum safe token context length for bitnet-embedding on AVX2 (kernel batch decode ceiling is 256; clamped at 255)
MAX_SAFE_EMBED_TOKENS = min(int(os.getenv("MAX_SAFE_EMBED_TOKENS", "240")), 255)

# Maximum supported prompt character length for generation (~4,096 tokens at ~4 chars/token)
MAX_GENERATE_PROMPT_CHARS = int(os.getenv("MAX_GENERATE_PROMPT_CHARS", "16384"))

SANDBOX_EXEC_PATH = Path("/usr/bin/sandbox-exec")
BITNET_SANDBOX_NETWORK_DENY = os.getenv("BITNET_SANDBOX_NETWORK_DENY", "false").lower() in ("true", "1")


def wrap_sandboxed_cmd(cmd: list[str]) -> list[str]:
    """Wrap command with macOS sandbox-exec to kernel-enforce zero network egress if enabled."""
    if BITNET_SANDBOX_NETWORK_DENY and sys.platform == "darwin" and SANDBOX_EXEC_PATH.is_file():
        return [str(SANDBOX_EXEC_PATH), "-p", "(version 1)(allow default)(deny network*)"] + cmd
    return cmd


def is_tokenizer_available() -> bool:
    """Check whether the native llama-tokenize CLI and embedding weights are present and executable."""
    return (
        BITNET_TOKENIZE_CLI_PATH.is_file()
        and os.access(str(BITNET_TOKENIZE_CLI_PATH), os.X_OK)
        and BITNET_EMBED_MODEL_PATH.is_file()
    )


class ContextOverflowError(Exception):
    """Raised when input prompt exceeds the BitNet model's context or batch memory capacity."""
    pass


class BitNetExecutionError(Exception):
    """Raised when the native BitNet binary fails, crashes, or produces malformed output."""
    pass


class TokenizerUnavailableError(BitNetExecutionError):
    """Raised when the tokenizer pre-check cannot verify token count for long prompts."""
    pass


def count_embed_tokens(prompt: str) -> Optional[int]:
    """
    Count prompt tokens using native llama-tokenize against bitnet-embedding weights.
    Returns integer token count if CLI is available, otherwise None.
    """
    import subprocess
    if not (BITNET_TOKENIZE_CLI_PATH.is_file() and os.access(str(BITNET_TOKENIZE_CLI_PATH), os.X_OK) and BITNET_EMBED_MODEL_PATH.is_file()):
        return None

    cmd = wrap_sandboxed_cmd([
        str(BITNET_TOKENIZE_CLI_PATH),
        "-m", str(BITNET_EMBED_MODEL_PATH),
        "--show-count",
        "--stdin",
        "--log-disable"
    ])
    try:
        res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=5.0)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "Total number of tokens:" in line:
                    return int(line.split(":")[-1].strip())
    except Exception as e:
        logger.warning(f"llama-tokenize pre-check error: {e}")
    return None


async def count_embed_tokens_async(prompt: str) -> Optional[int]:
    """Asynchronously count prompt tokens using llama-tokenize against bitnet-embedding weights."""
    if not (BITNET_TOKENIZE_CLI_PATH.is_file() and os.access(str(BITNET_TOKENIZE_CLI_PATH), os.X_OK) and BITNET_EMBED_MODEL_PATH.is_file()):
        return None

    cmd = wrap_sandboxed_cmd([
        str(BITNET_TOKENIZE_CLI_PATH),
        "-m", str(BITNET_EMBED_MODEL_PATH),
        "--show-count",
        "--stdin",
        "--log-disable"
    ])
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            ),
            timeout=5.0
        )
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=5.0
        )
        if proc.returncode == 0:
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            for line in stdout_str.splitlines():
                if "Total number of tokens:" in line:
                    return int(line.split(":")[-1].strip())
    except Exception as e:
        logger.warning(f"async llama-tokenize pre-check error: {e}")
    return None



def is_completion_cli_available() -> bool:
    """Check whether native llama-completion binary is accessible and executable on the host."""
    return BITNET_COMPLETION_CLI_PATH.is_file() and os.access(str(BITNET_COMPLETION_CLI_PATH), os.X_OK)


def build_bitnet_generate_cmd(
    input_file: str = "/dev/stdin",
    max_tokens: int = 128,
    temperature: float = 0.7,
    adapter_path: Optional[str] = None,
    context_size: int = 4096
) -> list[str]:
    """Construct argument list for native BitNet inference, preferring llama-completion if available."""
    use_completion = is_completion_cli_available()
    cli_bin = str(BITNET_COMPLETION_CLI_PATH) if use_completion else str(BITNET_CLI_PATH)

    cmd = [
        cli_bin,
        "-m", str(BITNET_MODEL_PATH),
        "-c", str(max(128, context_size)),
        "-n", str(max(1, max_tokens)),
        "-t", str(max(1, BITNET_THREADS)),
        "--temp", str(max(0.0, temperature)),
        "-ngl", "0",
        "--no-display-prompt",
    ]
    if use_completion:
        cmd.extend(["-no-cnv"])
    else:
        cmd.extend(["-st", "--simple-io", "--log-disable"])

    cmd.extend(["-f", input_file])

    if adapter_path:
        p = Path(adapter_path)
        if p.is_file():
            cmd.extend(["--lora", str(p)])
        elif p.is_dir() and (p / "adapter.gguf").is_file():
            cmd.extend(["--lora", str(p / "adapter.gguf")])
        else:
            logger.warning(f"Requested adapter at '{adapter_path}' does not contain a GGUF file; running base BitNet")
    return wrap_sandboxed_cmd(cmd)


def build_bitnet_embed_cmd(
    input_file: str = "/dev/stdin",
    embd_separator: str = "<#sep#>"
) -> list[str]:
    """Construct argument list for native llama-embedding inference."""
    cmd = [
        str(BITNET_EMBED_CLI_PATH),
        "-m", str(BITNET_EMBED_MODEL_PATH),
        "-t", str(max(1, BITNET_THREADS)),
        "-c", "512",
        "--pooling", "last",
        "--embd-normalize", "2",
        "--embd-separator", embd_separator,
        "--embd-output-format", "array",
        "-ngl", "0",
        "-f", input_file
    ]
    return wrap_sandboxed_cmd(cmd)


def is_bitnet_model(model_name: Optional[str]) -> bool:
    """Return True if the specified model identifier corresponds to a BitNet architecture."""
    if not model_name:
        return False
    return "bitnet" in model_name.lower()


def is_bitnet_available() -> bool:
    """Check whether a native bitnet binary (llama-completion or llama-cli) and gguf weights are accessible on the host."""
    cli_ok = is_completion_cli_available() or (
        BITNET_CLI_PATH.is_file() and os.access(str(BITNET_CLI_PATH), os.X_OK)
    )
    model_ok = BITNET_MODEL_PATH.is_file()
    return cli_ok and model_ok


def get_bitnet_status() -> dict:
    """Return diagnostic telemetry regarding BitNet binary and weight accessibility."""
    cli_exists = BITNET_CLI_PATH.is_file()
    cli_exec = cli_exists and os.access(str(BITNET_CLI_PATH), os.X_OK)
    completion_exists = BITNET_COMPLETION_CLI_PATH.is_file()
    completion_exec = is_completion_cli_available()
    model_exists = BITNET_MODEL_PATH.is_file()
    model_size_mb = round(BITNET_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if model_exists else 0.0

    tok_available = is_tokenizer_available()
    embed_cli_exists = BITNET_EMBED_CLI_PATH.is_file()
    embed_cli_exec = embed_cli_exists and os.access(str(BITNET_EMBED_CLI_PATH), os.X_OK)
    embed_model_exists = BITNET_EMBED_MODEL_PATH.is_file()
    embed_model_size_mb = round(BITNET_EMBED_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if embed_model_exists else 0.0

    return {
        "available": (cli_exec or completion_exec) and model_exists,
        "cliPath": str(BITNET_CLI_PATH),
        "cliExecutable": cli_exec,
        "completionPath": str(BITNET_COMPLETION_CLI_PATH),
        "completionExecutable": completion_exec,
        "modelPath": str(BITNET_MODEL_PATH),
        "modelPresent": model_exists,
        "modelSizeMb": model_size_mb,
        "tokenizerAvailable": tok_available,
        "embedAvailable": embed_cli_exec and embed_model_exists and tok_available,
        "embedCliPath": str(BITNET_EMBED_CLI_PATH),
        "embedCliExecutable": embed_cli_exec,
        "embedModelPath": str(BITNET_EMBED_MODEL_PATH),
        "embedModelPresent": embed_model_exists,
        "embedModelSizeMb": embed_model_size_mb,
        "threads": BITNET_THREADS
    }


def sanitize_untrusted_prompt(prompt: str) -> str:
    """
    Neutralize special token sequences (<|...|>) in untrusted user/document text
    to prevent prompt injection and forged assistant turns.
    """
    return re.sub(r"<\|.*?\|>", lambda m: f"[{m.group(0)[2:-2]}]", prompt)


def clean_completion_output(raw_output: str) -> str:
    """
    Clean output from llama-completion (-no-cnv mode).
    Because llama-completion in -no-cnv mode outputs ONLY newly generated tokens
    (without echoing prompt, banner, or interactive prompts), we only need to strip
    special termination markers and trailing whitespace.
    Crucially: does NOT split on 'Assistant:' or '> ', preserving legitimate legal/domain text.
    """
    text = raw_output.strip()
    for end_tok in ("[end of text]", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"):
        if text.endswith(end_tok):
            text = text[:-len(end_tok)].strip()
    return text


def format_bitnet_chat_prompt(prompt: str, apply_template: bool = True) -> str:
    """
    Format prompt for BitNet generation.
    When apply_template is True, neutralizes special tokens in untrusted input and wraps
    in the official SFT chat template (User: ... <|eot_id|>\nAssistant:).
    When False, passes the raw prompt as-is for custom low-level completion.
    """
    if not apply_template:
        return prompt
    sanitized = sanitize_untrusted_prompt(prompt)
    return f"User: {sanitized}<|eot_id|>\nAssistant:"


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
    elif "... (truncated)" in text:
        text = text.split("... (truncated)", 1)[1]
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
    adapter_path: Optional[str] = None,
    apply_chat_template: bool = True
) -> Tuple[str, float]:
    """
    Execute asynchronous native BitNet C++ inference via llama-completion or llama-cli.

    Args:
        prompt: Raw input text from the user or Arena.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        adapter_path: Optional path to a GGUF LoRA adapter file.
        apply_chat_template: Whether to wrap input in the BitNet chat template with sanitized special tokens.

    Returns:
        tuple[str, float]: (generated_text, elapsed_ms)
    """
    if len(prompt) > MAX_GENERATE_PROMPT_CHARS:
        raise ContextOverflowError(
            f"Input prompt ({len(prompt)} characters) exceeds maximum supported generation "
            f"context capacity ({MAX_GENERATE_PROMPT_CHARS} characters / 4,096 tokens)."
        )

    if not is_bitnet_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ native runtime is not available: "
            f"cli_exec={status['cliExecutable']} ({status['cliPath']}), "
            f"model_present={status['modelPresent']} ({status['modelPath']}). "
            f"Set BITNET_CLI_PATH and BITNET_MODEL_PATH environment variables."
        )

    use_completion = is_completion_cli_available()
    formatted_prompt = format_bitnet_chat_prompt(prompt, apply_template=apply_chat_template)

    start_t = time.perf_counter()
    if DEV_STDIN.exists():
        cmd = build_bitnet_generate_cmd(
            input_file="/dev/stdin",
            max_tokens=max_tokens,
            temperature=temperature,
            adapter_path=adapter_path
        )
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=BITNET_TIMEOUT_SECONDS
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=formatted_prompt.encode("utf-8")),
                timeout=BITNET_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(f"BitNet inference timed out after {BITNET_TIMEOUT_SECONDS}s")
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise TimeoutError(f"BitNet inference process timed out after {BITNET_TIMEOUT_SECONDS} seconds")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile("w+", delete=True) as tf:
            tf.write(formatted_prompt)
            tf.flush()
            cmd = build_bitnet_generate_cmd(
                input_file=tf.name,
                max_tokens=max_tokens,
                temperature=temperature,
                adapter_path=adapter_path
            )
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
                    await proc.wait()
                except Exception:
                    pass
                raise TimeoutError(f"BitNet inference process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace").strip()
        logger.error(f"BitNet inference process failed (exit {proc.returncode}): {err_msg}")
        raise RuntimeError(f"BitNet inference exited with code {proc.returncode}: {err_msg}")

    raw_output = stdout.decode("utf-8", errors="replace")
    if use_completion:
        completion_text = clean_completion_output(raw_output)
    else:
        completion_text = parse_llama_cli_output(raw_output, formatted_prompt)

    return completion_text, elapsed_ms


def generate_bitnet_sync(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    adapter_path: Optional[str] = None,
    apply_chat_template: bool = True
) -> Tuple[str, float]:
    """
    Synchronous execution of native BitNet C++ inference via llama-completion or llama-cli.

    Args:
        prompt: Raw input text from the user or Arena.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        adapter_path: Optional path to a GGUF LoRA adapter file.
        apply_chat_template: Whether to wrap input in the BitNet chat template with sanitized special tokens.

    Returns:
        tuple[str, float]: (generated_text, elapsed_ms)
    """
    import subprocess

    if len(prompt) > MAX_GENERATE_PROMPT_CHARS:
        raise ContextOverflowError(
            f"Input prompt ({len(prompt)} characters) exceeds maximum supported generation "
            f"context capacity ({MAX_GENERATE_PROMPT_CHARS} characters / 4,096 tokens)."
        )

    if not is_bitnet_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ native runtime is not available: "
            f"cli_exec={status['cliExecutable']} ({status['cliPath']}), "
            f"model_present={status['modelPresent']} ({status['modelPath']}). "
            f"Set BITNET_CLI_PATH and BITNET_MODEL_PATH environment variables."
        )

    use_completion = is_completion_cli_available()
    formatted_prompt = format_bitnet_chat_prompt(prompt, apply_template=apply_chat_template)

    start_t = time.perf_counter()
    if DEV_STDIN.exists():
        cmd = build_bitnet_generate_cmd(
            input_file="/dev/stdin",
            max_tokens=max_tokens,
            temperature=temperature,
            adapter_path=adapter_path
        )
        try:
            res = subprocess.run(
                cmd,
                input=formatted_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=BITNET_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            logger.error(f"BitNet inference timed out after {BITNET_TIMEOUT_SECONDS}s")
            raise TimeoutError(f"BitNet inference process timed out after {BITNET_TIMEOUT_SECONDS} seconds")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile("w+", delete=True) as tf:
            tf.write(formatted_prompt)
            tf.flush()
            cmd = build_bitnet_generate_cmd(
                input_file=tf.name,
                max_tokens=max_tokens,
                temperature=temperature,
                adapter_path=adapter_path
            )
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=BITNET_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                logger.error(f"BitNet inference timed out after {BITNET_TIMEOUT_SECONDS}s")
                raise TimeoutError(f"BitNet inference process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if res.returncode != 0:
        logger.error(f"BitNet inference process failed (exit {res.returncode}): {res.stderr}")
        raise RuntimeError(f"BitNet inference exited with code {res.returncode}: {res.stderr}")

    if use_completion:
        completion_text = clean_completion_output(res.stdout)
    else:
        completion_text = parse_llama_cli_output(res.stdout, formatted_prompt)
    return completion_text, elapsed_ms



def is_bitnet_embed_available() -> bool:
    """Check whether the native llama-embedding binary, gguf weights, and tokenizer are accessible on the host."""
    cli_ok = BITNET_EMBED_CLI_PATH.is_file() and os.access(str(BITNET_EMBED_CLI_PATH), os.X_OK)
    model_ok = BITNET_EMBED_MODEL_PATH.is_file()
    tok_ok = is_tokenizer_available()
    return cli_ok and model_ok and tok_ok


async def embed_bitnet(prompt: str) -> Tuple[list[float], float]:
    """Asynchronous execution of native BitNet C++ embedding via llama-embedding."""
    if not is_bitnet_embed_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ embedding runtime is not available: "
            f"cli_exec={status['embedCliExecutable']} ({status['embedCliPath']}), "
            f"model_present={status['embedModelPresent']} ({status['embedModelPath']})."
        )

    # Prevent separator collision from splitting text into multiple vectors
    clean_prompt = prompt.replace("<#sep#>", " ")
    clean_prompt_bytes = clean_prompt.encode("utf-8")

    # Tokens cannot exceed UTF-8 bytes (each token is at least 1 byte).
    # Skip tokenization only when byte length <= (MAX_SAFE_EMBED_TOKENS - 2), guaranteeing token count <= 238 <= MAX_SAFE_EMBED_TOKENS.
    if len(clean_prompt_bytes) > (MAX_SAFE_EMBED_TOKENS - 2):
        token_count = await count_embed_tokens_async(clean_prompt)
        if token_count is None:
            raise TokenizerUnavailableError(
                f"Unable to verify token count for prompt exceeding safe byte bound ({len(clean_prompt_bytes)} bytes). "
                f"Request rejected because token verification service is unavailable."
            )
        if token_count > MAX_SAFE_EMBED_TOKENS:
            raise ContextOverflowError(
                f"Input prompt ({token_count} tokens) exceeds the maximum supported context limit "
                f"({MAX_SAFE_EMBED_TOKENS} tokens) for the 1-bit embedding engine."
            )

    start_t = time.perf_counter()
    if DEV_STDIN.exists():
        cmd = build_bitnet_embed_cmd("/dev/stdin")
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=BITNET_TIMEOUT_SECONDS
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=clean_prompt.encode("utf-8")),
                timeout=BITNET_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(f"BitNet embedding timed out after {BITNET_TIMEOUT_SECONDS}s")
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise TimeoutError(f"BitNet embedding process timed out after {BITNET_TIMEOUT_SECONDS} seconds")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile("w+", delete=True, encoding="utf-8") as tf:
            tf.write(clean_prompt)
            tf.flush()
            cmd = build_bitnet_embed_cmd(tf.name)
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    ),
                    timeout=BITNET_TIMEOUT_SECONDS
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=BITNET_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error(f"BitNet embedding timed out after {BITNET_TIMEOUT_SECONDS}s")
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                raise TimeoutError(f"BitNet embedding process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
    stdout_str = stdout_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
        logger.error(f"BitNet embedding process failed (exit {proc.returncode}): {err_msg}")
        if "exceeds batch size" in err_msg or proc.returncode == -11:
            raise ContextOverflowError("Input prompt token length exceeds the maximum context capacity for the 1-bit embedding engine.")
        raise BitNetExecutionError(f"BitNet embedding exited with code {proc.returncode}: {err_msg}")

    return _parse_embed_output(stdout_str, elapsed_ms)


def embed_bitnet_sync(prompt: str) -> Tuple[list[float], float]:
    """Synchronous execution of native BitNet C++ embedding via llama-embedding."""
    import subprocess

    if not is_bitnet_embed_available():
        status = get_bitnet_status()
        raise RuntimeError(
            f"BitNet C++ embedding runtime is not available: "
            f"cli_exec={status['embedCliExecutable']} ({status['embedCliPath']}), "
            f"model_present={status['embedModelPresent']} ({status['embedModelPath']})."
        )

    # Prevent separator collision from splitting text into multiple vectors
    clean_prompt = prompt.replace("<#sep#>", " ")
    clean_prompt_bytes = clean_prompt.encode("utf-8")

    # Tokens cannot exceed UTF-8 bytes (each token is at least 1 byte).
    # Skip tokenization only when byte length <= (MAX_SAFE_EMBED_TOKENS - 2), guaranteeing token count <= 238 <= MAX_SAFE_EMBED_TOKENS.
    if len(clean_prompt_bytes) > (MAX_SAFE_EMBED_TOKENS - 2):
        token_count = count_embed_tokens(clean_prompt)
        if token_count is None:
            raise TokenizerUnavailableError(
                f"Unable to verify token count for prompt exceeding safe byte bound ({len(clean_prompt_bytes)} bytes). "
                f"Request rejected because token verification service is unavailable."
            )
        if token_count > MAX_SAFE_EMBED_TOKENS:
            raise ContextOverflowError(
                f"Input prompt ({token_count} tokens) exceeds the maximum supported context limit "
                f"({MAX_SAFE_EMBED_TOKENS} tokens) for the 1-bit embedding engine."
            )

    start_t = time.perf_counter()
    if DEV_STDIN.exists():
        cmd = build_bitnet_embed_cmd("/dev/stdin")
        try:
            res = subprocess.run(
                cmd,
                input=clean_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=BITNET_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            logger.error(f"BitNet embedding timed out after {BITNET_TIMEOUT_SECONDS}s")
            raise TimeoutError(f"BitNet embedding process timed out after {BITNET_TIMEOUT_SECONDS} seconds")
    else:
        import tempfile
        with tempfile.NamedTemporaryFile("w+", delete=True, encoding="utf-8") as tf:
            tf.write(clean_prompt)
            tf.flush()
            cmd = build_bitnet_embed_cmd(tf.name)
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=BITNET_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                logger.error(f"BitNet embedding timed out after {BITNET_TIMEOUT_SECONDS}s")
                raise TimeoutError(f"BitNet embedding process timed out after {BITNET_TIMEOUT_SECONDS} seconds")

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if res.returncode != 0:
        logger.error(f"BitNet embedding process failed (exit {res.returncode}): {res.stderr}")
        if "exceeds batch size" in res.stderr or res.returncode == -11:
            raise ContextOverflowError("Input prompt token length exceeds the maximum context capacity for the 1-bit embedding engine.")
        raise BitNetExecutionError(f"BitNet embedding exited with code {res.returncode}: {res.stderr}")

    return _parse_embed_output(res.stdout, elapsed_ms)


def _parse_embed_output(stdout_str: str, elapsed_ms: float) -> Tuple[list[float], float]:
    import json
    match = re.search(r'\[\s*\[\s*-?\d+\.?\d*', stdout_str)
    if not match:
        raise BitNetExecutionError("No JSON array found in output.")

    start_idx = match.start()
    end_idx = stdout_str.rfind(']]')
    if end_idx == -1 or end_idx < start_idx:
        raise BitNetExecutionError("Malformed JSON array in output.")

    array_str = stdout_str[start_idx:end_idx+2]

    try:
        embeddings = json.loads(array_str)
    except json.JSONDecodeError as e:
        raise BitNetExecutionError(f"Failed to decode JSON array: {e}")

    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
        if len(embeddings) != 1:
            raise BitNetExecutionError(f"Expected single embedding vector, got {len(embeddings)} vectors (check prompt formatting)")
        embeddings = embeddings[0]

    if len(embeddings) != 640:
        raise BitNetExecutionError(f"Expected embedding dimension 640, got {len(embeddings)}")

    return embeddings, elapsed_ms

