"""Shared utilities for cbplay."""

from datetime import datetime
from pathlib import Path
import os
import re
import subprocess
import shutil

DEBUG = os.getenv('DEBUG') == '1'
DEBUG_FILE = None

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".opus"}

DEFAULT_STREAMING_INSTRUCTIONS = """
Accent/Affect: Warm, refined, and gently instructive, reminiscent of a friendly engaging, fun instructor.
Tone: Calm, encouraging, engaging, fun, and articulate, clearly describing each step with patience.
Pacing: Slow and deliberate, pausing often to allow the listener to follow instructions comfortably. Pause between paragraphs to allow the reader to digest the info.
Emotion: Cheerful, supportive, and pleasantly enthusiastic; convey genuine enjoyment and appreciation of art.
Pronunciation: Clearly articulate artistic terminology (e.g., "brushstrokes," "landscape," "palette") with gentle emphasis.
Personality Affect: Friendly and approachable with a hint of sophistication; speak confidently and reassuringly, guiding users through each painting step patiently and warmly.
Notes: If you see markdown-like formatting, mostly ignore it, (e.g. "#Title", don't say "hash title", say "Title").
"""


def set_debug_file(path: str):
    global DEBUG_FILE
    try:
        DEBUG_FILE = str(Path(path).expanduser().resolve())
        with open(DEBUG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== cbplay session {datetime.now().isoformat()} =====\n")
    except Exception:
        DEBUG_FILE = None


def debug_print(*args, **kwargs):
    if DEBUG:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}]", *args, **kwargs)


def debug_log_file(message: str):
    global DEBUG_FILE
    if not DEBUG_FILE:
        return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(DEBUG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def clean_text_for_display(text: str) -> str:
    text = re.sub(r'[├└┌┐┘┤┬┴┼╭╮╯╰╱╲╳]', '', text)
    text = re.sub(r'\*{10,}', '', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines)


def prepare_text_for_tts(text: str) -> str:
    if text is None:
        return ""
    prepared = clean_text_for_display(str(text)).strip("\r\n")
    return prepared if prepared.strip() else ""


def split_text_intelligently(text: str, max_chars: int = 600) -> list:
    chunks = []
    current_chunk = ""
    
    for line in text.split('\n'):
        line_with_newline = line + '\n'
        
        if len(current_chunk) + len(line_with_newline) <= max_chars:
            current_chunk += line_with_newline
        else:
            if current_chunk:
                chunks.append(current_chunk.rstrip())
            current_chunk = line_with_newline
    
    if current_chunk:
        chunks.append(current_chunk.rstrip())
    
    debug_print(f"Split into {len(chunks)} chunks")
    return chunks


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTS


def resolve_audio_path(audio_file_arg: str, clipboard_content: str):
    if audio_file_arg:
        candidate = audio_file_arg
    else:
        candidate = clipboard_content.strip().splitlines()[0] if clipboard_content.strip() else ""

    if not candidate:
        return None
    candidate = candidate.strip()
    if len(candidate) >= 2 and candidate[0] in ("'", '"') and candidate[-1] == candidate[0]:
        candidate = candidate[1:-1].strip()
    lower = candidate.lower()
    if not any(lower.endswith(ext) for ext in _AUDIO_EXTS):
        return None
    try:
        path = Path(candidate).expanduser()
        return path if path.exists() and path.is_file() else None
    except (OSError, ValueError):
        return None


def get_audio_duration(audio_path) -> float:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    try:
        size = Path(audio_path).stat().st_size
        return size / 16000
    except Exception:
        return 0.0


def ffplay_available() -> bool:
    return shutil.which("ffplay") is not None
