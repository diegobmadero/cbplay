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
    # Keep bold/italic markers - TTS handles them well and keeping them
    # improves alignment between display text and TTS for word highlighting
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines)


# Box drawing and other characters that indicate ASCII art
_BOX_DRAWING_CHARS = set('│─┌┐└┘├┤┬┴┼╭╮╯╰═║╔╗╚╝╠╣╦╩╬')
_ASCII_ART_CHARS = _BOX_DRAWING_CHARS | set('●○◄►▲▼◀▶■□▪▫')


def _is_ascii_art_line(line: str) -> bool:
    """Check if a line is likely ASCII art."""
    if not line.strip():
        return False
    stripped = line.strip()

    # Line starts/ends with box drawing (framed content)
    if stripped[0] in _BOX_DRAWING_CHARS or stripped[-1] in _BOX_DRAWING_CHARS:
        return True

    # Line is mostly box drawing (horizontal rules, etc)
    art_chars = sum(1 for c in stripped if c in _ASCII_ART_CHARS)
    if art_chars / len(stripped) > 0.3:
        return True

    return False


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks (```...```) from text."""
    # Remove fenced code blocks with optional language specifier
    text = re.sub(r'```[a-zA-Z]*\n.*?```', '', text, flags=re.DOTALL)
    return text


def _is_table_line(line: str) -> bool:
    """Check if line is part of a markdown table."""
    stripped = line.strip()
    if not stripped:
        return False
    # Table lines start and end with |
    if stripped.startswith('|') and stripped.endswith('|'):
        return True
    # Separator line like |---|---|
    if re.match(r'^\|[\s\-:]+\|', stripped):
        return True
    return False


def _is_separator_line(line: str) -> bool:
    """Check if line is a table separator (|---|---|)."""
    stripped = line.strip()
    return bool(re.match(r'^\|[\s\-:|]+\|$', stripped) and '-' in stripped)


def _parse_table_row(line: str) -> list:
    """Parse a table row into cells."""
    stripped = line.strip()
    if stripped.startswith('|'):
        stripped = stripped[1:]
    if stripped.endswith('|'):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split('|')]


def _convert_table_to_prose(table_lines: list) -> str:
    """Convert a markdown table to readable prose."""
    if len(table_lines) < 2:
        return ''

    # Find header and data rows
    header_line = None
    data_rows = []

    for i, line in enumerate(table_lines):
        if _is_separator_line(line):
            # Header is the line before separator
            if i > 0:
                header_line = table_lines[i - 1]
            continue
        if header_line is None and not _is_separator_line(line):
            # Before finding separator, this could be header
            continue
        if header_line is not None and not _is_separator_line(line):
            data_rows.append(line)

    # If no separator found, first line is header
    if header_line is None and table_lines:
        header_line = table_lines[0]
        data_rows = table_lines[1:]

    if not header_line:
        return '[Table omitted]'

    headers = _parse_table_row(header_line)
    headers = [h for h in headers if h]  # Remove empty

    if not headers:
        return '[Table omitted]'

    result_parts = []

    for row_line in data_rows:
        if _is_separator_line(row_line):
            continue
        cells = _parse_table_row(row_line)
        row_desc = []
        for i, cell in enumerate(cells):
            if i < len(headers) and cell:
                row_desc.append(f"{headers[i]}: {cell}")
        if row_desc:
            result_parts.append('. '.join(row_desc) + '.')

    if result_parts:
        return '\n'.join(result_parts)
    return '[Table omitted]'


def _convert_tables_to_prose(text: str) -> str:
    """Find and convert markdown tables to prose."""
    lines = text.split('\n')
    result = []
    table_buffer = []
    in_table = False

    for line in lines:
        is_table = _is_table_line(line)

        if is_table:
            in_table = True
            table_buffer.append(line)
        else:
            if in_table and table_buffer:
                # End of table, convert it
                prose = _convert_table_to_prose(table_buffer)
                if prose:
                    result.append(prose)
                table_buffer = []
            in_table = False
            result.append(line)

    # Handle table at end of text
    if table_buffer:
        prose = _convert_table_to_prose(table_buffer)
        if prose:
            result.append(prose)

    return '\n'.join(result)


def _strip_ascii_art(text: str) -> str:
    """Remove ASCII art blocks from text."""
    lines = text.split('\n')
    result = []
    in_art_block = False
    art_block_count = 0

    for line in lines:
        is_art = _is_ascii_art_line(line)

        if is_art:
            art_block_count += 1
            # Start of art block (2+ consecutive art lines)
            if art_block_count >= 2:
                in_art_block = True
        else:
            if in_art_block:
                # Just exited an art block, add a note
                result.append('[Diagram omitted]')
            in_art_block = False
            art_block_count = 0
            result.append(line)

    # Handle trailing art block
    if in_art_block:
        result.append('[Diagram omitted]')

    return '\n'.join(result)


def prepare_text_for_tts(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # Strip code blocks, convert tables, remove ASCII art
    text = _strip_code_blocks(text)
    text = _convert_tables_to_prose(text)
    text = _strip_ascii_art(text)
    # Convert markdown headers to sentences (TTS may skip standalone title lines)
    # e.g., "# Title" -> "Title." - adding period makes TTS read it as content
    def header_to_sentence(match):
        title = match.group(1).strip()
        # Add period if doesn't end with punctuation
        if title and title[-1] not in '.!?:;':
            title += '.'
        return title
    text = re.sub(r'^#{1,6}\s+(.+)$', header_to_sentence, text, flags=re.MULTILINE)
    # Collapse multiple newlines to single space - TTS skips titles followed by blank lines
    text = re.sub(r'\n{2,}', ' ', text)
    prepared = clean_text_for_display(text).strip("\r\n")
    # Collapse multiple "[Diagram omitted]" into one
    prepared = re.sub(r'(\[Diagram omitted\]\s*)+', '[Diagram omitted]\n', prepared)
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
