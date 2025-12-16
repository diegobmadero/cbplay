#!/usr/bin/env python3
# cbplay - Clipboard Play: Text-to-speech for clipboard content

import asyncio
import hashlib
import pyperclip
import openai
try:
    from openai import OpenAI, AsyncOpenAI
    from openai.helpers import LocalAudioPlayer
except Exception:
    from openai import OpenAI
    AsyncOpenAI = None
    LocalAudioPlayer = None
import os
import subprocess
import signal
import shutil
from pathlib import Path
import time
import threading
import queue
import textwrap
import sys
import select
import tty
import termios
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import argparse

DEBUG = os.getenv('DEBUG') == '1'
DEBUG_FILE = None
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".opus"}
DEFAULT_STREAMING_INSTRUCTIONS = """
Accent/Affect: Warm, refined, and gently instructive, reminiscent of a friendly art instructor.
Tone: Calm, encouraging, and articulate, clearly describing each step with patience.
Pacing: Slow and deliberate, pausing often to allow the listener to follow instructions comfortably. Pause between paragraphs to allow the reader to digest the info.
Emotion: Cheerful, supportive, and pleasantly enthusiastic; convey genuine enjoyment and appreciation of art.
Pronunciation: Clearly articulate artistic terminology (e.g., "brushstrokes," "landscape," "palette") with gentle emphasis.
Personality Affect: Friendly and approachable with a hint of sophistication; speak confidently and reassuringly, guiding users through each painting step patiently and warmly.
Notes: If you see markdown-like formatting, mostly ignore it, (e.g. "#Title", don't say "hash title", say "Title").
"""

def debug_print(*args, **kwargs):
    if DEBUG:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}]", *args, **kwargs)

def debug_log_file(message):
    """Append a debug line to a log file when configured."""
    global DEBUG_FILE
    if not DEBUG_FILE:
        return
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(DEBUG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass

class RateLimiter:
    """Rate limiter to respect OpenAI TTS API limits"""
    def __init__(self, requests_per_minute=50):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
            
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate how long to wait
                oldest_request = self.request_times[0]
                wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    debug_print(f"Rate limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    return self.wait_if_needed()
            
            self.request_times.append(now)

class TTSFile:
    def __init__(self, voice="nova", response_format="aac", file_prefix="tts_clipboard", model="tts-1-hd", instructions=None):
        self.timeout = 240
        self.voice = voice
        self.response_format = response_format
        self.file_prefix = file_prefix
        self.model = model
        self.instructions = instructions
        self.client = OpenAI(timeout=self.timeout)
        self.rate_limiter = RateLimiter(requests_per_minute=50)  # OpenAI's default TTS limit
        self.generated_files = []
        self.cache_dir = Path.home() / ".whisper" / "audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache index
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Clean old cache entries
        self._clean_cache()

    def _load_cache_index(self):
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _clean_cache(self, max_age_days=7):
        """Remove cache files older than max_age_days"""
        now = time.time()
        to_remove = []
        
        for hash_key, data in self.cache_index.items():
            if now - data['timestamp'] > max_age_days * 86400:
                cached_file = self.cache_dir / f"{hash_key}.{self.response_format}"
                if cached_file.exists():
                    cached_file.unlink()
                to_remove.append(hash_key)
        
        for key in to_remove:
            del self.cache_index[key]
        
        if to_remove:
            self._save_cache_index()
            debug_print(f"Cleaned {len(to_remove)} old cache entries")

    def _effective_instructions(self):
        instructions = (self.instructions or "").strip()
        if not instructions:
            return None
        if self.model.startswith("gpt-4o"):
            return instructions
        return None

    def _set_params(self, text):
        raw_text = "" if text is None else str(text)
        if not raw_text.strip():
            debug_print("Text input is empty after sanitization. Skipping.")
            return None
        params = {
            "voice": self.voice,
            "model": self.model,
            "response_format": self.response_format,
            "input": raw_text,
        }
        instructions = self._effective_instructions()
        if instructions:
            params["instructions"] = instructions
        return params

    def _hash_text(self, text):
        # Backwards-compatible cache key for the legacy default (voice + text only).
        instructions = self._effective_instructions()
        if self.model == "tts-1-hd" and not instructions:
            content = f"{self.voice}:{text}"
        else:
            content = "\n".join([
                f"model:{self.model}",
                f"voice:{self.voice}",
                f"format:{self.response_format}",
                f"instructions:{instructions or ''}",
                f"text:{text}",
            ])
        return hashlib.sha256(content.encode()).hexdigest()

    def to_file(self, text, out_file):
        text_hash = self._hash_text(text)
        cached_file = self.cache_dir / f"{text_hash}.{self.response_format}"
        
        # Remove existing output file if it exists
        if out_file.exists():
            out_file.unlink()
        
        # Check cache
        if text_hash in self.cache_index and cached_file.exists():
            debug_print(f"Using cached audio file: {cached_file}")
            # Use copy instead of hard link to avoid issues
            import shutil
            shutil.copy2(cached_file, out_file)
            return out_file

        params = self._set_params(text)
        if params is None:
            return None
        
        # Wait for rate limit
        self.rate_limiter.wait_if_needed()
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = self.client.audio.speech.create(**params)
                break
            except openai.RateLimitError as e:
                retry_count += 1
                wait_time = min(2 ** retry_count, 30)  # Exponential backoff
                print(f"Rate limit error, waiting {wait_time}s (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
            except openai.BadRequestError as e:
                print(f"Failed to generate audio due to bad request: {e}")
                return None
            except Exception as e:
                print(f"Failed to generate audio due to error: {e}")
                return None
        else:
            print(f"Failed to generate audio after {max_retries} attempts")
            return None
        
        with open(out_file, "wb") as file:
            file.write(response.content)
        
        # Cache the file
        import shutil
        shutil.copy2(out_file, cached_file)
        self.cache_index[text_hash] = {
            'timestamp': time.time(),
            'text_preview': text[:50],
            'voice': self.voice
        }
        self._save_cache_index()
        
        debug_print(f"Generated audio file at: {out_file}")
        return out_file

def clean_text_for_display(text):
    """Minimal text cleaning - just remove the worst formatting artifacts"""
    import re
    
    # Remove box drawing characters
    text = re.sub(r'[│├└─┌┐┘┤┬┴┼╭╮╯╰╱╲╳]', '', text)
    
    # Remove excessive asterisks (more than 10 in a row)
    text = re.sub(r'\*{10,}', '', text)
    
    # Clean up excessive newlines (more than 3)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    
    return '\n'.join(lines)

def prepare_text_for_tts(text: str) -> str:
    """Prepare text once so TTS input, display, and highlighting stay in sync."""
    if text is None:
        return ""
    prepared = clean_text_for_display(str(text)).strip("\r\n")
    return prepared if prepared.strip() else ""

def get_clipboard_content():
    try:
        content = pyperclip.paste()
    except Exception as e:
        debug_print(f"Clipboard read failed: {e}")
        return ""
    if content is None:
        return ""
    content = str(content)
    debug_print(f"Clipboard content: {content[:100]}...")
    return content

def split_text_intelligently(text, max_chars=600):
    """Split text into chunks without breaking the structure"""
    chunks = []
    current_chunk = ""
    
    # Process the text line by line to preserve structure
    lines = text.split('\n')
    
    for line in lines:
        # Check if adding this line would exceed the limit
        line_with_newline = line + '\n'
        
        if len(current_chunk) + len(line_with_newline) <= max_chars:
            # Add the line to current chunk
            current_chunk += line_with_newline
        else:
            # Start a new chunk
            if current_chunk:
                chunks.append(current_chunk.rstrip())
            current_chunk = line_with_newline
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.rstrip())
    
    debug_print(f"Split into {len(chunks)} chunks")
    return chunks

async def stream_tts_chunks(combined_texts, voice, model, instructions, response_format="pcm", timeout=240):
    """Stream TTS audio using the async OpenAI client and play immediately."""
    if not AsyncOpenAI or not LocalAudioPlayer:
        raise ImportError("AsyncOpenAI/LocalAudioPlayer not available in installed openai package.")
    
    # LocalAudioPlayer expects PCM from streaming responses; enforce it
    if response_format.lower() != "pcm":
        debug_print("Streaming overrides response_format to 'pcm' for LocalAudioPlayer playback.")
        response_format = "pcm"
    
    client = AsyncOpenAI(timeout=timeout)
    player = LocalAudioPlayer()
    total = len(combined_texts)
    for index, text in enumerate(combined_texts, 1):
        prepared = prepare_text_for_tts(text)
        if not prepared:
            continue
        
        params = {
            "model": model,
            "voice": voice,
            "input": prepared,
            "response_format": response_format,
        }
        if instructions:
            params["instructions"] = instructions
        
        header = f"Chunk {index}/{total} ({len(prepared)} chars)"
        print(f"\n{header}")
        print("-" * len(header))
        print(prepared)
        print()
        print("Streaming and playing...")
        async with client.audio.speech.with_streaming_response.create(**params) as response:
            await player.play(response)

def generate_audio_files_streaming(combined_texts, tts, audio_queue, status_queue):
    """Generate audio files concurrently and stream them to the queue as they're ready"""
    debug_print(f"Starting generation for {len(combined_texts)} chunks")
    if not combined_texts:
        debug_print("No valid text found to generate audio.")
        status_queue.put(("done", 0, 0))
        return
    
    total = len(combined_texts)
    completed = 0
    
    # First, check what's already cached
    cached_count = 0
    for i, text in enumerate(combined_texts):
        text_hash = tts._hash_text(text)
        cached_file = tts.cache_dir / f"{text_hash}.{tts.response_format}"
        if text_hash in tts.cache_index and cached_file.exists():
            cached_count += 1
            debug_print(f"Chunk {i+1} is cached: {text[:50]}...")
    
    debug_print(f"Found {cached_count} cached chunks out of {total}")
    status_queue.put(("cached", cached_count, total))
    
    # Use ThreadPoolExecutor for concurrent generation
    max_workers = min(3, total)  # Limit concurrent requests to respect rate limits
    
    def generate_single(index_text):
        index, text = index_text
        debug_print(f"Generating chunk {index + 1}/{total}")
        text_hash = tts._hash_text(text)
        unique_filename = f"{tts.file_prefix}_{text_hash}.{tts.response_format}"
        # Store temp files in ~/.whisper/temp
        temp_dir = Path.home() / ".whisper" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_file = temp_dir / unique_filename
        
        generated_file = tts.to_file(text, out_file)
        if generated_file is not None:
            debug_print(f"Successfully generated chunk {index + 1}")
            return (index, generated_file, text)
        else:
            debug_print(f"Failed to generate chunk {index + 1}: {text[:100]}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for i, text in enumerate(combined_texts):
            future = executor.submit(generate_single, (i, text))
            futures.append((i, future))
        
        # Stream results as they complete
        pending_results = {}
        next_index = 0
        
        for i, future in futures:
            result = future.result()
            if result:
                index, audio_file, text = result
                pending_results[index] = (audio_file, text)
                completed += 1
                status_queue.put(("progress", completed, total))
                
                # Check if we can stream any sequential results
                while next_index in pending_results:
                    audio_queue.put(pending_results[next_index])
                    debug_print(f"Added chunk {next_index + 1} to queue")
                    del pending_results[next_index]
                    next_index += 1
        
        # Add any remaining results in order
        for i in sorted(pending_results.keys()):
            audio_queue.put(pending_results[i])
            debug_print(f"Added remaining chunk {i + 1} to queue")
    
    debug_print(f"Generation complete. Added {completed} chunks to queue")
    status_queue.put(("done", completed, total))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def is_data():
    try:
        return bool(select.select([sys.stdin.fileno()], [], [], 0)[0])
    except Exception:
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def play_audio_files_with_status(
    audio_queue,
    status_queue,
    tts,
    highlight=False,
    highlight_model="gpt-4o-transcribe",
    highlight_window=8,
    resume_rewind=2.0,
    playhead_lag=None,
    esc_timeout=None,
    ui_debug=False,
):
    # Check if we're in a TTY or if NOTTY is set
    if not sys.stdin.isatty() or os.getenv('NOTTY') == '1':
        print("Not running in a terminal, audio generation only mode")
        # Just wait for generation to complete and show debug info
        total_received = 0
        while True:
            # Check queue
            while not audio_queue.empty():
                try:
                    chunk = audio_queue.get_nowait()
                    total_received += 1
                    debug_print(f"Received chunk {total_received} from queue")
                except queue.Empty:
                    break
            
            if not status_queue.empty():
                status = status_queue.get()
                if status[0] == "cached":
                    print(f"Found {status[1]} cached chunks out of {status[2]} total")
                elif status[0] == "done":
                    print(f"Generated {status[1]} audio files")
                    # Final queue check
                    while not audio_queue.empty():
                        try:
                            chunk = audio_queue.get_nowait()
                            total_received += 1
                            debug_print(f"Received chunk {total_received} from queue (final)")
                        except queue.Empty:
                            break
                    print(f"Total chunks received: {total_received}")
                    break
            time.sleep(0.1)
        return
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        stdin_fd = sys.stdin.fileno()
        active_color = '\033[93m'  # Yellow
        prev_color = '\033[90m'    # Grey
        reset_color = '\033[0m'
        info_color = '\033[96m'    # Cyan
        success_color = '\033[92m'  # Green
        term_width = shutil.get_terminal_size((100, 20)).columns
        display_width = max(20, min(100, term_width))
        ffplay_available = shutil.which("ffplay") is not None
        rewind_padding = max(0.0, float(resume_rewind or 0.0))
        highlight_enabled = bool(highlight)
        ui_debug = bool(ui_debug)
        try:
            highlight_context_words = max(0, int(highlight_window or 0))
        except Exception:
            highlight_context_words = 1

        if playhead_lag is None:
            # Start with a small/zero correction; users can tune via --playhead-lag if needed.
            playhead_lag_seconds = 0.0 if ffplay_available else 0.05
        else:
            try:
                playhead_lag_seconds = max(0.0, float(playhead_lag))
            except Exception:
                playhead_lag_seconds = 0.0 if ffplay_available else 0.05

        esc_timeout_first = 0.60 if os.getenv("TMUX") else 0.25
        if esc_timeout is not None:
            try:
                esc_timeout_first = max(0.01, float(esc_timeout))
            except Exception:
                pass
        else:
            esc_timeout_env = os.getenv("CBPLAY_ESC_TIMEOUT")
            if esc_timeout_env:
                try:
                    esc_timeout_first = max(0.01, float(esc_timeout_env))
                except Exception:
                    pass
        esc_timeout_rest = 0.01
        if ui_debug:
            debug_log_file(
                f"ui config ffplay={ffplay_available} playhead_lag={playhead_lag_seconds} "
                f"esc_timeout_first={esc_timeout_first} tmux={'y' if os.getenv('TMUX') else 'n'}"
            )
        highlight_inflight = set()
        highlight_lock = threading.Lock()
        highlight_task_queue = queue.Queue() if highlight_enabled else None

        def clear_pause_message():
            sys.stdout.write("\033[F\033[K")
            sys.stdout.flush()

        def read_char():
            try:
                data = os.read(stdin_fd, 1)
            except Exception:
                return ""
            if not data:
                return ""
            return chr(data[0])

        def read_escape_sequence(max_chars=32, timeout_first=0.05, timeout_rest=0.005):
            """Read remaining chars after an ESC keypress (e.g. arrow-key sequence)."""
            seq_chars = []
            for i in range(int(max_chars or 0)):
                timeout = float(timeout_first if i == 0 else timeout_rest)
                try:
                    ready, _, _ = select.select([stdin_fd], [], [], timeout)
                except Exception:
                    break
                if not ready:
                    break
                try:
                    data = os.read(stdin_fd, 1)
                except Exception:
                    break
                if not data:
                    break
                seq_chars.append(chr(data[0]))
                if seq_chars[0] == "O" and len(seq_chars) >= 2:
                    break
                if seq_chars[0] == "[" and len(seq_chars) >= 2:
                    last = seq_chars[-1]
                    if last.isalpha() or last == "~":
                        break
            return "".join(seq_chars)

        def word_cache_path_for_text(text):
            return tts.cache_dir / f"{tts._hash_text(text)}.words.json"

        def highlight_worker():
            while True:
                audio_path, cache_path, key = highlight_task_queue.get()
                try:
                    existing = load_word_timestamps(cache_path, expected_model=highlight_model)
                    if existing and existing.get("status") == "ok":
                        continue
                    words = transcribe_audio_words(Path(audio_path), model=highlight_model)
                    if words:
                        save_word_timestamps(cache_path, highlight_model, words, status="ok", error=None)
                    else:
                        save_word_timestamps(
                            cache_path,
                            highlight_model,
                            [],
                            status="error",
                            error="No word timestamps returned",
                        )
                except Exception as e:
                    debug_print(f"Highlight transcription worker failed: {e}")
                    try:
                        save_word_timestamps(cache_path, highlight_model, [], status="error", error=str(e))
                    except Exception:
                        pass
                finally:
                    with highlight_lock:
                        highlight_inflight.discard(key)
                    highlight_task_queue.task_done()

        if highlight_enabled:
            for _ in range(2):
                threading.Thread(target=highlight_worker, daemon=True).start()

        def schedule_word_timestamps(audio_path, text):
            cache_path = word_cache_path_for_text(text)
            if cache_path.exists():
                existing = load_word_timestamps(cache_path, expected_model=highlight_model)
                if existing and existing.get("status") == "ok":
                    return cache_path
                if existing and existing.get("status") == "error":
                    try:
                        created_at = float(existing.get("created_at") or 0.0)
                    except Exception:
                        created_at = 0.0
                    if created_at > 0 and (time.time() - created_at) < 60:
                        return cache_path
            key = str(cache_path)
            with highlight_lock:
                if key in highlight_inflight:
                    return cache_path
                highlight_inflight.add(key)
            highlight_task_queue.put((audio_path, cache_path, key))
            return cache_path

        import re
        ansi_escape_re = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
        token_re = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")

        def hard_wrap_ansi(text: str, width: int):
            if width <= 0:
                return [text]
            lines = []
            buf = []
            visible = 0
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == "\n":
                    lines.append("".join(buf))
                    buf = []
                    visible = 0
                    i += 1
                    continue
                if ch == "\x1b":
                    m = ansi_escape_re.match(text, i)
                    if m:
                        buf.append(m.group(0))
                        i = m.end()
                        continue
                buf.append(ch)
                visible += 1
                i += 1
                if visible >= width:
                    lines.append("".join(buf))
                    buf = []
                    visible = 0
            lines.append("".join(buf))
            return lines

        _number_norm_map = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "sixty": "60",
            "seventy": "70",
            "eighty": "80",
            "ninety": "90",
        }

        def normalize_for_match(word: str):
            cleaned = re.sub(r"[^a-z0-9]+", "", (word or "").lower())
            return _number_norm_map.get(cleaned, cleaned)

        def extract_text_tokens(text: str):
            tokens = []
            for match in token_re.finditer(text or ""):
                raw = match.group(0)
                norm = normalize_for_match(raw)
                if not norm:
                    continue
                tokens.append({"norm": norm, "start": match.start(), "end": match.end()})
            return tokens

        def align_words_to_text(words, tokens, lookahead=12):
            """Align transcription words to text tokens while staying monotonic.

            Uses a lightweight global alignment so repeats and small mismatches don't
            cause greedy drift.
            """
            mapping = [None] * (len(words) if isinstance(words, list) else 0)
            if not words or not tokens:
                return mapping

            word_norms = []
            for item in words:
                if isinstance(item, dict):
                    w = str(item.get("word", "")).strip()
                else:
                    w = str(item).strip()
                word_norms.append(normalize_for_match(w))
            token_norms = [str(t.get("norm") or "") for t in tokens]

            def is_match(a: str, b: str) -> bool:
                if not a or not b:
                    return False
                if a == b:
                    return True
                if len(a) >= 4 and len(b) >= 4:
                    return a.startswith(b) or b.startswith(a)
                return False

            n = len(word_norms)
            m = len(token_norms)
            if n == 0 or m == 0:
                return mapping

            # If inputs are very large, fall back to a bounded greedy scan.
            if n * m > 600_000:
                token_index = 0
                for i, w_norm in enumerate(word_norms):
                    if not w_norm:
                        continue
                    end = min(token_index + int(lookahead or 0), m)
                    found = None
                    for j in range(token_index, end):
                        if is_match(w_norm, token_norms[j]):
                            found = j
                            break
                    if found is not None:
                        mapping[i] = found
                        token_index = found + 1
                return mapping

            gap_penalty = 1
            match_score = 2
            neg_inf = -10**9

            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                dp[i][0] = dp[i - 1][0] - gap_penalty
            for j in range(1, m + 1):
                dp[0][j] = dp[0][j - 1] - gap_penalty

            for i in range(1, n + 1):
                a = word_norms[i - 1]
                row = dp[i]
                prev_row = dp[i - 1]
                for j in range(1, m + 1):
                    best = max(prev_row[j] - gap_penalty, row[j - 1] - gap_penalty)
                    b = token_norms[j - 1]
                    if is_match(a, b):
                        best = max(best, prev_row[j - 1] + match_score)
                    else:
                        best = max(best, prev_row[j - 1] + neg_inf)
                    row[j] = best

            i = n
            j = m
            while i > 0 and j > 0:
                a = word_norms[i - 1]
                b = token_norms[j - 1]
                if is_match(a, b) and dp[i][j] == dp[i - 1][j - 1] + match_score:
                    mapping[i - 1] = j - 1
                    i -= 1
                    j -= 1
                    continue
                if dp[i][j] == dp[i - 1][j] - gap_penalty:
                    i -= 1
                    continue
                j -= 1

            return mapping

        def render_current_region(text: str, highlight_spans=None, paused=False, debug_line=None):
            label = "Current"
            if paused:
                label += " [PAUSED]"
            label_line = f"{active_color}{label}:{reset_color}"

            rendered_text = text or ""
            if highlight_spans:
                events = {}
                for span in highlight_spans:
                    if not span or len(span) != 4:
                        continue
                    try:
                        start, end, start_code, end_code = span
                        start = int(start)
                        end = int(end)
                        start_code = str(start_code)
                        end_code = str(end_code)
                    except Exception:
                        continue
                    if start_code:
                        events.setdefault(start, []).append(start_code)
                    if end_code:
                        events.setdefault(end, []).append(end_code)
                for idx in sorted(events.keys(), reverse=True):
                    if 0 <= idx <= len(rendered_text):
                        rendered_text = rendered_text[:idx] + "".join(events[idx]) + rendered_text[idx:]

            wrapped = hard_wrap_ansi(rendered_text, display_width)
            lines = [label_line]
            # Keep body text in default terminal color; only highlight spans add styling.
            lines.extend(wrapped)
            lines.append(reset_color)
            if debug_line:
                lines.append(f"{info_color}{debug_line}{reset_color}")
            return lines

        def build_highlight_spans(context_span, current_word_span):
            spans = []
            if context_span and isinstance(context_span, (tuple, list)) and len(context_span) == 2:
                try:
                    start, end = int(context_span[0]), int(context_span[1])
                except Exception:
                    start, end = None, None
                if start is not None and end is not None and 0 <= start < end:
                    spans.append((start, end, "\033[4m", "\033[24m"))  # underline
            if current_word_span and isinstance(current_word_span, (tuple, list)) and len(current_word_span) == 2:
                try:
                    start, end = int(current_word_span[0]), int(current_word_span[1])
                except Exception:
                    start, end = None, None
                if start is not None and end is not None and 0 <= start < end:
                    spans.append((start, end, "\033[7m", "\033[27m"))  # reverse video
            return spans if spans else None

        def redraw_region(lines, line_count):
            if not lines:
                return
            if line_count > 1:
                sys.stdout.write(f"\033[{line_count - 1}F")
            else:
                sys.stdout.write("\r")
            for i, ln in enumerate(lines):
                sys.stdout.write("\r\033[K" + (ln or ""))
                if i < len(lines) - 1:
                    sys.stdout.write("\n")
            sys.stdout.flush()

        def highlight_debug_info(audio_path, cache_path, status, error):
            if not ui_debug:
                return None
            try:
                audio_ext = Path(audio_path).suffix.lower() if audio_path else ""
            except Exception:
                audio_ext = ""
            cache_exists = bool(cache_path and Path(cache_path).exists())
            cache_name = ""
            try:
                cache_name = Path(cache_path).name if cache_path else ""
            except Exception:
                cache_name = ""
            qsize = 0
            if highlight_task_queue is not None:
                try:
                    qsize = int(highlight_task_queue.qsize())
                except Exception:
                    qsize = 0
            with highlight_lock:
                inflight = len(highlight_inflight)
            err = (error or "").replace("\n", " ").strip()
            if len(err) > 160:
                err = err[:157] + "..."
            parts = [
                f"dbg st={status}",
                f"m={highlight_model}",
                f"audio={audio_ext or '?'}",
                f"cache={'y' if cache_exists else 'n'}",
                f"q={qsize}",
                f"in={inflight}",
            ]
            if cache_name:
                parts.append(f"file={cache_name}")
            if err:
                parts.append(f"err={err}")
            return " ".join(parts)

        def start_player(audio_path, start_at=0.0):
            if ffplay_available:
                cmd = ['ffplay', '-nodisp', '-autoexit']
                if start_at > 0:
                    cmd.extend(['-ss', f'{start_at}'])
                cmd.append(str(audio_path))
                return subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            return subprocess.Popen(
                ['afplay', str(audio_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        history = []
        current_index = -1
        generation_done = False
        total_chunks = 0
        generated_chunks = 0
        expected_total = 0  # Track the expected total from status updates
        
        # Start time for showing generation speed
        start_time = time.time()

        while True:
            clear_screen()
            debug_log_file(f"ui_loop start idx={current_index} hist={len(history)} gen_done={generation_done} q_empty={audio_queue.empty()} status_q_empty={status_queue.empty()}")
            
            # Check for status updates
            while not status_queue.empty():
                try:
                    status = status_queue.get_nowait()
                    if status[0] == "cached":
                        print(f"{success_color}Found {status[1]} cached chunks out of {status[2]} total{reset_color}")
                        debug_log_file(f"status cached={status[1]} total={status[2]}")
                    elif status[0] == "progress":
                        generated_chunks = status[1]
                        total_chunks = status[2]
                        debug_log_file(f"status progress {generated_chunks}/{total_chunks}")
                    elif status[0] == "done":
                        generation_done = True
                        total_chunks = status[2]  # Make sure we have the total
                        elapsed = time.time() - start_time
                        print(f"{success_color}Generation complete! Generated {status[1]} chunks in {elapsed:.1f}s{reset_color}")
                        debug_print(f"Generation done. Total chunks: {total_chunks}, History size: {len(history)}, Queue empty: {audio_queue.empty()}")
                        debug_log_file(f"status done chunks={status[1]} total={total_chunks} history={len(history)} queue_empty={audio_queue.empty()}")
                except queue.Empty:
                    break
            
            # Show generation status
            if not generation_done and total_chunks > 0:
                percent = (generated_chunks / total_chunks) * 100
                bar_length = 40
                filled = int(bar_length * generated_chunks / total_chunks)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"Generating: [{bar}] {percent:.0f}% ({generated_chunks}/{total_chunks})\n")
            
            # Show controls and status
            if 0 <= current_index < len(history):
                queue_status = f"Playing {current_index + 1}"
                if generation_done:
                    queue_status += f"/{total_chunks}" if total_chunks > 0 else f"/{len(history)}"
                else:
                    queue_status += f"/{len(history)}+ (generating...)"
                print(f"{info_color}{queue_status} | Controls: ↑/↓ - Navigate | Space - Pause/Resume | Q or ESC - Exit | Auto-advances when ready{reset_color}\n")
            else:
                print(f"{info_color}Controls: ↑/↓ - Navigate | Space - Pause/Resume | Q or ESC - Exit | Auto-advances when ready{reset_color}\n")
            
            # Get next audio if available and we're at the end
            debug_print(f"Loop iteration: current_index={current_index}, history_len={len(history)}, queue_empty={audio_queue.empty()}")
            debug_log_file(f"loop iteration idx={current_index} hist={len(history)} queue_empty={audio_queue.empty()}")
            
            # First check if we should use existing history
            if 0 <= current_index < len(history):
                current_text_chunk = history[current_index]
                debug_print(f"Using existing chunk at index {current_index}")
                debug_log_file(f"using existing chunk idx={current_index}")
            # Only get from queue if we need a new chunk
            elif not audio_queue.empty():
                # Get the first available chunk from queue
                try:
                    current_text_chunk = audio_queue.get_nowait()
                    history.append(current_text_chunk)
                    debug_log_file(f"dequeued chunk idx={len(history)-1} queue_size={audio_queue.qsize()}")
                    if highlight_enabled:
                        try:
                            schedule_word_timestamps(current_text_chunk[0], current_text_chunk[1])
                            debug_log_file("scheduled word timestamps for dequeued chunk")
                        except Exception as e:
                            debug_log_file(f"failed schedule word timestamps: {e}")
                            pass
                    if current_index == -1:
                        current_index = 0
                    else:
                        current_index = len(history) - 1
                    debug_print(f"Retrieved chunk from queue, now at index {current_index}, history size: {len(history)}")
                except queue.Empty:
                    debug_print("Queue was empty when trying to get chunk")
                    debug_log_file("queue empty when trying to dequeue")
                    pass
            else:
                if generation_done and len(history) == 0:
                    print("No audio files were generated.")
                    debug_log_file("done generating but history empty, exiting")
                    break
                elif current_index >= len(history) and generation_done:
                    print(f"{success_color}Playback complete!{reset_color}")
                    debug_log_file("playback complete, exiting main loop")
                    break
                else:
                    # Waiting for first chunk or next chunk
                    if current_index == -1:
                        debug_print("Waiting for first audio chunk to be generated...")
                        debug_log_file("waiting for first chunk")
                    else:
                        debug_print(f"Waiting for next audio chunk...")
                        debug_log_file("waiting for next chunk")
                    time.sleep(0.2)
                    continue

            # Show history context
            if current_index > 0:
                prev_file, prev_text = history[current_index - 1]
                cleaned_prev = prev_text
                print(f"{prev_color}Previous:\n{cleaned_prev}\n{reset_color}")
    
            if 0 <= current_index < len(history):
                audio_file, original_text_chunk = history[current_index]
                cleaned_text = original_text_chunk
                if not highlight_enabled:
                    print(f"{active_color}Current:{reset_color}\n{cleaned_text}\n")
    
                words_cache_path = None
                words = None
                words_error = None
                words_status = "preparing"
                current_word_index = -1
                last_words_status = words_status
                tokens = None
                word_to_token = None
                mapping_total = 0
                mapping_mapped = 0
                text_debug_hash = None
                if ui_debug:
                    try:
                        text_debug_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()[:10]
                    except Exception:
                        text_debug_hash = None
                current_highlight_span = None
                current_highlight_word_span = None
                region_lines = None
                region_line_count = 0
                last_debug_line = None
                last_paused_state = False
                if highlight_enabled:
                    try:
                        words_cache_path = schedule_word_timestamps(audio_file, original_text_chunk)
                        payload = load_word_timestamps(words_cache_path, expected_model=highlight_model)
                        if payload is None:
                            words_status = "preparing"
                        elif payload.get("status") == "ok":
                            words_status = "ok"
                            words = payload.get("words") or None
                        else:
                            words_status = "error"
                            words_error = payload.get("error")
                    except Exception:
                        words_cache_path = None
                        words = None
                        words_error = None
                        words_status = "preparing"
                if words and words_status == "ok":
                    try:
                        if float(words[0].get("start", 0.0)) <= 0.05:
                            current_word_index = 0
                    except Exception:
                        pass
                    tokens = extract_text_tokens(cleaned_text)
                    word_to_token = align_words_to_text(words, tokens)
                    if word_to_token:
                        mapping_total = len(word_to_token)
                        mapping_mapped = sum(1 for idx in word_to_token if idx is not None)
                    if current_word_index >= 0 and word_to_token and tokens:
                        spans = []
                        for wi in range(
                            current_word_index - highlight_context_words,
                            current_word_index + highlight_context_words + 1,
                        ):
                            if 0 <= wi < len(word_to_token):
                                token_idx = word_to_token[wi]
                                if token_idx is not None and 0 <= token_idx < len(tokens):
                                    t = tokens[token_idx]
                                    spans.append((t["start"], t["end"]))
                        if spans:
                            current_highlight_span = (min(s for s, _ in spans), max(e for _, e in spans))
                        token_idx = word_to_token[current_word_index]
                        if token_idx is not None and 0 <= token_idx < len(tokens):
                            t = tokens[token_idx]
                            current_highlight_word_span = (t["start"], t["end"])
                if words_status == "ok":
                    try:
                        first_word = words[0] if words else None
                        last_word = words[-1] if words else None
                        debug_log_file(
                            f"words ready count={len(words or [])} "
                            f"first={first_word} last={last_word}"
                        )
                    except Exception:
                        debug_log_file(f"words ready count={len(words or [])}")
                last_words_status = words_status
                last_debug_line = highlight_debug_info(audio_file, words_cache_path, words_status, words_error)
                if ui_debug:
                    try:
                        if text_debug_hash:
                            last_debug_line = (last_debug_line or "") + f" txt={text_debug_hash}"
                        if mapping_total:
                            cov = mapping_mapped / mapping_total if mapping_total else 0.0
                            last_debug_line = (last_debug_line or "") + f" map={mapping_mapped}/{mapping_total}({cov:.0%})"
                    except Exception:
                        pass
                region_lines = render_current_region(
                    cleaned_text,
                    highlight_spans=build_highlight_spans(current_highlight_span, current_highlight_word_span),
                    paused=False,
                    debug_line=last_debug_line,
                )
                region_line_count = len(region_lines)
                for i, ln in enumerate(region_lines):
                    sys.stdout.write((ln or ""))
                    if i < len(region_lines) - 1:
                        sys.stdout.write("\n")
                sys.stdout.flush()
                debug_log_file(f"rendered initial region lines={region_line_count} words_status={words_status} cache={words_cache_path}")

            # Play audio
            playback_offset = 0.0  # Where playback started for this process
            process = start_player(audio_file, start_at=playback_offset)
            playback_started_at = time.monotonic()
            playback_lag = playhead_lag_seconds
            paused_at = None  # Playhead position when paused
            user_interrupted = False
            paused = False
            pause_message_shown = False
            while process.poll() is None:
                if is_data():
                    c = read_char()
                    if c == ' ':
                        now = time.monotonic()
                        current_playhead = max(0.0, playback_offset + (now - playback_started_at) - playback_lag)
                        if paused:
                            if (not highlight_enabled) and pause_message_shown:
                                clear_pause_message()
                            resume_from = paused_at if paused_at is not None else current_playhead
                            target_offset = max(resume_from - rewind_padding, 0.0)
                            if ffplay_available:
                                if process.poll() is None:
                                    try:
                                        process.send_signal(signal.SIGCONT)
                                        process.terminate()
                                        process.wait(timeout=0.3)
                                    except Exception:
                                        try:
                                            process.kill()
                                        except Exception:
                                            pass
                                process = start_player(audio_file, start_at=target_offset)
                                playback_offset = target_offset
                                playback_lag = playhead_lag_seconds
                            else:
                                process.send_signal(signal.SIGCONT)
                                playback_offset = resume_from
                                playback_lag = 0.0
                            playback_started_at = time.monotonic()
                            paused = False
                            paused_at = None
                            pause_message_shown = False
                            debug_log_file(f"resumed playback_offset={playback_offset}")
                            if highlight_enabled and region_lines is not None:
                                debug_line = highlight_debug_info(audio_file, words_cache_path, words_status, words_error)
                                region_lines = render_current_region(
                                    cleaned_text,
                                    highlight_spans=build_highlight_spans(current_highlight_span, current_highlight_word_span),
                                    paused=False,
                                    debug_line=debug_line,
                                )
                                redraw_region(region_lines, region_line_count)
                                last_paused_state = False
                                last_debug_line = debug_line
                        else:
                            process.send_signal(signal.SIGSTOP)
                            paused_at = current_playhead
                            paused = True
                            pause_message_shown = False
                            debug_log_file(f"paused at playhead={paused_at}")
                            if highlight_enabled and region_lines is not None:
                                debug_line = highlight_debug_info(audio_file, words_cache_path, words_status, words_error)
                                region_lines = render_current_region(
                                    cleaned_text,
                                    highlight_spans=build_highlight_spans(current_highlight_span, current_highlight_word_span),
                                    paused=True,
                                    debug_line=debug_line,
                                )
                                redraw_region(region_lines, region_line_count)
                                last_paused_state = True
                                last_debug_line = debug_line
                        continue
                    if c == '\x1b':  # ESC or arrow key sequence
                        seq = read_escape_sequence(
                            max_chars=32,
                            timeout_first=esc_timeout_first,
                            timeout_rest=esc_timeout_rest,
                        )
                        debug_log_file(f"key esc seq={seq!r} bytes={[ord(ch) for ch in seq]}")
                        if seq and seq[-1] in ("A", "B", "C", "D"):
                            suffix = seq[-1]
                        else:
                            suffix = None
                        if suffix == "A":  # Up arrow
                            if current_index > 0:
                                current_index = current_index - 1
                                user_interrupted = True
                                if paused:
                                    process.send_signal(signal.SIGCONT)
                                process.terminate()
                                break
                            continue
                        elif suffix == "B":  # Down arrow
                            if current_index < len(history) - 1:
                                current_index = current_index + 1
                                user_interrupted = True
                                if paused:
                                    process.send_signal(signal.SIGCONT)
                                process.terminate()
                                break
                            continue
                        if seq:
                            debug_log_file(f"unhandled esc sequence: {seq!r}")
                            continue
                        if paused:
                            process.send_signal(signal.SIGCONT)
                        process.terminate()
                        debug_log_file("exit via ESC")
                        return
                    if c == 'q' or c == 'Q':  # Also allow 'q' to quit
                        if paused:
                            process.send_signal(signal.SIGCONT)
                        process.terminate()
                        debug_log_file("exit via q/Q")
                        return
                if paused:
                    if (not highlight_enabled) and (not pause_message_shown):
                        print(f"{info_color}Paused. Press Space to resume.{reset_color}")
                        pause_message_shown = True
                    time.sleep(0.1)
                    continue
                if highlight_enabled:
                    if words_cache_path is not None and words_status != "ok":
                        payload = load_word_timestamps(words_cache_path, expected_model=highlight_model)
                        if payload is None:
                            words_status = "preparing"
                        elif payload.get("status") == "ok":
                            words_status = "ok"
                            words_error = None
                            words = payload.get("words") or None
                            current_word_index = -1
                            tokens = extract_text_tokens(cleaned_text)
                            word_to_token = align_words_to_text(words, tokens)
                            if word_to_token:
                                mapping_total = len(word_to_token)
                                mapping_mapped = sum(1 for idx in word_to_token if idx is not None)
                        else:
                            words_status = "error"
                            words_error = payload.get("error")
                            words = None
                            tokens = None
                            word_to_token = None
                            mapping_total = 0
                            mapping_mapped = 0
                    playhead = max(0.0, playback_offset + (time.monotonic() - playback_started_at) - playback_lag)
                if words:
                    while (current_word_index + 1) < len(words) and playhead >= words[current_word_index + 1].get("start", 0.0):
                        current_word_index += 1
                        debug_log_file(f"word advance -> idx={current_word_index} playhead={playhead:.3f}")
                    while current_word_index >= 0 and playhead < words[current_word_index].get("start", 0.0):
                        current_word_index -= 1
                        debug_log_file(f"word rewind -> idx={current_word_index} playhead={playhead:.3f}")
                new_context_span = None
                new_word_span = None
                if words and word_to_token and tokens and current_word_index >= 0:
                    spans = []
                    for wi in range(
                        current_word_index - highlight_context_words,
                        current_word_index + highlight_context_words + 1,
                    ):
                        if 0 <= wi < len(word_to_token):
                            token_idx = word_to_token[wi]
                            if token_idx is not None and 0 <= token_idx < len(tokens):
                                t = tokens[token_idx]
                                spans.append((t["start"], t["end"]))
                    if spans:
                        new_context_span = (min(s for s, _ in spans), max(e for _, e in spans))
                    token_idx = word_to_token[current_word_index]
                    if token_idx is not None and 0 <= token_idx < len(tokens):
                        t = tokens[token_idx]
                        new_word_span = (t["start"], t["end"])
                debug_line = highlight_debug_info(audio_file, words_cache_path, words_status, words_error)
                if ui_debug and words and current_word_index >= 0:
                    try:
                        w = str(words[current_word_index].get("word", "")).strip()
                        ws = float(words[current_word_index].get("start", 0.0))
                        we = float(words[current_word_index].get("end", 0.0))
                    except Exception:
                        w = ""
                        ws = None
                        we = None
                    extra = f" ph={playhead:.2f} lag={playback_lag:.2f} idx={current_word_index}/{len(words)}"
                    if w:
                        extra += f" w={w!r}"
                    if ws is not None and we is not None:
                        extra += f" ws={ws:.2f} we={we:.2f}"
                    extra += f" ctx={highlight_context_words}"
                    if text_debug_hash:
                        extra += f" txt={text_debug_hash}"
                    if mapping_total:
                        try:
                            cov = mapping_mapped / mapping_total if mapping_total else 0.0
                        except Exception:
                            cov = 0.0
                        extra += f" map={mapping_mapped}/{mapping_total}({cov:.0%})"
                    tok_idx = None
                    tok_text = None
                    if word_to_token and tokens and 0 <= current_word_index < len(word_to_token):
                        tok_idx = word_to_token[current_word_index]
                        if tok_idx is not None and 0 <= tok_idx < len(tokens):
                            t = tokens[tok_idx]
                            try:
                                tok_text = cleaned_text[t["start"]:t["end"]]
                            except Exception:
                                tok_text = None
                    if tok_text:
                        tok_text = tok_text.replace("\n", " ")
                        if len(tok_text) > 24:
                            tok_text = tok_text[:21] + "..."
                        extra += f" tok={tok_text!r}"
                    debug_line = (debug_line or "") + extra
                if (
                    new_context_span != current_highlight_span
                    or new_word_span != current_highlight_word_span
                    or words_status != last_words_status
                    or debug_line != last_debug_line
                    or paused != last_paused_state
                ):
                    current_highlight_span = new_context_span
                    current_highlight_word_span = new_word_span
                    last_words_status = words_status
                    last_debug_line = debug_line
                    last_paused_state = paused
                    if region_lines is not None:
                        region_lines = render_current_region(
                            cleaned_text,
                            highlight_spans=build_highlight_spans(current_highlight_span, current_highlight_word_span),
                            paused=paused,
                            debug_line=debug_line,
                        )
                    redraw_region(region_lines, region_line_count)
                    debug_log_file(f"highlight update word_idx={current_word_index} span={current_highlight_span} status={words_status}")
                # Check for new audio while playing
                if not audio_queue.empty() and current_index == len(history) - 1:
                    try:
                        new_chunk = audio_queue.get_nowait()
                        history.append(new_chunk)
                        debug_log_file(f"prefetched next chunk history={len(history)} queue_size={audio_queue.qsize()}")
                        if highlight_enabled:
                            try:
                                schedule_word_timestamps(new_chunk[0], new_chunk[1])
                            except Exception:
                                pass
                        debug_print(f"Pre-fetched next chunk while playing, history size: {len(history)}")
                    except queue.Empty:
                        pass
                time.sleep(0.01)
                
            # Only auto-advance if user didn't interrupt
            if not user_interrupted:
                # Audio finished playing
                debug_print(f"Audio finished for chunk {current_index + 1}")
                debug_log_file(f"audio finished idx={current_index}")
                
                # Check if we're at the last chunk
                is_at_last_chunk = (current_index == len(history) - 1 and audio_queue.empty() and generation_done)
                
                if is_at_last_chunk:
                    # Don't auto-advance from the last chunk
                    debug_print("At last chunk, stopping auto-advance")
                    print(f"\n{success_color}Reached end. Use ↑ arrow to replay previous chunks.{reset_color}\n")
                    # Wait for user input
                    while True:
                        if is_data():
                            c = read_char()
                            if c == '\x1b':  # ESC or arrow key sequence
                                seq = read_escape_sequence(
                                    max_chars=32,
                                    timeout_first=esc_timeout_first,
                                    timeout_rest=esc_timeout_rest,
                                )
                                debug_log_file(f"end key esc seq={seq!r} bytes={[ord(ch) for ch in seq]}")
                                suffix = seq[-1] if seq else None
                                if suffix == "A":  # Up arrow
                                    if current_index > 0:
                                        current_index = current_index - 1
                                        break
                                    continue
                                if suffix == "B":  # Down arrow at end
                                    continue  # Stay at last chunk
                                if seq:
                                    debug_log_file(f"end unhandled esc sequence: {seq!r}")
                                    continue
                                return  # Plain ESC
                            elif c == 'q' or c == 'Q':
                                debug_log_file("exit at end via q")
                                return
                        time.sleep(0.1)
                elif current_index < len(history) - 1:
                    # We already have the next chunk in history
                    current_index += 1
                    debug_print(f"Auto-advancing to chunk {current_index + 1} (already in history)")
                elif not audio_queue.empty():
                    # Try to get next chunk from queue
                    try:
                        new_chunk = audio_queue.get_nowait()
                        history.append(new_chunk)
                        if highlight_enabled:
                            try:
                                schedule_word_timestamps(new_chunk[0], new_chunk[1])
                            except Exception:
                                pass
                        current_index += 1
                        debug_print(f"Auto-advancing to chunk {current_index + 1} (fetched from queue)")
                    except queue.Empty:
                        debug_print("Queue was empty when trying to auto-advance")
            else:
                debug_print(f"User navigated to chunk {current_index + 1}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def graceful_exit(signal_received, frame):
    debug_print("Graceful exit initiated.")
    debug_log_file(f"graceful_exit signal={signal_received}")
    subprocess.call(['killall', 'afplay'])
    subprocess.call(['killall', 'ffplay'])
    # Clean up temp files
    temp_dir = Path.home() / ".whisper" / "temp"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    exit(0)

def is_audio_file(path: Path):
    """Basic audio extension check to avoid mis-detecting text clipboard content"""
    return path.suffix.lower() in _AUDIO_EXTS

def transcribe_audio_file(audio_path, model="gpt-4o-transcribe"):
    """Transcribe a local audio file using the Audio API"""
    client = OpenAI(timeout=300)
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )
            text = None
            if isinstance(response, dict):
                text = response.get("text")
            else:
                text = getattr(response, "text", None)
                if text is None:
                    try:
                        data = response.model_dump()
                    except Exception:
                        data = None
                    if isinstance(data, dict):
                        text = data.get("text")
            if text is None:
                print("Transcription response did not include text.")
                return None
            return text
    except FileNotFoundError:
        print(f"Audio file not found: {audio_path}")
        return None
    except openai.BadRequestError as e:
        print(f"Failed to transcribe audio due to bad request: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"Rate limit error during transcription: {e}")
        return None
    except Exception as e:
        print(f"Failed to transcribe audio due to error: {e}")
        return None

def load_word_timestamps(cache_path: Path, expected_model=None):
    """Load cached word timings.

    Returns None if cache missing or model mismatch.
    Otherwise returns a dict: {"status": "ok"|"error", "words": [...], "error": str|None, "created_at": float|None}.
    """
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if expected_model and data.get("transcription_model") != expected_model:
            return None

        status = str(data.get("status", "ok")).strip().lower() or "ok"
        created_at = data.get("created_at", None)
        error = data.get("error", None)
        raw_words = data.get("words", [])
        if not isinstance(raw_words, list):
            raw_words = []

        normalized = []
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            word = str(item.get("word", "")).strip()
            if not word:
                continue
            try:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
            except Exception:
                continue
            normalized.append({"word": word, "start": start, "end": end})

        if status == "ok" and not normalized:
            status = "error"
            if not error:
                error = "No word timings available"

        return {"status": status, "words": normalized, "error": error, "created_at": created_at}
    except Exception:
        return None

def save_word_timestamps(cache_path: Path, transcription_model: str, words, status="ok", error=None):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "version": 1,
        "transcription_model": transcription_model,
        "created_at": time.time(),
        "status": status,
        "error": error,
        "words": words,
    }
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    tmp_path.replace(cache_path)

def transcribe_audio_words(audio_path: Path, model: str = "gpt-4o-transcribe", timeout: int = 300):
    """Return word-level timestamps for an audio file (requires a model that supports verbose_json + word timestamps)."""
    client = OpenAI(timeout=timeout)
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
    except Exception as e:
        debug_print(f"Word-timestamp transcription failed: {e}")
        return None

    data = response
    if not isinstance(data, dict):
        try:
            data = response.model_dump()
        except Exception:
            data = None
    words = None
    if isinstance(data, dict):
        words = data.get("words")
    if words is None:
        words = getattr(response, "words", None)

    if not isinstance(words, list):
        segments = None
        if isinstance(data, dict):
            segments = data.get("segments")
        if segments is None:
            segments = getattr(response, "segments", None)
        if isinstance(segments, list):
            flattened = []
            for segment in segments:
                seg_words = None
                if isinstance(segment, dict):
                    seg_words = segment.get("words")
                else:
                    seg_words = getattr(segment, "words", None)
                if not isinstance(seg_words, list):
                    continue
                flattened.extend(seg_words)
            words = flattened

    if not isinstance(words, list) or not words:
        return None

    normalized = []
    for item in words:
        if isinstance(item, dict):
            word = str(item.get("word", "")).strip()
            start = item.get("start", 0.0)
            end = item.get("end", 0.0)
        else:
            word = str(getattr(item, "word", "")).strip()
            start = getattr(item, "start", 0.0)
            end = getattr(item, "end", 0.0)
        if not word:
            continue
        try:
            start = float(start)
            end = float(end)
        except Exception:
            continue
        normalized.append({"word": word, "start": start, "end": end})
    return normalized if normalized else None

def resolve_audio_path(audio_file_arg, clipboard_content):
    """Pick an audio file path from CLI or clipboard"""
    if audio_file_arg:
        candidate = audio_file_arg
    else:
        # Use the first line from clipboard to avoid trailing notes
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

def run_interactive_tts_flow(clipboard_content, combined_texts, args, model):
    """Interactive cached file-based playback using afplay/ffplay."""
    print(f"Processing {len(clipboard_content)} characters...")
    
    response_format = "mp3" if args.highlight else "aac"
    tts = TTSFile(voice=args.voice, model=model, instructions=args.instructions, response_format=response_format)
    audio_queue = queue.Queue()
    status_queue = queue.Queue()

    print(f"Split into {len(combined_texts)} chunks")
    
    generation_thread = threading.Thread(
        target=generate_audio_files_streaming,
        args=(combined_texts, tts, audio_queue, status_queue),
        daemon=True
    )
    generation_thread.start()
    play_audio_files_with_status(
        audio_queue,
        status_queue,
        tts,
        highlight=args.highlight,
        highlight_model=args.highlight_model,
        highlight_window=args.highlight_window,
        resume_rewind=args.resume_rewind,
        playhead_lag=args.playhead_lag,
        esc_timeout=args.esc_timeout,
        ui_debug=args.debug,
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='cbplay - Clipboard Play: auto-detect clipboard/file and transcribe or speak')
    parser.add_argument('--mode',
                        choices=['auto', 'tts', 'stt'],
                        default='auto',
                        help='Mode: auto (default, stt when clipboard/file is audio, otherwise tts), tts (clipboard text to speech), stt (audio file to text using gpt-4o-transcribe)')
    parser.add_argument('-v', '--voice', 
                        choices=['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 
                                 'nova', 'onyx', 'sage', 'shimmer', 'verse'],
                        default='nova',
                        help='Voice to use for text-to-speech (default: nova)')
    parser.add_argument('-a', '--audio-file',
                        help='Audio file to transcribe in stt mode. If omitted, clipboard content is treated as a path.')
    parser.add_argument('--transcription-model',
                        default='gpt-4o-mini-transcribe-2025-12-15',
                        help='Audio transcription model to use in stt mode (default: gpt-4o-mini-transcribe-2025-12-15)')
    parser.add_argument('--model',
                        default="gpt-4o-mini-tts-2025-12-15",
                        help='TTS model to use (default: gpt-4o-mini-tts-2025-12-15; legacy: tts-1-hd)')
    parser.add_argument('--highlight-tts-model',
                        default=None,
                        help='TTS model to use when --highlight is enabled (default: gpt-4o-mini-tts-2025-12-15).')
    parser.add_argument('--instructions',
                        default=DEFAULT_STREAMING_INSTRUCTIONS,
                        help='Delivery instructions sent to TTS (gpt-4o* TTS models only; empty string to disable).')
    parser.add_argument('--chunk-size',
                        type=int,
                        default=600,
                        help='Max characters per clipboard chunk before TTS splitting (default: 600).')
    parser.add_argument('--stream',
                        action='store_true',
                        help='Use async streaming playback (AsyncOpenAI + LocalAudioPlayer). Disables interactive UI.')
    parser.add_argument('--highlight',
                        action='store_true',
                        help='Highlight the currently spoken word (does an extra transcription call per chunk; cached). Only supported in interactive UI.')
    parser.add_argument('--highlight-model',
                        default='whisper-1',
                        help='Transcription model used for word-level timestamps when --highlight is enabled (word timestamps require whisper-1).')
    parser.add_argument('--highlight-window',
                        type=int,
                        default=1,
                        help='Words of context on each side of the highlighted word when --highlight is enabled (default: 1).')
    parser.add_argument('--resume-rewind',
                        type=float,
                        default=2.0,
                        help='Seconds to rewind on resume (helps compensate for player buffering; default: 2.0).')
    parser.add_argument('--playhead-lag',
                        type=float,
                        default=None,
                        help='Seconds to subtract from the internal playhead to better sync highlighting with audio output (default: 0.05 for afplay, 0.0 for ffplay).')
    parser.add_argument('--esc-timeout',
                        type=float,
                        default=None,
                        help='Seconds to wait after receiving ESC for an escape sequence (helps arrow keys under tmux/screen; defaults to 0.60 in tmux, otherwise 0.25).')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Show debug info in the interactive UI (e.g., highlight cache status and errors).')
    parser.add_argument('--debug-file',
                        default=None,
                        help='When set (or implied by --debug), append detailed diagnostics to this file (default: debug_cbplay.log in CWD).')
    args = parser.parse_args()

    if args.highlight and not args.highlight_tts_model:
        args.highlight_tts_model = "gpt-4o-mini-tts-2025-12-15"

    if args.highlight and args.highlight_model != "whisper-1":
        print("Note: word-level timestamps currently require whisper-1; overriding --highlight-model to whisper-1.")
        args.highlight_model = "whisper-1"

    # Configure debug file logging
    global DEBUG_FILE
    if args.debug_file or args.debug:
        path = args.debug_file if args.debug_file else "debug_cbplay.log"
        try:
            DEBUG_FILE = str(Path(path).expanduser().resolve())
            with open(DEBUG_FILE, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== cbplay session {datetime.now().isoformat()} =====\n")
        except Exception:
            DEBUG_FILE = None
    debug_log_file(f"args: {args}")

    if args.stream and args.highlight:
        print("--highlight is only supported in interactive UI. Remove --stream to use highlighting.")
        return
    
    signal.signal(signal.SIGINT, graceful_exit)

    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    debug_print("Starting script...")
    debug_print(f"Using voice: {args.voice}")
    
    print("Reading clipboard content...")
    clipboard_content = get_clipboard_content()

    # Determine effective mode
    audio_path = None
    if args.mode == 'auto':
        candidate_path = resolve_audio_path(args.audio_file, clipboard_content)
        if candidate_path and is_audio_file(candidate_path):
            audio_path = candidate_path
            effective_mode = 'stt'
        else:
            effective_mode = 'tts'
    else:
        effective_mode = args.mode
        if effective_mode == 'stt':
            audio_path = resolve_audio_path(args.audio_file, clipboard_content)

    if effective_mode == 'stt':
        if not audio_path:
            print("No valid audio file found. Provide --audio-file or copy an audio file path to the clipboard.")
            return
        print(f"Transcribing {audio_path} using {args.transcription_model}...")
        transcript = transcribe_audio_file(audio_path, model=args.transcription_model)
        if transcript is None:
            return
        cleaned = clean_text_for_display(transcript)
        print("\nTranscript:\n")
        print(cleaned)
        try:
            pyperclip.copy(transcript)
            print("\nTranscript copied to clipboard.")
        except pyperclip.PyperclipException:
            debug_print("Could not copy transcript to clipboard.")
        return
    
    # Default: TTS mode
    if not clipboard_content.strip():
        print("Clipboard is empty!")
        return
    
    try:
        chunk_size = int(args.chunk_size)
    except Exception:
        chunk_size = 600
    if chunk_size <= 0:
        chunk_size = 600
    combined_texts = [prepare_text_for_tts(t) for t in split_text_intelligently(clipboard_content, max_chars=chunk_size)]
    combined_texts = [t for t in combined_texts if t]

    # Default: interactive UI unless --stream is requested
    if not args.stream:
        tts_model = args.highlight_tts_model if args.highlight else args.model
        run_interactive_tts_flow(clipboard_content, combined_texts, args, tts_model)
        return

    # Streaming path (falls back to interactive UI on errors)
    try:
        print(f"Streaming {len(combined_texts)} chunk(s) with {args.model}/{args.voice}...")
        asyncio.run(stream_tts_chunks(
            combined_texts=combined_texts,
            voice=args.voice,
            model=args.model,
            instructions=args.instructions,
            response_format="pcm",
            timeout=240,
        ))
    except Exception as e:
        print(f"Streaming path failed ({e}). Falling back to interactive UI...")
        run_interactive_tts_flow(clipboard_content, combined_texts, args, args.model)

if __name__ == "__main__":
    main()
