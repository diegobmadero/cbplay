"""Audio playback coordination for cbplay."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Optional, List, Callable

from cbplay_tts import TTSProvider
from cbplay_stt import transcribe_audio_words, load_word_timestamps, save_word_timestamps
from cbplay_utils import debug_print, debug_log_file, get_audio_duration, ffplay_available, prepare_text_for_tts


def generate_audio_files_streaming(combined_texts: List[str], tts: TTSProvider, audio_queue: queue.Queue, status_queue: queue.Queue):
    debug_print(f"Starting generation for {len(combined_texts)} chunks")
    if not combined_texts:
        debug_print("No valid text found to generate audio.")
        status_queue.put(("done", 0, 0))
        return
    
    total = len(combined_texts)
    completed = 0
    
    cached_count = 0
    for i, text in enumerate(combined_texts):
        text_hash = tts._hash_text(text)
        cached_file = tts.cache_dir / f"{text_hash}.{tts.response_format}"
        if text_hash in tts.cache_index and cached_file.exists():
            cached_count += 1
            debug_print(f"Chunk {i+1} is cached: {text[:50]}...")
    
    debug_print(f"Found {cached_count} cached chunks out of {total}")
    status_queue.put(("cached", cached_count, total))
    
    max_workers = min(3, total)
    
    def generate_single(index_text):
        index, text = index_text
        debug_print(f"Generating chunk {index + 1}/{total}")
        text_hash = tts._hash_text(text)
        unique_filename = f"tts_clipboard_{text_hash}.{tts.response_format}"
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
        futures = []
        for i, text in enumerate(combined_texts):
            future = executor.submit(generate_single, (i, text))
            futures.append((i, future))
        
        pending_results = {}
        next_index = 0
        
        for i, future in futures:
            result = future.result()
            if result:
                index, audio_file, text = result
                pending_results[index] = (audio_file, text)
                completed += 1
                status_queue.put(("progress", completed, total))
                
                while next_index in pending_results:
                    audio_queue.put(pending_results[next_index])
                    debug_print(f"Added chunk {next_index + 1} to queue")
                    del pending_results[next_index]
                    next_index += 1
        
        for i in sorted(pending_results.keys()):
            audio_queue.put(pending_results[i])
            debug_print(f"Added remaining chunk {i + 1} to queue")
    
    debug_print(f"Generation complete. Added {completed} chunks to queue")
    status_queue.put(("done", completed, total))


def start_audio_player(audio_path, start_at: float = 0.0):
    if ffplay_available():
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


class PlaybackController:
    def __init__(
        self,
        tts: TTSProvider,
        highlight: bool = False,
        highlight_model: str = "gpt-4o-transcribe",
        highlight_window: int = 8,
        resume_rewind: float = 2.0,
        playhead_lag: Optional[float] = None,
        esc_timeout: Optional[float] = None,
        ui_debug: bool = False,
    ):
        self.tts = tts
        self.highlight_enabled = highlight
        self.highlight_model = highlight_model
        self.highlight_window = highlight_window
        self.rewind_padding = max(0.0, resume_rewind)
        self.ui_debug = ui_debug
        
        use_ffplay = ffplay_available()
        if playhead_lag is None:
            self.playhead_lag = 0.0 if use_ffplay else 0.05
        else:
            self.playhead_lag = max(0.0, playhead_lag)
        
        esc_timeout_first = 0.60 if os.getenv("TMUX") else 0.25
        if esc_timeout is not None:
            esc_timeout_first = max(0.01, esc_timeout)
        else:
            esc_timeout_env = os.getenv("CBPLAY_ESC_TIMEOUT")
            if esc_timeout_env:
                try:
                    esc_timeout_first = max(0.01, float(esc_timeout_env))
                except Exception:
                    pass
        self.esc_timeout_first = esc_timeout_first
        self.esc_timeout_rest = 0.01
        
        self.highlight_inflight = set()
        self.highlight_lock = threading.Lock()
        self.highlight_task_queue = queue.Queue() if highlight else None
        
        if highlight:
            for _ in range(2):
                threading.Thread(target=self._highlight_worker, daemon=True).start()
    
    def _highlight_worker(self):
        while True:
            audio_path, cache_path, key = self.highlight_task_queue.get()
            try:
                existing = load_word_timestamps(cache_path, expected_model=self.highlight_model)
                if existing and existing.get("status") == "ok":
                    continue
                words = transcribe_audio_words(Path(audio_path), model=self.highlight_model)
                if words:
                    save_word_timestamps(cache_path, self.highlight_model, words, status="ok", error=None)
                else:
                    save_word_timestamps(cache_path, self.highlight_model, [], status="error", error="No word timestamps returned")
            except Exception as e:
                debug_print(f"Highlight transcription worker failed: {e}")
                try:
                    save_word_timestamps(cache_path, self.highlight_model, [], status="error", error=str(e))
                except Exception:
                    pass
            finally:
                with self.highlight_lock:
                    self.highlight_inflight.discard(key)
                self.highlight_task_queue.task_done()
    
    def word_cache_path_for_text(self, text: str) -> Path:
        return self.tts.cache_dir / f"{self.tts._hash_text(text)}.words.json"
    
    def schedule_word_timestamps(self, audio_path, text: str) -> Path:
        cache_path = self.word_cache_path_for_text(text)
        if cache_path.exists():
            existing = load_word_timestamps(cache_path, expected_model=self.highlight_model)
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
        with self.highlight_lock:
            if key in self.highlight_inflight:
                return cache_path
            self.highlight_inflight.add(key)
        self.highlight_task_queue.put((audio_path, cache_path, key))
        return cache_path


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[''][A-Za-z0-9]+)*")

NUMBER_NORM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}


def normalize_for_match(word: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", (word or "").lower())
    return NUMBER_NORM_MAP.get(cleaned, cleaned)


def extract_text_tokens(text: str) -> List[dict]:
    tokens = []
    for match in TOKEN_RE.finditer(text or ""):
        raw = match.group(0)
        norm = normalize_for_match(raw)
        if not norm:
            continue
        tokens.append({"norm": norm, "start": match.start(), "end": match.end()})
    return tokens


def align_words_to_text(words, tokens, lookahead: int = 12):
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

    if n * m > 600_000:
        token_index = 0
        for i, w_norm in enumerate(word_norms):
            if not w_norm:
                continue
            end = min(token_index + lookahead, m)
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
