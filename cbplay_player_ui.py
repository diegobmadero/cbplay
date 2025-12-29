"""UI-based audio playback for cbplay (ANSI and curses modes)."""

import curses
import hashlib
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
from pathlib import Path
from typing import Optional, List

from cbplay_tts import TTSProvider
from cbplay_stt import load_word_timestamps, save_word_timestamps, transcribe_audio_words
from cbplay_utils import debug_print, debug_log_file, get_audio_duration, ffplay_available
from cbplay_ui_ansi import AnsiTheme, UiLayout, AnsiKaraokeRenderer, build_highlight_spans
from cbplay_player import (
    PlaybackController,
    start_audio_player,
    extract_text_tokens,
    align_words_to_text,
)


def play_audio_files_with_status(
    audio_queue,
    status_queue,
    tts: TTSProvider,
    highlight: bool = False,
    highlight_model: str = "gpt-4o-transcribe",
    highlight_window: int = 8,
    resume_rewind: float = 2.0,
    playhead_lag: Optional[float] = None,
    esc_timeout: Optional[float] = None,
    ui_debug: bool = False,
    ui_mode: str = "ansi",
    all_text_chunks: Optional[List[str]] = None,
):
    if not sys.stdin.isatty() or os.getenv('NOTTY') == '1':
        print("Not running in a terminal, audio generation only mode")
        total_received = 0
        while True:
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
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
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                            total_received += 1
                        except queue.Empty:
                            break
                    print(f"Total chunks received: {total_received}")
                    break
            time.sleep(0.1)
        return
    
    if ui_mode == "curses":
        return _play_curses_mode(
            audio_queue, status_queue, tts,
            highlight=highlight, highlight_model=highlight_model,
            highlight_window=highlight_window, resume_rewind=resume_rewind,
            playhead_lag=playhead_lag, ui_debug=ui_debug,
            all_text_chunks=all_text_chunks,
        )
    
    return _play_ansi_mode(
        audio_queue, status_queue, tts,
        highlight=highlight, highlight_model=highlight_model,
        highlight_window=highlight_window, resume_rewind=resume_rewind,
        playhead_lag=playhead_lag, esc_timeout=esc_timeout,
        ui_debug=ui_debug, all_text_chunks=all_text_chunks,
    )


def _play_ansi_mode(
    audio_queue,
    status_queue,
    tts: TTSProvider,
    highlight: bool,
    highlight_model: str,
    highlight_window: int,
    resume_rewind: float,
    playhead_lag: Optional[float],
    esc_timeout: Optional[float],
    ui_debug: bool,
    all_text_chunks: Optional[List[str]],
):
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        stdin_fd = sys.stdin.fileno()
        
        active_color = '\033[93m'
        prev_color = '\033[90m'
        reset_color = '\033[0m'
        info_color = '\033[96m'
        success_color = '\033[92m'
        
        term_width = shutil.get_terminal_size((100, 20)).columns
        display_width = max(20, min(100, term_width))
        use_ffplay = ffplay_available()
        rewind_padding = max(0.0, float(resume_rewind or 0.0))
        highlight_enabled = bool(highlight)
        debug_log_file(f"[ANSI] highlight_enabled={highlight_enabled}, highlight_model={highlight_model}")
        ui_debug = bool(ui_debug)
        try:
            highlight_context_words = max(0, int(highlight_window or 0))
        except Exception:
            highlight_context_words = 1

        if playhead_lag is None:
            playhead_lag_seconds = 0.0 if use_ffplay else 0.05
        else:
            try:
                playhead_lag_seconds = max(0.0, float(playhead_lag))
            except Exception:
                playhead_lag_seconds = 0.0 if use_ffplay else 0.05

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
        
        highlight_inflight = set()
        highlight_lock = threading.Lock()
        highlight_task_queue = queue.Queue() if highlight_enabled else None

        def read_char():
            try:
                data = os.read(stdin_fd, 1)
            except Exception:
                return ""
            if not data:
                return ""
            return chr(data[0])

        def read_escape_sequence(max_chars=32, timeout_first=0.05, timeout_rest=0.005):
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
                        save_word_timestamps(cache_path, highlight_model, [], status="error", error="No word timestamps returned")
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
            debug_log_file(f"[ANSI] Starting 2 highlight_worker threads")
            for _ in range(2):
                threading.Thread(target=highlight_worker, daemon=True).start()

        def schedule_word_timestamps(audio_path, text):
            debug_log_file(f"[ANSI] schedule_word_timestamps called: audio={audio_path}, text={text[:50]}...")
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

        BOX_TL, BOX_TR, BOX_BL, BOX_BR = '╭', '╮', '╰', '╯'
        BOX_H, BOX_V = '─', '│'
        progress_full = "█"
        progress_empty = "░"
        status_icon_play = "▶"
        status_icon_pause = "⏸"
        
        theme = AnsiTheme(
            active_color=active_color, prev_color=prev_color,
            info_color=info_color, reset_color=reset_color,
            box_tl=BOX_TL, box_tr=BOX_TR, box_bl=BOX_BL, box_br=BOX_BR,
            box_h=BOX_H, box_v=BOX_V,
            progress_full=progress_full, progress_empty=progress_empty,
        )
        layout = UiLayout(term_width=term_width, display_width=display_width)
        renderer = AnsiKaraokeRenderer(theme, layout)

        def render_minimap(current_idx: int, total: int, max_dots: int = 20) -> str:
            if total <= 0:
                return ""
            if total <= max_dots:
                dots = []
                for i in range(total):
                    dots.append("●" if i == current_idx else "○")
                return "".join(dots)
            result = []
            for i in range(min(3, total)):
                result.append("●" if i == current_idx else "○")
            if current_idx > 4:
                result.append("…")
            if 3 <= current_idx < total - 3:
                for i in range(max(3, current_idx - 1), min(total - 3, current_idx + 2)):
                    result.append("●" if i == current_idx else "○")
            if current_idx < total - 5:
                result.append("…")
            for i in range(max(total - 3, 0), total):
                if i > current_idx + 1 or i < 3:
                    continue
                result.append("●" if i == current_idx else "○")
            for i in range(max(total - 3, current_idx + 2), total):
                result.append("●" if i == current_idx else "○")
            return "".join(result)

        def render_controls_bar(paused: bool = False, chunk_idx: int = 0, total_chunks: int = 0) -> str:
            if paused:
                controls = "␣ Resume  │  ↑↓ Navigate  │  Q/Esc Exit"
            else:
                controls = "␣ Pause  │  ↑↓ Navigate  │  Q/Esc Exit"
            if total_chunks > 1:
                minimap = render_minimap(chunk_idx, total_chunks)
                return f"{info_color}{minimap}  {controls}{reset_color}"
            return f"{info_color}{controls}{reset_color}"

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
            parts = [f"dbg st={status}", f"m={highlight_model}",
                     f"audio={audio_ext or '?'}", f"cache={'y' if cache_exists else 'n'}",
                     f"q={qsize}", f"in={inflight}"]
            if cache_name:
                parts.append(f"file={cache_name}")
            if err:
                parts.append(f"err={err}")
            return " ".join(parts)

        def is_data():
            try:
                return bool(select.select([stdin_fd], [], [], 0)[0])
            except Exception:
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

        def clear_screen():
            os.system('cls' if os.name == 'nt' else 'clear')

        history = []
        current_index = -1
        generation_done = False
        total_chunks = 0
        generated_chunks = 0
        start_time = time.time()

        while True:
            clear_screen()
            debug_log_file(f"ui_loop start idx={current_index} hist={len(history)} gen_done={generation_done}")
            
            while not status_queue.empty():
                try:
                    status = status_queue.get_nowait()
                    if status[0] == "cached":
                        print(f"{success_color}Found {status[1]} cached chunks out of {status[2]} total{reset_color}")
                    elif status[0] == "progress":
                        generated_chunks = status[1]
                        total_chunks = status[2]
                    elif status[0] == "done":
                        generation_done = True
                        total_chunks = status[2]
                        elapsed = time.time() - start_time
                        print(f"{success_color}Generation complete! Generated {status[1]} chunks in {elapsed:.1f}s{reset_color}")
                except queue.Empty:
                    break
            
            if not generation_done and total_chunks > 0:
                percent = (generated_chunks / total_chunks) * 100
                bar_length = 40
                filled = int(bar_length * generated_chunks / total_chunks)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"Generating: [{bar}] {percent:.0f}% ({generated_chunks}/{total_chunks})\n")
            
            effective_total = total_chunks if total_chunks > 0 else len(history)
            print(render_controls_bar(paused=False, chunk_idx=max(0, current_index), total_chunks=effective_total))
            print()
            
            if 0 <= current_index < len(history):
                current_text_chunk = history[current_index]
            elif not audio_queue.empty():
                try:
                    current_text_chunk = audio_queue.get_nowait()
                    history.append(current_text_chunk)
                    if highlight_enabled:
                        try:
                            schedule_word_timestamps(current_text_chunk[0], current_text_chunk[1])
                        except Exception:
                            pass
                    if current_index == -1:
                        current_index = 0
                    else:
                        current_index = len(history) - 1
                except queue.Empty:
                    pass
            else:
                if generation_done and len(history) == 0:
                    print("No audio files were generated.")
                    break
                elif current_index >= len(history) and generation_done:
                    print(f"{success_color}Playback complete!{reset_color}")
                    break
                else:
                    if current_index == -1:
                        debug_print("Waiting for first audio chunk...")
                    time.sleep(0.2)
                    continue

            if current_index > 0:
                prev_file, prev_text = history[current_index - 1]
                prev_lines = renderer.render_previous_section(prev_text, max_lines=4)
                for line in prev_lines:
                    print(line)
                print()
    
            if 0 <= current_index < len(history):
                audio_file, original_text_chunk = history[current_index]
                cleaned_text = original_text_chunk
                audio_duration = get_audio_duration(audio_file)
                effective_total = total_chunks if total_chunks > 0 else len(history)

                words_cache_path = None
                words = None
                words_error = None
                words_status = "preparing"
                current_word_index = -1
                tokens = None
                word_to_token = None
                current_highlight_span = None
                current_highlight_word_span = None
                current_dim_span = None
                region_lines = None
                region_line_count = 0
                last_debug_line = None

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

                if words and words_status == "ok":
                    try:
                        if float(words[0].get("start", 0.0)) <= 0.05:
                            current_word_index = 0
                    except Exception:
                        pass
                    tokens = extract_text_tokens(cleaned_text)
                    word_to_token = align_words_to_text(words, tokens)

                last_debug_line = highlight_debug_info(audio_file, words_cache_path, words_status, words_error)
                region_lines = renderer.render_current_region(
                    cleaned_text,
                    highlight_spans=build_highlight_spans(current_highlight_span, current_highlight_word_span, current_dim_span),
                    paused=False, debug_line=last_debug_line,
                    chunk_index=current_index, total_chunks=effective_total,
                    playhead=0.0, duration=audio_duration, status_icon=status_icon_play,
                )
                region_line_count = len(region_lines)
                for i, ln in enumerate(region_lines):
                    sys.stdout.write((ln or ""))
                    if i < len(region_lines) - 1:
                        sys.stdout.write("\n")
                sys.stdout.flush()

            playback_offset = 0.0
            process = start_audio_player(audio_file, start_at=playback_offset)
            playback_started_at = time.monotonic()
            playback_lag = playhead_lag_seconds
            playhead = 0.0
            paused_at = None
            user_interrupted = False
            paused = False

            while process.poll() is None:
                if is_data():
                    c = read_char()
                    if c == ' ':
                        now = time.monotonic()
                        current_playhead = max(0.0, playback_offset + (now - playback_started_at) - playback_lag)
                        if paused:
                            resume_from = paused_at if paused_at is not None else current_playhead
                            target_offset = max(resume_from - rewind_padding, 0.0)
                            if use_ffplay:
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
                                process = start_audio_player(audio_file, start_at=target_offset)
                                playback_offset = target_offset
                                playback_lag = playhead_lag_seconds
                            else:
                                process.send_signal(signal.SIGCONT)
                                playback_offset = resume_from
                                playback_lag = 0.0
                            playback_started_at = time.monotonic()
                            paused = False
                            paused_at = None
                        else:
                            process.send_signal(signal.SIGSTOP)
                            paused_at = current_playhead
                            paused = True
                        continue
                    if c == '\x1b':
                        seq = read_escape_sequence(max_chars=32, timeout_first=esc_timeout_first, timeout_rest=esc_timeout_rest)
                        suffix = seq[-1] if seq and seq[-1] in ("A", "B", "C", "D") else None
                        if suffix == "A":
                            if current_index > 0:
                                current_index -= 1
                                user_interrupted = True
                                if paused:
                                    process.send_signal(signal.SIGCONT)
                                process.terminate()
                                break
                            continue
                        elif suffix == "B":
                            if current_index < len(history) - 1:
                                current_index += 1
                                user_interrupted = True
                                if paused:
                                    process.send_signal(signal.SIGCONT)
                                process.terminate()
                                break
                            continue
                        if seq:
                            continue
                        if paused:
                            process.send_signal(signal.SIGCONT)
                        process.terminate()
                        return
                    if c in ('q', 'Q'):
                        if paused:
                            process.send_signal(signal.SIGCONT)
                        process.terminate()
                        return

                if paused:
                    time.sleep(0.1)
                    continue

                if highlight_enabled and words_cache_path is not None and words_status != "ok":
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
                    else:
                        words_status = "error"
                        words_error = payload.get("error")

                if highlight_enabled:
                    playhead = max(0.0, playback_offset + (time.monotonic() - playback_started_at) - playback_lag)

                if words:
                    while (current_word_index + 1) < len(words) and playhead >= words[current_word_index + 1].get("start", 0.0):
                        current_word_index += 1
                    while current_word_index >= 0 and playhead < words[current_word_index].get("start", 0.0):
                        current_word_index -= 1

                new_context_span = None
                new_word_span = None
                new_dim_span = None
                if words and word_to_token and tokens and current_word_index >= 0:
                    spans = []
                    for wi in range(current_word_index - highlight_context_words, current_word_index + highlight_context_words + 1):
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
                        if t["start"] > 0:
                            new_dim_span = (0, t["start"])

                debug_line = highlight_debug_info(audio_file, words_cache_path, words_status, words_error)

                if (new_context_span != current_highlight_span or new_word_span != current_highlight_word_span or
                    new_dim_span != current_dim_span or debug_line != last_debug_line):
                    current_highlight_span = new_context_span
                    current_highlight_word_span = new_word_span
                    current_dim_span = new_dim_span
                    last_debug_line = debug_line
                    if region_lines is not None:
                        region_lines = renderer.render_current_region(
                            cleaned_text,
                            highlight_spans=build_highlight_spans(current_highlight_span, current_highlight_word_span, current_dim_span),
                            paused=paused, debug_line=debug_line,
                            chunk_index=current_index, total_chunks=effective_total,
                            playhead=playhead, duration=audio_duration,
                            status_icon=status_icon_pause if paused else status_icon_play,
                        )
                    redraw_region(region_lines, region_line_count)

                if not audio_queue.empty() and current_index == len(history) - 1:
                    try:
                        new_chunk = audio_queue.get_nowait()
                        history.append(new_chunk)
                        if highlight_enabled:
                            try:
                                schedule_word_timestamps(new_chunk[0], new_chunk[1])
                            except Exception:
                                pass
                    except queue.Empty:
                        pass

                time.sleep(0.01)
                
            if not user_interrupted:
                is_at_last_chunk = (current_index == len(history) - 1 and audio_queue.empty() and generation_done)
                if is_at_last_chunk:
                    print(f"\n{success_color}Reached end. Use ↑ arrow to replay previous chunks.{reset_color}\n")
                    while True:
                        if is_data():
                            c = read_char()
                            if c == '\x1b':
                                seq = read_escape_sequence(max_chars=32, timeout_first=esc_timeout_first, timeout_rest=esc_timeout_rest)
                                suffix = seq[-1] if seq else None
                                if suffix == "A" and current_index > 0:
                                    current_index -= 1
                                    break
                                if not seq:
                                    return
                            elif c in ('q', 'Q'):
                                return
                        time.sleep(0.1)
                elif current_index < len(history) - 1:
                    current_index += 1
                elif not audio_queue.empty():
                    try:
                        new_chunk = audio_queue.get_nowait()
                        history.append(new_chunk)
                        if highlight_enabled:
                            try:
                                schedule_word_timestamps(new_chunk[0], new_chunk[1])
                            except Exception:
                                pass
                        current_index += 1
                    except queue.Empty:
                        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def _play_curses_mode(
    audio_queue,
    status_queue,
    tts: TTSProvider,
    highlight: bool,
    highlight_model: str,
    highlight_window: int,
    resume_rewind: float,
    playhead_lag: Optional[float],
    ui_debug: bool,
    all_text_chunks: Optional[List[str]],
):
    from cbplay_ui_curses import CursesKaraokeScreen
    
    use_ffplay = ffplay_available()
    rewind_padding = max(0.0, float(resume_rewind or 0.0))
    highlight_enabled = bool(highlight)
    debug_log_file(f"[CURSES] highlight_enabled={highlight_enabled}, highlight_model={highlight_model}")
    
    try:
        highlight_context_words = max(0, int(highlight_window or 0))
    except Exception:
        highlight_context_words = 1

    if playhead_lag is None:
        playhead_lag_seconds = 0.0 if use_ffplay else 0.05
    else:
        try:
            playhead_lag_seconds = max(0.0, float(playhead_lag))
        except Exception:
            playhead_lag_seconds = 0.0 if use_ffplay else 0.05

    highlight_inflight = set()
    highlight_lock = threading.Lock()
    highlight_task_queue = queue.Queue() if highlight_enabled else None

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
                debug_log_file(f"[CURSES] highlight_worker got words={len(words) if words else 0}")
                if words:
                    save_word_timestamps(cache_path, highlight_model, words, status="ok", error=None)
                    debug_log_file(f"[CURSES] highlight_worker saved {len(words)} words to {cache_path}")
                else:
                    save_word_timestamps(cache_path, highlight_model, [], status="error", error="No word timestamps returned")
                    debug_log_file(f"[CURSES] highlight_worker saved error (no words)")
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
        debug_log_file(f"[CURSES] Starting 2 highlight_worker threads")
        for _ in range(2):
            threading.Thread(target=highlight_worker, daemon=True).start()

    def schedule_word_timestamps(audio_path, text):
        debug_log_file(f"[CURSES] schedule_word_timestamps called: audio={audio_path}, text={text[:50] if text else '?'}...")
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

    def curses_main(stdscr):
        screen = CursesKaraokeScreen(stdscr, debug=ui_debug)
        stdscr.nodelay(True)
        stdscr.timeout(0)

        history = []
        current_index = -1
        generation_done = False
        total_chunks = 0
        generated_chunks = 0
        start_time = time.time()
        needs_full_render = True
        full_lines = []
        chunk_line_ranges = []
        line_ranges_by_chunk = []
        pad_top = 0
        last_history_len = 0
        last_current_index = None

        def get_display_chunks():
            if all_text_chunks:
                return list(all_text_chunks)
            return [item[1] for item in history]

        def effective_total_count():
            if total_chunks > 0:
                return total_chunks
            if all_text_chunks:
                return len(all_text_chunks)
            return len(history)

        def compute_pad_top():
            nonlocal pad_top
            if current_index < 0 or current_index >= len(chunk_line_ranges):
                pad_top = 0
                return
            start, end = chunk_line_ranges[current_index]
            if end - start >= screen.body_height:
                pad_top = max(0, start)
                return
            target = max(0, start - max(1, int(screen.body_height * 0.2)))
            max_top = max(0, screen.pad_height - screen.body_height)
            pad_top = min(target, max_top)

        def rebuild_fullpage():
            nonlocal full_lines, chunk_line_ranges, line_ranges_by_chunk, needs_full_render
            display_chunks = get_display_chunks()
            full_lines, chunk_line_ranges, line_ranges_by_chunk = screen.build_fullpage_lines_with_ranges(display_chunks, current_index)
            screen.draw_pad_lines(full_lines)
            compute_pad_top()
            needs_full_render = False

        def redraw_current_chunk_base():
            if current_index < 0 or current_index >= len(chunk_line_ranges):
                return
            start, end = chunk_line_ranges[current_index]
            for pad_line in range(start, end):
                if 0 <= pad_line < len(full_lines):
                    text, attr = full_lines[pad_line]
                    screen._safe_addnstr(screen.body_pad, pad_line, 0, text, attr)

        def apply_highlights(dim_span, highlight_span, word_span):
            if not highlight_enabled:
                return
            if current_index < 0 or current_index >= len(chunk_line_ranges):
                return
            if current_index >= len(line_ranges_by_chunk):
                return
            redraw_current_chunk_base()
            line_ranges = line_ranges_by_chunk[current_index]
            chunk_start_line = chunk_line_ranges[current_index][0]
            if dim_span:
                screen.apply_span(
                    chunk_start_line,
                    line_ranges,
                    dim_span[0],
                    dim_span[1],
                    full_lines,
                    screen.colors.get("spoken", 0),
                )
            if highlight_span:
                screen.apply_span(
                    chunk_start_line,
                    line_ranges,
                    highlight_span[0],
                    highlight_span[1],
                    full_lines,
                    screen.colors.get("current", 0) | curses.A_UNDERLINE,
                )
            if word_span:
                screen.apply_span(
                    chunk_start_line,
                    line_ranges,
                    word_span[0],
                    word_span[1],
                    full_lines,
                    screen.colors.get("current_word", 0),
                )

        def build_status_line(paused: bool, playhead: float, duration: float):
            if not generation_done and total_chunks > 0:
                percent = (generated_chunks / total_chunks) * 100
                bar_length = 20
                filled = int(bar_length * generated_chunks / total_chunks)
                bar = "#" * filled + "-" * (bar_length - filled)
                return f"Generating: [{bar}] {percent:.0f}% ({generated_chunks}/{total_chunks})"
            total = effective_total_count()
            if current_index >= 0 and total > 0:
                icon = "⏸" if paused else "▶"
                line = f"{icon} Now Playing [{current_index + 1}/{total}]"
                if duration > 0:
                    line += f"  {playhead:.1f}s/{duration:.1f}s"
                return line
            return "Waiting for audio..."

        def refresh_ui(paused: bool, playhead: float, duration: float, debug_line: str):
            status_line = build_status_line(paused, playhead, duration)
            controls_line = "Space Pause  |  ↑↓ Navigate  |  Q/Esc Exit"
            if paused:
                controls_line = "Space Resume |  ↑↓ Navigate  |  Q/Esc Exit"
            screen.draw_header([status_line, controls_line])
            screen.draw_footer(debug_line or "")
            screen.refresh(pad_top)

        while True:
            if screen.handle_resize():
                needs_full_render = True

            while not status_queue.empty():
                try:
                    status = status_queue.get_nowait()
                    if status[0] == "cached":
                        generated_chunks = status[1]
                        total_chunks = status[2]
                    elif status[0] == "progress":
                        generated_chunks = status[1]
                        total_chunks = status[2]
                    elif status[0] == "done":
                        generation_done = True
                        total_chunks = status[2]
                except queue.Empty:
                    break

            if current_index > 0 and current_index >= len(history):
                current_index = len(history) - 1

            if 0 <= current_index < len(history):
                current_text_chunk = history[current_index]
            elif not audio_queue.empty():
                try:
                    current_text_chunk = audio_queue.get_nowait()
                    history.append(current_text_chunk)
                    if highlight_enabled:
                        try:
                            schedule_word_timestamps(current_text_chunk[0], current_text_chunk[1])
                        except Exception:
                            pass
                    if current_index == -1:
                        current_index = 0
                    else:
                        current_index = len(history) - 1
                except queue.Empty:
                    current_text_chunk = None
            else:
                current_text_chunk = None

            history_len = len(history)
            if history_len != last_history_len:
                last_history_len = history_len
                needs_full_render = True
            if current_index != last_current_index:
                last_current_index = current_index
                needs_full_render = True

            if needs_full_render:
                rebuild_fullpage()

            if current_text_chunk is None:
                refresh_ui(False, 0.0, 0.0, "")
                time.sleep(0.1)
                if generation_done and len(history) == 0:
                    break
                continue

            audio_file, original_text_chunk = current_text_chunk
            debug_log_file(f"[CURSES] Playing chunk: audio={audio_file}, text={original_text_chunk[:50] if original_text_chunk else '?'}...")
            cleaned_text = original_text_chunk
            audio_duration = get_audio_duration(audio_file)

            words_cache_path = None
            words = None
            words_error = None
            words_status = "preparing"
            current_word_index = -1
            tokens = None
            word_to_token = None
            current_highlight_span = None
            current_highlight_word_span = None
            current_dim_span = None

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

            if words and words_status == "ok":
                try:
                    if float(words[0].get("start", 0.0)) <= 0.05:
                        current_word_index = 0
                except Exception:
                    pass
                tokens = extract_text_tokens(cleaned_text)
                word_to_token = align_words_to_text(words, tokens)

            apply_highlights(current_dim_span, current_highlight_span, current_highlight_word_span)
            refresh_ui(False, 0.0, audio_duration, "")

            playback_offset = 0.0
            process = start_audio_player(audio_file, start_at=playback_offset)
            playback_started_at = time.monotonic()
            playback_lag = playhead_lag_seconds
            playhead = 0.0
            paused_at = None
            user_interrupted = False
            paused = False

            while process.poll() is None:
                if screen.handle_resize():
                    needs_full_render = True
                    rebuild_fullpage()
                    apply_highlights(current_dim_span, current_highlight_span, current_highlight_word_span)
                    refresh_ui(paused, playhead, audio_duration, "")

                ch = stdscr.getch()
                if ch != -1:
                    if ch == ord(' '):
                        now = time.monotonic()
                        current_playhead = max(0.0, playback_offset + (now - playback_started_at) - playback_lag)
                        if paused:
                            resume_from = paused_at if paused_at is not None else current_playhead
                            target_offset = max(resume_from - rewind_padding, 0.0)
                            if use_ffplay:
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
                                process = start_audio_player(audio_file, start_at=target_offset)
                                playback_offset = target_offset
                                playback_lag = playhead_lag_seconds
                            else:
                                process.send_signal(signal.SIGCONT)
                                playback_offset = resume_from
                                playback_lag = 0.0
                            playback_started_at = time.monotonic()
                            paused = False
                            paused_at = None
                            refresh_ui(False, playback_offset, audio_duration, "")
                        else:
                            process.send_signal(signal.SIGSTOP)
                            paused_at = current_playhead
                            paused = True
                            refresh_ui(True, current_playhead, audio_duration, "")
                        continue
                    if ch in (27, ord('q'), ord('Q')):
                        if paused:
                            process.send_signal(signal.SIGCONT)
                        process.terminate()
                        return
                    if ch == curses.KEY_UP:
                        if current_index > 0:
                            current_index -= 1
                            user_interrupted = True
                            if paused:
                                process.send_signal(signal.SIGCONT)
                            process.terminate()
                            break
                    if ch == curses.KEY_DOWN:
                        if current_index < len(history) - 1:
                            current_index += 1
                            user_interrupted = True
                            if paused:
                                process.send_signal(signal.SIGCONT)
                            process.terminate()
                            break

                if paused:
                    time.sleep(0.05)
                    continue

                if highlight_enabled and words_cache_path is not None and words_status != "ok":
                    payload = load_word_timestamps(words_cache_path, expected_model=highlight_model)
                    if payload is None:
                        words_status = "preparing"
                    elif payload.get("status") == "ok":
                        words_status = "ok"
                        words_error = None
                        words = payload.get("words") or None
                        debug_log_file(f"[CURSES] words_status changed to OK, words={len(words) if words else 0}")
                        current_word_index = -1
                        tokens = extract_text_tokens(cleaned_text)
                        word_to_token = align_words_to_text(words, tokens)
                    else:
                        words_status = "error"
                        words_error = payload.get("error")
                        debug_log_file(f"[CURSES] words_status changed to ERROR: {words_error}")

                if highlight_enabled:
                    playhead = max(0.0, playback_offset + (time.monotonic() - playback_started_at) - playback_lag)

                if words:
                    while (current_word_index + 1) < len(words) and playhead >= words[current_word_index + 1].get("start", 0.0):
                        current_word_index += 1
                    while current_word_index >= 0 and playhead < words[current_word_index].get("start", 0.0):
                        current_word_index -= 1

                new_context_span = None
                new_word_span = None
                new_dim_span = None
                if words and word_to_token and tokens and current_word_index >= 0:
                    spans = []
                    for wi in range(current_word_index - highlight_context_words, current_word_index + highlight_context_words + 1):
                        if 0 <= wi < len(word_to_token):
                            token_idx = word_to_token[wi]
                            if token_idx is not None and 0 <= token_idx < len(tokens):
                                t = tokens[token_idx]
                                spans.append((t["start"], t["end"]))
                    if spans:
                        new_context_span = (min(s for s, _ in spans), max(e for _, e in spans))
                    if 0 <= current_word_index < len(word_to_token):
                        token_idx = word_to_token[current_word_index]
                        if token_idx is not None and 0 <= token_idx < len(tokens):
                            t = tokens[token_idx]
                            new_word_span = (t["start"], t["end"])
                            if t["start"] > 0:
                                new_dim_span = (0, t["start"])

                if (new_context_span != current_highlight_span or 
                    new_word_span != current_highlight_word_span or 
                    new_dim_span != current_dim_span):
                    current_highlight_span = new_context_span
                    current_highlight_word_span = new_word_span
                    current_dim_span = new_dim_span
                    apply_highlights(current_dim_span, current_highlight_span, current_highlight_word_span)

                refresh_ui(paused, playhead, audio_duration, "")

                if not audio_queue.empty() and current_index == len(history) - 1:
                    try:
                        new_chunk = audio_queue.get_nowait()
                        history.append(new_chunk)
                        if highlight_enabled:
                            try:
                                schedule_word_timestamps(new_chunk[0], new_chunk[1])
                            except Exception:
                                pass
                        needs_full_render = True
                    except queue.Empty:
                        pass

                time.sleep(0.01)

            if not user_interrupted:
                is_at_last_chunk = (current_index == len(history) - 1 and audio_queue.empty() and generation_done)
                if is_at_last_chunk:
                    while True:
                        refresh_ui(False, playhead, audio_duration, "Reached end. ↑ to replay, Q to exit.")
                        ch = stdscr.getch()
                        if ch == curses.KEY_UP:
                            if current_index > 0:
                                current_index -= 1
                                needs_full_render = True
                                break
                            continue
                        if ch in (27, ord('q'), ord('Q')):
                            return
                        time.sleep(0.1)
                elif current_index < len(history) - 1:
                    current_index += 1
                elif not audio_queue.empty():
                    try:
                        new_chunk = audio_queue.get_nowait()
                        history.append(new_chunk)
                        if highlight_enabled:
                            try:
                                schedule_word_timestamps(new_chunk[0], new_chunk[1])
                            except Exception:
                                pass
                        current_index += 1
                        needs_full_render = True
                    except queue.Empty:
                        pass

    curses.wrapper(curses_main)
