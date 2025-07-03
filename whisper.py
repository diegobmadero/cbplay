#!/usr/bin/env python3

import hashlib
import pyperclip
import openai
from openai import OpenAI
import os
import subprocess
import signal
from pathlib import Path
import time
import threading
import queue
import textwrap
import sys
import select
import tty
import termios
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json

DEBUG = os.getenv('DEBUG') == '1'

def debug_print(*args, **kwargs):
    if DEBUG:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}]", *args, **kwargs)

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
    def __init__(self, voice="echo", response_format="aac", file_prefix="tts_clipboard"):
        self.timeout = 240
        self.voice = voice
        self.response_format = response_format
        self.file_prefix = file_prefix
        self.model = "tts-1"
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

    def _set_params(self, text):
        sanitized_text = text.strip()
        if not sanitized_text:
            debug_print("Text input is empty after sanitization. Skipping.")
            return None
        return {
            "voice": self.voice,
            "model": self.model,
            "response_format": self.response_format,
            "input": sanitized_text,
        }

    def _hash_text(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

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
            'text_preview': text[:50]
        }
        self._save_cache_index()
        
        debug_print(f"Generated audio file at: {out_file}")
        return out_file

def clean_text_for_display(text):
    """Clean text for better display by removing excessive symbols and formatting"""
    import re
    
    # First, preserve bullet point structure by ensuring each bullet is on its own line
    text = re.sub(r'([.!?])\s*•', r'\1\n    •', text)
    text = re.sub(r'•', '\n    •', text)
    
    # Remove box drawing characters (│, ─, └, ├, etc.)
    text = re.sub(r'[│├└─┌┐┘┤┬┴┼╭╮╯╰╱╲╳]', '', text)
    
    # Remove multiple asterisks but keep markdown bold (**)
    text = re.sub(r'\*{3,}', '', text)
    
    # Clean up markdown headers but keep the text and add spacing
    text = re.sub(r'^#{1,6}\s*', '\n', text, flags=re.MULTILINE)
    
    # Remove standalone pipes and clean up table formatting
    text = re.sub(r'\s*\|\s*\n', '\n', text)
    text = re.sub(r'\n\s*\|\s*', '\n', text)
    text = re.sub(r'\s*\|\s+(?=\n)', '', text)  # Remove trailing pipes
    
    # Clean up excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{4,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    
    # Clean lines
    lines = text.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        line = line.strip()
        # Skip lines that are just formatting characters
        if line and not all(c in '│├└─┌┐┘┤┬┴┼ ' for c in line):
            # Check if this is a bullet point line
            if line.startswith('•'):
                # Ensure proper formatting for bullet points
                cleaned_lines.append('    ' + line)
            else:
                cleaned_lines.append(line)
            prev_empty = False
        elif not line:
            # Keep single empty lines for paragraph breaks
            if not prev_empty:
                cleaned_lines.append('')
            prev_empty = True
    
    # Join and do final cleanup
    result = '\n'.join(cleaned_lines)
    
    # Clean up any remaining excessive newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()

def get_clipboard_content():
    content = pyperclip.paste()
    debug_print(f"Clipboard content: {content[:100]}...")
    return content

def split_text_intelligently(text, max_chars=1600):
    """Split text into chunks intelligently, preserving sentence boundaries"""
    chunks = []
    current_chunk = ""
    
    # Split by double newlines first (paragraphs)
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph is too long, split by sentences
        if len(para) > max_chars:
            # Try splitting by periods, question marks, and exclamation marks
            import re
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 2 <= max_chars:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
        else:
            # Try to fit whole paragraph
            if len(current_chunk) + len(para) + 4 <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    debug_print(f"Split into {len(chunks)} chunks")
    return chunks

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
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def play_audio_files_with_status(audio_queue, status_queue, tts):
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
        active_color = '\033[93m'  # Yellow
        prev_color = '\033[90m'    # Grey
        reset_color = '\033[0m'
        info_color = '\033[96m'    # Cyan
        success_color = '\033[92m'  # Green
        wrapper = textwrap.TextWrapper(width=100)

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
            
            # Check for status updates
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
                        total_chunks = status[2]  # Make sure we have the total
                        elapsed = time.time() - start_time
                        print(f"{success_color}Generation complete! Generated {status[1]} chunks in {elapsed:.1f}s{reset_color}")
                        debug_print(f"Generation done. Total chunks: {total_chunks}, History size: {len(history)}, Queue empty: {audio_queue.empty()}")
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
                print(f"{info_color}{queue_status} | Controls: ↑/↓ - Navigate | Q or ESC - Exit | Auto-advances when ready{reset_color}\n")
            else:
                print(f"{info_color}Controls: ↑/↓ - Navigate | Q or ESC - Exit | Auto-advances when ready{reset_color}\n")
            
            # Get next audio if available and we're at the end
            debug_print(f"Loop iteration: current_index={current_index}, history_len={len(history)}, queue_empty={audio_queue.empty()}")
            
            # First check if we should use existing history
            if 0 <= current_index < len(history):
                current_text_chunk = history[current_index]
                debug_print(f"Using existing chunk at index {current_index}")
            # Only get from queue if we need a new chunk
            elif not audio_queue.empty():
                # Get the first available chunk from queue
                try:
                    current_text_chunk = audio_queue.get_nowait()
                    history.append(current_text_chunk)
                    if current_index == -1:
                        current_index = 0
                    else:
                        current_index = len(history) - 1
                    debug_print(f"Retrieved chunk from queue, now at index {current_index}, history size: {len(history)}")
                except queue.Empty:
                    debug_print("Queue was empty when trying to get chunk")
                    pass
            else:
                if generation_done and len(history) == 0:
                    print("No audio files were generated.")
                    break
                elif current_index >= len(history) and generation_done:
                    print(f"{success_color}Playback complete!{reset_color}")
                    break
                else:
                    # Waiting for first chunk or next chunk
                    if current_index == -1:
                        debug_print("Waiting for first audio chunk to be generated...")
                    else:
                        debug_print(f"Waiting for next audio chunk...")
                    time.sleep(0.2)
                    continue

            # Show history context
            if current_index > 0:
                prev_file, prev_text = history[current_index - 1]
                # Clean and show entire previous chunk with proper wrapping
                cleaned_prev = clean_text_for_display(prev_text)
                wrapped_prev = '\n'.join(['\n'.join(['  ' + line for line in wrapper.wrap(line)]) if line else '' for line in cleaned_prev.split('\n')])
                print(f"{prev_color}Previous:\n{wrapped_prev}\n{reset_color}")

            if 0 <= current_index < len(history):
                audio_file, original_text_chunk = history[current_index]
                # Clean text for display
                cleaned_text = clean_text_for_display(original_text_chunk)
                wrapped_text = '\n'.join(['\n'.join(wrapper.wrap(line)) for line in cleaned_text.split('\n')])
                
                # Just show the current chunk text without "Playing X/Y"
                print(f"{active_color}Current:\n{wrapped_text}\n{reset_color}")

                # Play audio
                process = subprocess.Popen(['afplay', str(audio_file)])
                user_interrupted = False
                while process.poll() is None:
                    if is_data():
                        c = sys.stdin.read(1)
                        if c == '\x1b':  # x1b is ESC
                            next_char = sys.stdin.read(2)
                            if next_char == '[A':  # Up arrow
                                if current_index > 0:
                                    current_index = current_index - 1
                                    user_interrupted = True
                                    process.terminate()
                                    break
                            elif next_char == '[B':  # Down arrow
                                if current_index < len(history) - 1:
                                    current_index = current_index + 1
                                    user_interrupted = True
                                    process.terminate()
                                    break
                        elif c == 'q' or c == 'Q':  # Also allow 'q' to quit
                            process.terminate()
                            return
                    # Check for new audio while playing
                    if not audio_queue.empty() and current_index == len(history) - 1:
                        try:
                            new_chunk = audio_queue.get_nowait()
                            history.append(new_chunk)
                            debug_print(f"Pre-fetched next chunk while playing, history size: {len(history)}")
                        except queue.Empty:
                            pass
                
                # Only auto-advance if user didn't interrupt
                if not user_interrupted:
                    # Audio finished playing, auto-advance to next chunk
                    debug_print(f"Audio finished for chunk {current_index + 1}")
                    
                    # Check if we can advance
                    if current_index < len(history) - 1:
                        # We already have the next chunk in history
                        current_index += 1
                        debug_print(f"Auto-advancing to chunk {current_index + 1} (already in history)")
                    elif not audio_queue.empty():
                        # Try to get next chunk from queue
                        try:
                            new_chunk = audio_queue.get_nowait()
                            history.append(new_chunk)
                            current_index += 1
                            debug_print(f"Auto-advancing to chunk {current_index + 1} (fetched from queue)")
                        except queue.Empty:
                            debug_print("Queue was empty when trying to auto-advance")
                    else:
                        debug_print(f"No more chunks to play")
                else:
                    debug_print(f"User navigated to chunk {current_index + 1}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def graceful_exit(signal_received, frame):
    debug_print("Graceful exit initiated.")
    subprocess.call(['killall', 'afplay'])
    # Clean up temp files
    temp_dir = Path.home() / ".whisper" / "temp"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    exit(0)

def main():
    signal.signal(signal.SIGINT, graceful_exit)

    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    debug_print("Starting script...")
    
    print("Reading clipboard content...")
    clipboard_content = get_clipboard_content()
    
    if not clipboard_content.strip():
        print("Clipboard is empty!")
        return
    
    print(f"Processing {len(clipboard_content)} characters...")
    
    tts = TTSFile()
    combined_texts = split_text_intelligently(clipboard_content)
    audio_queue = queue.Queue()
    status_queue = queue.Queue()

    print(f"Split into {len(combined_texts)} chunks")
    
    # Start generation in background thread
    generation_thread = threading.Thread(
        target=generate_audio_files_streaming,
        args=(combined_texts, tts, audio_queue, status_queue),
        daemon=True
    )
    generation_thread.start()
    
    # Start playing immediately as audio becomes available
    play_audio_files_with_status(audio_queue, status_queue, tts)

if __name__ == "__main__":
    main()