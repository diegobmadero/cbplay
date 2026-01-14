#!/usr/bin/env python3
"""cbplay - Clipboard Play: Text-to-speech for clipboard content."""

import argparse
import asyncio
import os
import pyperclip
import queue
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path

try:
    from openai import AsyncOpenAI
    from openai.helpers import LocalAudioPlayer
except Exception:
    AsyncOpenAI = None
    LocalAudioPlayer = None

from cbplay_tts import (
    create_tts_provider,
    get_available_voices,
    get_available_models,
    OPENAI_VOICES,
    GEMINI_VOICES,
    GEMINI_AVAILABLE,
)
from cbplay_stt import transcribe_audio_file
from cbplay_utils import (
    DEFAULT_STREAMING_INSTRUCTIONS,
    debug_print,
    debug_log_file,
    set_debug_file,
    clean_text_for_display,
    prepare_text_for_tts,
    split_text_intelligently,
    is_audio_file,
    resolve_audio_path,
)
from cbplay_player import generate_audio_files_streaming


def get_clipboard_content() -> str:
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


async def stream_tts_chunks(combined_texts, voice, model, instructions, response_format="pcm", timeout=240):
    if not AsyncOpenAI or not LocalAudioPlayer:
        raise ImportError("AsyncOpenAI/LocalAudioPlayer not available in installed openai package.")
    
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
        
        header = f"Chunk {index}/{total} ({len(prepared)} chars)"
        print(f"\n{header}")
        print("-" * len(header))
        print(prepared)
        print()
        print("Streaming and playing...")
        
        if instructions:
            async with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,  # type: ignore
                input=prepared,
                response_format="pcm",
                instructions=instructions,
            ) as response:
                await player.play(response)
        else:
            async with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,  # type: ignore
                input=prepared,
                response_format="pcm",
            ) as response:
                await player.play(response)


def graceful_exit(signal_received, frame):
    debug_print("Graceful exit initiated.")
    debug_log_file(f"graceful_exit signal={signal_received}")
    subprocess.call(['killall', 'afplay'], stderr=subprocess.DEVNULL)
    subprocess.call(['killall', 'ffplay'], stderr=subprocess.DEVNULL)
    temp_dir = Path.home() / ".whisper" / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    sys.exit(0)


def run_interactive_tts_flow(clipboard_content, tts_texts, display_texts, args, model, provider):
    print(f"Processing {len(clipboard_content)} characters...")

    response_format = "mp3" if args.highlight else "aac"
    if provider == "gemini":
        response_format = "wav"

    tts = create_tts_provider(
        provider=provider,
        voice=args.voice,
        model=model,
        response_format=response_format,
        instructions=args.instructions,
        refresh_cache=args.refresh_cache,
    )

    audio_queue = queue.Queue()
    status_queue = queue.Queue()

    print(f"Split into {len(tts_texts)} chunks")

    generation_thread = threading.Thread(
        target=generate_audio_files_streaming,
        args=(tts_texts, tts, audio_queue, status_queue),
        daemon=True
    )
    generation_thread.start()

    # Track which chunks should skip highlighting (structural differences like tables/diagrams)
    # Only skip for major structural differences, not minor formatting like bold markers
    skip_highlight_chunks = set()
    for i, (tts_chunk, disp_chunk) in enumerate(zip(tts_texts, display_texts)):
        # Skip highlighting if chunk has diagram placeholder or table conversion
        if '[Diagram omitted]' in tts_chunk or '[Table omitted]' in tts_chunk:
            skip_highlight_chunks.add(i)
            debug_log_file(f"[MAIN] skip chunk {i}: has diagram/table omitted placeholder")
        # Skip if table was converted (look for "Header: Value." pattern after "|" in display)
        elif '|' in disp_chunk and ': ' in tts_chunk and '|' not in tts_chunk:
            skip_highlight_chunks.add(i)
            debug_log_file(f"[MAIN] skip chunk {i}: table converted to prose")
    debug_log_file(f"[MAIN] total chunks={len(tts_texts)}, skip_highlight_chunks={skip_highlight_chunks}")

    from cbplay_player_ui import play_audio_files_with_status
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
        ui_mode=args.ui,
        all_text_chunks=display_texts,
        skip_highlight_chunks=skip_highlight_chunks,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='cbplay - Clipboard Play: auto-detect clipboard/file and transcribe or speak')
    
    parser.add_argument('--mode', choices=['auto', 'tts', 'stt'], default='auto',
                        help='Mode: auto (stt for audio, tts otherwise), tts, stt')
    
    parser.add_argument('--provider', choices=['openai', 'gemini'], default='gemini',
                        help='TTS provider: openai or gemini (default: gemini)')
    
    openai_voices_str = ', '.join(OPENAI_VOICES)
    gemini_voices_str = ', '.join(GEMINI_VOICES[:5]) + '...'
    parser.add_argument('-v', '--voice', default=None,
                        help=f'Voice name. OpenAI: {openai_voices_str}. Gemini: {gemini_voices_str}')
    
    parser.add_argument('-a', '--audio-file',
                        help='Audio file to transcribe in stt mode')
    
    parser.add_argument('--transcription-model', default='gpt-4o-mini-transcribe-2025-12-15',
                        help='Audio transcription model (default: gpt-4o-mini-transcribe-2025-12-15)')
    
    parser.add_argument('--model', default=None,
                        help='TTS model (default: provider-specific)')
    
    parser.add_argument('--highlight-tts-model', default=None,
                        help='TTS model when --highlight enabled')
    
    parser.add_argument('--instructions', default=DEFAULT_STREAMING_INSTRUCTIONS,
                        help='Delivery instructions for TTS (gpt-4o models only)')
    
    parser.add_argument('--chunk-size', type=int, default=300,
                        help='Max chars per chunk (default: 300)')
    
    parser.add_argument('--stream', action='store_true',
                        help='Use async streaming playback (disables interactive UI)')
    
    parser.add_argument('--ui', choices=['ansi', 'curses'], default='curses',
                        help='Interactive UI mode (default: curses)')
    
    highlight_group = parser.add_mutually_exclusive_group()
    highlight_group.add_argument('--highlight', dest='highlight', action='store_true',
                                  help='Highlight currently spoken word (default: on)')
    highlight_group.add_argument('--no-highlight', dest='highlight', action='store_false',
                                  help='Disable word-level highlighting')
    parser.set_defaults(highlight=True)
    
    parser.add_argument('--highlight-model', default='whisper-1',
                        help='Model for word timestamps (default: whisper-1)')
    
    parser.add_argument('--highlight-window', type=int, default=1,
                        help='Words of context around highlight (default: 1)')
    
    parser.add_argument('--resume-rewind', type=float, default=2.0,
                        help='Seconds to rewind on resume (default: 2.0)')
    
    parser.add_argument('--playhead-lag', type=float, default=None,
                        help='Playhead sync offset in seconds')
    
    parser.add_argument('--esc-timeout', type=float, default=None,
                        help='ESC sequence timeout in seconds')
    
    parser.add_argument('--debug', action='store_true',
                        help='Show debug info in UI')
    
    parser.add_argument('--debug-file', default=None,
                        help='Debug log file path')
    
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Regenerate all audio files, ignoring cache')
    
    parser.add_argument('--list-voices', action='store_true',
                        help='List available voices for the selected provider')

    parser.add_argument('--full-audio', metavar='FILE',
                        help='Export all audio to a single file (mp3/wav). Reuses cached audio.')

    return parser


def export_full_audio(combined_texts, args, model, provider, output_path: str):
    """Generate all audio chunks and concatenate into a single output file."""
    output_file = Path(output_path)
    output_ext = output_file.suffix.lower().lstrip('.')

    if output_ext not in ('mp3', 'wav', 'm4a', 'aac'):
        print(f"Unsupported output format: {output_ext}. Use .mp3, .wav, .m4a, or .aac")
        return False

    response_format = "wav" if provider == "gemini" else "mp3"

    tts = create_tts_provider(
        provider=provider,
        voice=args.voice,
        model=model,
        response_format=response_format,
        instructions=args.instructions,
        refresh_cache=args.refresh_cache,
    )

    temp_dir = Path.home() / ".whisper" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total = len(combined_texts)
    audio_files = []

    cached_set = set()
    for i, text in enumerate(combined_texts):
        text_hash = tts._hash_text(text)
        cached_file = tts.cache_dir / f"{text_hash}.{tts.response_format}"
        if text_hash in tts.cache_index and cached_file.exists():
            cached_set.add(i)

    print(f"Exporting {total} chunks to {output_path} ({len(cached_set)} cached)")

    for i, text in enumerate(combined_texts):
        text_hash = tts._hash_text(text)
        unique_filename = f"tts_export_{text_hash}.{tts.response_format}"
        out_file = temp_dir / unique_filename

        result = tts.to_file(text, out_file)
        if result is None:
            print(f"Failed to generate audio for chunk {i + 1}")
            continue

        audio_files.append(str(out_file))
        status = "cached" if i in cached_set else "generated"
        print(f"  [{i + 1}/{total}] ({status})")

    if not audio_files:
        print("No audio files generated.")
        return False

    list_file = temp_dir / "concat_list.txt"
    with open(list_file, 'w') as f:
        for af in audio_files:
            f.write(f"file '{af}'\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file),
    ]

    if output_ext == 'mp3':
        ffmpeg_cmd.extend(['-c:a', 'libmp3lame', '-q:a', '2'])
    elif output_ext == 'wav':
        ffmpeg_cmd.extend(['-c:a', 'pcm_s16le'])
    elif output_ext in ('m4a', 'aac'):
        ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

    ffmpeg_cmd.append(str(output_file))

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False

    list_file.unlink(missing_ok=True)
    for af in audio_files:
        Path(af).unlink(missing_ok=True)

    print(f"Audio exported to: {output_file}")
    return True


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.list_voices:
        print(f"Available voices for {args.provider}:")
        for voice in get_available_voices(args.provider):
            print(f"  {voice}")
        return
    
    if args.provider == "gemini" and not GEMINI_AVAILABLE:
        print("Error: Gemini provider requires google-genai package.")
        print("Install with: pip install google-genai")
        sys.exit(1)
    
    if args.voice is None:
        args.voice = "verse" if args.provider == "openai" else "Kore"
    
    if args.model is None:
        args.model = "gpt-4o-mini-tts-2025-12-15" if args.provider == "openai" else "gemini-2.5-flash-preview-tts"
    
    if args.highlight and not args.highlight_tts_model:
        args.highlight_tts_model = args.model
    
    if args.highlight and args.highlight_model != "whisper-1":
        print("Note: word timestamps require whisper-1; overriding --highlight-model")
        args.highlight_model = "whisper-1"
    
    if args.debug_file or args.debug:
        path = args.debug_file if args.debug_file else "debug_cbplay.log"
        set_debug_file(path)
    debug_log_file(f"args: {args}")
    
    if args.stream and args.highlight:
        if "--highlight" in sys.argv:
            print("--highlight is only supported in interactive UI. Remove --stream.")
            return
        print("Note: --highlight disabled because --stream uses non-interactive playback.")
        args.highlight = False
    
    signal.signal(signal.SIGINT, graceful_exit)
    
    api_key_var = "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    if not os.getenv(api_key_var):
        print(f"Error: {api_key_var} is not set in environment variables.")
        sys.exit(1)
    
    debug_print("Starting script...")
    debug_print(f"Using provider: {args.provider}, voice: {args.voice}")
    
    print("Reading clipboard content...")
    clipboard_content = get_clipboard_content()
    
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
            print("No valid audio file found. Provide --audio-file or copy path to clipboard.")
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
    
    if not clipboard_content.strip():
        print("Clipboard is empty!")
        return
    
    try:
        chunk_size = int(args.chunk_size)
    except Exception:
        chunk_size = 300
    if chunk_size <= 0:
        chunk_size = 300
    
    # Split text and create parallel lists: raw for display, processed for TTS
    raw_chunks = split_text_intelligently(clipboard_content, max_chars=chunk_size)
    paired = [(raw, prepare_text_for_tts(raw)) for raw in raw_chunks]
    # Filter out chunks that become empty after processing
    paired = [(raw, tts) for raw, tts in paired if tts]
    # Strip backticks from display text for alignment with TTS (which doesn't say backticks)
    from cbplay_utils import strip_backticks
    display_texts = [strip_backticks(raw) for raw, tts in paired]
    tts_texts = [tts for raw, tts in paired]

    if args.full_audio:
        tts_model = args.model
        export_full_audio(tts_texts, args, tts_model, args.provider, args.full_audio)
        return

    if not args.stream:
        tts_model = args.highlight_tts_model if args.highlight else args.model
        run_interactive_tts_flow(clipboard_content, tts_texts, display_texts, args, tts_model, args.provider)
        return

    if args.provider == "gemini":
        print("Streaming mode not yet supported for Gemini. Using interactive mode.")
        tts_model = args.highlight_tts_model if args.highlight else args.model
        run_interactive_tts_flow(clipboard_content, tts_texts, display_texts, args, tts_model, args.provider)
        return

    try:
        print(f"Streaming {len(tts_texts)} chunk(s) with {args.model}/{args.voice}...")
        asyncio.run(stream_tts_chunks(
            combined_texts=tts_texts,
            voice=args.voice,
            model=args.model,
            instructions=args.instructions,
            response_format="pcm",
            timeout=240,
        ))
    except Exception as e:
        print(f"Streaming path failed ({e}). Falling back to interactive UI...")
        run_interactive_tts_flow(clipboard_content, tts_texts, display_texts, args, args.model, args.provider)


if __name__ == "__main__":
    main()
