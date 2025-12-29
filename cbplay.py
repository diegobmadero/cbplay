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


def run_interactive_tts_flow(clipboard_content, combined_texts, args, model, provider):
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
    )
    
    audio_queue = queue.Queue()
    status_queue = queue.Queue()

    print(f"Split into {len(combined_texts)} chunks")
    
    generation_thread = threading.Thread(
        target=generate_audio_files_streaming,
        args=(combined_texts, tts, audio_queue, status_queue),
        daemon=True
    )
    generation_thread.start()
    
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
        all_text_chunks=combined_texts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='cbplay - Clipboard Play: auto-detect clipboard/file and transcribe or speak')
    
    parser.add_argument('--mode', choices=['auto', 'tts', 'stt'], default='auto',
                        help='Mode: auto (stt for audio, tts otherwise), tts, stt')
    
    parser.add_argument('--provider', choices=['openai', 'gemini'], default='openai',
                        help='TTS provider: openai or gemini (default: openai)')
    
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
    
    parser.add_argument('--chunk-size', type=int, default=600,
                        help='Max chars per chunk (default: 600)')
    
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
    
    parser.add_argument('--list-voices', action='store_true',
                        help='List available voices for the selected provider')
    
    return parser


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
        chunk_size = 600
    if chunk_size <= 0:
        chunk_size = 600
    
    combined_texts = [prepare_text_for_tts(t) for t in split_text_intelligently(clipboard_content, max_chars=chunk_size)]
    combined_texts = [t for t in combined_texts if t]
    
    if not args.stream:
        tts_model = args.highlight_tts_model if args.highlight else args.model
        run_interactive_tts_flow(clipboard_content, combined_texts, args, tts_model, args.provider)
        return
    
    if args.provider == "gemini":
        print("Streaming mode not yet supported for Gemini. Using interactive mode.")
        tts_model = args.highlight_tts_model if args.highlight else args.model
        run_interactive_tts_flow(clipboard_content, combined_texts, args, tts_model, args.provider)
        return
    
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
        run_interactive_tts_flow(clipboard_content, combined_texts, args, args.model, args.provider)


if __name__ == "__main__":
    main()
