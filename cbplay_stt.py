"""Speech-to-text (transcription) functionality for cbplay."""

from pathlib import Path
import json
import time
from typing import Optional, List, Dict, Any

import openai
from openai import OpenAI

from cbplay_utils import debug_print, debug_log_file


def transcribe_audio_file(audio_path, model: str = "gpt-4o-transcribe") -> Optional[str]:
    debug_log_file(f"transcribe_audio_file start path={audio_path} model={model}")
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
                debug_log_file("transcribe_audio_file: response missing text")
                return None
            debug_log_file(f"transcribe_audio_file success len={len(text)}")
            return text
    except FileNotFoundError:
        print(f"Audio file not found: {audio_path}")
        debug_log_file(f"transcribe_audio_file FileNotFoundError: {audio_path}")
        return None
    except openai.BadRequestError as e:
        print(f"Failed to transcribe audio due to bad request: {e}")
        debug_log_file(f"transcribe_audio_file BadRequestError: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"Rate limit error during transcription: {e}")
        debug_log_file(f"transcribe_audio_file RateLimitError: {e}")
        return None
    except openai.APITimeoutError as e:
        print(f"Timeout error during transcription: {e}")
        debug_log_file(f"transcribe_audio_file APITimeoutError: {type(e).__name__} {e}")
        return None
    except openai.APIConnectionError as e:
        print(f"Connection error during transcription: {e}")
        debug_log_file(f"transcribe_audio_file APIConnectionError: {type(e).__name__} {e}")
        return None
    except Exception as e:
        print(f"Failed to transcribe audio due to error: {e}")
        debug_log_file(f"transcribe_audio_file Exception: {type(e).__name__} {e}")
        return None


def transcribe_audio_words(audio_path: Path, model: str = "gpt-4o-transcribe", timeout: int = 300) -> Optional[List[Dict[str, Any]]]:
    debug_log_file(f"transcribe_audio_words start path={audio_path} model={model}")
    client = OpenAI(timeout=timeout)
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
    except openai.APITimeoutError as e:
        debug_print(f"Word-timestamp transcription timeout: {e}")
        debug_log_file(f"transcribe_audio_words APITimeoutError: {type(e).__name__} {e}")
        return None
    except openai.APIConnectionError as e:
        debug_print(f"Word-timestamp transcription connection error: {e}")
        debug_log_file(f"transcribe_audio_words APIConnectionError: {type(e).__name__} {e}")
        return None
    except Exception as e:
        debug_print(f"Word-timestamp transcription failed: {e}")
        debug_log_file(f"transcribe_audio_words Exception: {type(e).__name__} {e}")
        return None

    debug_log_file(f"transcribe_audio_words API call complete, parsing response...")
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
        debug_log_file(f"transcribe_audio_words: no words found in response")
        return None

    debug_log_file(f"transcribe_audio_words: found {len(words)} raw words")
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
    debug_log_file(f"transcribe_audio_words: returning {len(normalized) if normalized else 0} normalized words")
    return normalized if normalized else None


def load_word_timestamps(cache_path: Path, expected_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
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


def save_word_timestamps(cache_path: Path, transcription_model: str, words: List, status: str = "ok", error: Optional[str] = None):
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
