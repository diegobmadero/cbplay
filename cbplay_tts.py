"""TTS Provider abstraction and implementations for cbplay."""

from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
import base64
import hashlib
import json
import os
import shutil
import struct
import threading
import time
from typing import Optional, List, Literal

import openai
from openai import OpenAI

from cbplay_utils import debug_log_file

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


AudioFormat = Literal['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm']


class RateLimiter:
    def __init__(self, requests_per_minute: int = 50):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[datetime] = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = datetime.now()
            self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
            
            if len(self.request_times) >= self.requests_per_minute:
                oldest_request = self.request_times[0]
                wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time)
                    return self.wait_if_needed()
            
            self.request_times.append(now)


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    def __init__(
        self,
        voice: str,
        model: str,
        response_format: str = "mp3",
        instructions: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        refresh_cache: bool = False,
    ):
        self.voice = voice
        self.model = model
        self.response_format = response_format
        self.instructions = instructions
        self.refresh_cache = refresh_cache
        self.cache_dir = cache_dir or (Path.home() / ".whisper" / "audio_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        self._clean_cache()
    
    def _load_cache_index(self) -> dict:
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache_index(self):
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _clean_cache(self, max_age_days: int = 7):
        now = time.time()
        to_remove = []
        
        for hash_key, data in self.cache_index.items():
            if now - data.get('timestamp', 0) > max_age_days * 86400:
                cached_file = self.cache_dir / f"{hash_key}.{self.response_format}"
                if cached_file.exists():
                    cached_file.unlink()
                to_remove.append(hash_key)
        
        for key in to_remove:
            del self.cache_index[key]
        
        if to_remove:
            self._save_cache_index()
    
    def _hash_text(self, text: str) -> str:
        content = "\n".join([
            f"provider:{self.provider_name}",
            f"model:{self.model}",
            f"voice:{self.voice}",
            f"format:{self.response_format}",
            f"instructions:{self.instructions or ''}",
            f"text:{text}",
        ])
        return hashlib.sha256(content.encode()).hexdigest()
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
    
    @abstractmethod
    def _generate_audio(self, text: str, out_file: Path) -> Optional[Path]:
        pass
    
    def to_file(self, text: str, out_file: Path) -> Optional[Path]:
        text_hash = self._hash_text(text)
        cached_file = self.cache_dir / f"{text_hash}.{self.response_format}"
        
        if out_file.exists():
            out_file.unlink()
        
        if not self.refresh_cache and text_hash in self.cache_index and cached_file.exists():
            shutil.copy2(cached_file, out_file)
            return out_file
        
        result = self._generate_audio(text, out_file)
        
        if result is not None and out_file.exists():
            shutil.copy2(out_file, cached_file)
            self.cache_index[text_hash] = {
                'timestamp': time.time(),
                'text_preview': text[:50],
                'voice': self.voice,
                'provider': self.provider_name,
            }
            self._save_cache_index()
        
        return result


OPENAI_VOICES = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 
                 'nova', 'onyx', 'sage', 'shimmer', 'verse']

OPENAI_MODELS = ['tts-1', 'tts-1-hd', 'gpt-4o-mini-tts-2025-12-15', 'gpt-4o-audio-preview']


class OpenAITTSProvider(TTSProvider):
    def __init__(
        self,
        voice: str = "verse",
        model: str = "gpt-4o-mini-tts-2025-12-15",
        response_format: str = "mp3",
        instructions: Optional[str] = None,
        timeout: int = 240,
        refresh_cache: bool = False,
        **kwargs,
    ):
        super().__init__(voice, model, response_format, instructions, refresh_cache=refresh_cache, **kwargs)
        self.timeout = timeout
        self.client = OpenAI(timeout=self.timeout)
        self.rate_limiter = RateLimiter(requests_per_minute=50)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _effective_instructions(self) -> Optional[str]:
        instructions = (self.instructions or "").strip()
        if not instructions:
            return None
        if self.model.startswith("gpt-4o"):
            return instructions
        return None
    
    def _generate_audio(self, text: str, out_file: Path) -> Optional[Path]:
        raw_text = "" if text is None else str(text)
        if not raw_text.strip():
            return None
        
        debug_log_file(f"[OpenAI TTS] Starting generation for text: {raw_text[:100]}...")
        debug_log_file(f"[OpenAI TTS] Model: {self.model}, Voice: {self.voice}")
        
        audio_format: AudioFormat = "mp3"
        if self.response_format in ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'):
            audio_format = self.response_format  # type: ignore
        
        instructions = self._effective_instructions()
        debug_log_file(f"[OpenAI TTS] Format: {audio_format}, Instructions: {bool(instructions)}")
        
        self.rate_limiter.wait_if_needed()
        
        retry_count = 0
        max_retries = 3
        response = None
        
        while retry_count < max_retries:
            try:
                debug_log_file(f"[OpenAI TTS] API call attempt {retry_count + 1}/{max_retries}")
                if instructions:
                    response = self.client.audio.speech.create(
                        voice=self.voice,  # type: ignore
                        model=self.model,
                        response_format=audio_format,
                        input=raw_text,
                        instructions=instructions,
                    )
                else:
                    response = self.client.audio.speech.create(
                        voice=self.voice,  # type: ignore
                        model=self.model,
                        response_format=audio_format,
                        input=raw_text,
                    )
                debug_log_file(f"[OpenAI TTS] API call successful")
                break
            except openai.APITimeoutError as e:
                retry_count += 1
                debug_log_file(f"[OpenAI TTS] Timeout error: {e}")
                wait_time = min(2 ** retry_count, 30)
                time.sleep(wait_time)
            except openai.RateLimitError as e:
                retry_count += 1
                debug_log_file(f"[OpenAI TTS] Rate limit error: {e}")
                wait_time = min(2 ** retry_count, 30)
                time.sleep(wait_time)
            except openai.APIConnectionError as e:
                retry_count += 1
                debug_log_file(f"[OpenAI TTS] Connection error: {e}")
                wait_time = min(2 ** retry_count, 30)
                time.sleep(wait_time)
            except openai.BadRequestError as e:
                debug_log_file(f"[OpenAI TTS] Bad request error: {e}")
                return None
            except Exception as e:
                debug_log_file(f"[OpenAI TTS] Unexpected error: {e}")
                return None
        
        if response is None:
            debug_log_file(f"[OpenAI TTS] Failed after {max_retries} attempts")
            return None
        
        with open(out_file, "wb") as file:
            file.write(response.content)
        
        debug_log_file(f"[OpenAI TTS] Audio written to {out_file} ({len(response.content)} bytes)")
        return out_file


GEMINI_VOICES = [
    'Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir',
    'Aoede', 'Leda', 'Orus', 'Achernar', 'Achird',
    'Algenib', 'Algieba', 'Alnilam', 'Autonoe', 'Callirrhoe',
    'Despina', 'Enceladus', 'Erinome', 'Gacrux', 'Iapetus',
    'Laomedeia', 'Pulcherrima', 'Rasalgethi', 'Sadachbia',
    'Sadaltager', 'Schedar', 'Sulafat', 'Umbriel',
    'Vindemiatrix', 'Zubenelgenubi',
]

GEMINI_MODELS = ['gemini-2.5-flash-preview-tts', 'gemini-2.5-pro-preview-tts']


class GeminiTTSProvider(TTSProvider):
    def __init__(
        self,
        voice: str = "Kore",
        model: str = "gemini-2.5-flash-preview-tts",
        response_format: str = "mp3",
        instructions: Optional[str] = None,
        api_key: Optional[str] = None,
        refresh_cache: bool = False,
        timeout: int = 60,
        **kwargs,
    ):
        if not GEMINI_AVAILABLE or genai is None:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        super().__init__(voice, model, "wav", instructions, refresh_cache=refresh_cache, **kwargs)
        self.requested_format = response_format
        self.timeout = timeout

        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(
            api_key=self.api_key,
            http_options={'timeout': self.timeout * 1000},  # timeout in ms
        )
        self.rate_limiter = RateLimiter(requests_per_minute=60)
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    def _generate_audio(self, text: str, out_file: Path) -> Optional[Path]:
        raw_text = "" if text is None else str(text)
        if not raw_text.strip():
            return None
        
        debug_log_file(f"[Gemini TTS] Starting generation for text: {raw_text[:100]}...")
        debug_log_file(f"[Gemini TTS] Model: {self.model}, Voice: {self.voice}")
        
        self.rate_limiter.wait_if_needed()
        
        retry_count = 0
        max_retries = 3
        response = None
        
        while retry_count < max_retries:
            try:
                debug_log_file(f"[Gemini TTS] API call attempt {retry_count + 1}/{max_retries}")
                if self.instructions:
                    tts_prompt = f"{self.instructions}\n\nRead aloud the following text:\n\n{raw_text}"
                else:
                    tts_prompt = f"Read aloud the following text:\n\n{raw_text}"
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=tts_prompt,
                    config={
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {
                                    "voice_name": self.voice
                                }
                            }
                        }
                    },
                )
                debug_log_file(f"[Gemini TTS] API call successful")
                break
            except Exception as e:
                retry_count += 1
                debug_log_file(f"[Gemini TTS] API error: {e}")
                if retry_count >= max_retries:
                    debug_log_file(f"[Gemini TTS] Failed after {max_retries} attempts: {e}")
                    return None
                wait_time = min(2 ** retry_count, 30)
                debug_log_file(f"[Gemini TTS] Retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
        
        if response is None:
            return None
        
        try:
            audio_data = None
            mime_type = None
            if hasattr(response, 'candidates') and response.candidates:
                debug_log_file(f"[Gemini TTS] Response has {len(response.candidates)} candidates")
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            debug_log_file(f"[Gemini TTS] Content has {len(candidate.content.parts)} parts")
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    audio_data = part.inline_data.data
                                    mime_type = getattr(part.inline_data, 'mime_type', 'unknown')
                                    debug_log_file(f"[Gemini TTS] Found inline_data, mime_type: {mime_type}")
                                    debug_log_file(f"[Gemini TTS] Data type: {type(audio_data).__name__}, len: {len(audio_data) if audio_data else 0}")
                                    break
                    if audio_data:
                        break
            
            if not audio_data:
                debug_log_file("[Gemini TTS] No audio data found in response")
                return None
            
            # Gemini ALWAYS returns base64-encoded PCM data (as str or bytes)
            # Even when type is bytes, it contains ASCII base64 characters, not raw PCM
            debug_log_file(f"[Gemini TTS] Decoding base64 data (type={type(audio_data).__name__}, len={len(audio_data)})")
            debug_log_file(f"[Gemini TTS] First 50 bytes: {audio_data[:50] if audio_data else 'None'}")
            pcm_data = base64.b64decode(audio_data)
            
            debug_log_file(f"[Gemini TTS] PCM data size: {len(pcm_data)} bytes")
            self._write_wav(pcm_data, out_file, sample_rate=24000)
            debug_log_file(f"[Gemini TTS] WAV written to {out_file}")
            return out_file
            
        except Exception as e:
            debug_log_file(f"[Gemini TTS] Error extracting audio: {e}")
            return None
    
    def _write_wav(self, pcm_data: bytes, out_file: Path, sample_rate: int = 24000):
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)
        
        with open(out_file, 'wb') as f:
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))
            f.write(struct.pack('<H', 1))
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(pcm_data)


def create_tts_provider(
    provider: str = "openai",
    voice: Optional[str] = None,
    model: Optional[str] = None,
    response_format: str = "mp3",
    instructions: Optional[str] = None,
    refresh_cache: bool = False,
    **kwargs,
) -> TTSProvider:
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAITTSProvider(
            voice=voice or "verse",
            model=model or "gpt-4o-mini-tts-2025-12-15",
            response_format=response_format,
            instructions=instructions,
            refresh_cache=refresh_cache,
            **kwargs,
        )
    elif provider == "gemini":
        return GeminiTTSProvider(
            voice=voice or "Kore",
            model=model or "gemini-2.5-flash-preview-tts",
            response_format=response_format,
            instructions=instructions,
            refresh_cache=refresh_cache,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider}. Supported: openai, gemini")


def get_available_voices(provider: str = "openai") -> List[str]:
    provider = provider.lower()
    if provider == "openai":
        return OPENAI_VOICES.copy()
    elif provider == "gemini":
        return GEMINI_VOICES.copy()
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_available_models(provider: str = "openai") -> List[str]:
    provider = provider.lower()
    if provider == "openai":
        return OPENAI_MODELS.copy()
    elif provider == "gemini":
        return GEMINI_MODELS.copy()
    else:
        raise ValueError(f"Unknown provider: {provider}")
