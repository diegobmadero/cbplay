# cbplay

Clipboard Play - Text-to-speech tool that converts clipboard content to audio using OpenAI's TTS API.

## Setup

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

2. Make the script executable:
```bash
chmod +x cbplay.py
```

3. Create a symlink for easy access:
```bash
ln -s /path/to/cbplay/cbplay.py ~/bin/cbplay
```

## Usage

1. Copy text to clipboard
2. Run `cbplay`
3. Listen as it reads the text in chunks

### Voice Options

Use `-v` or `--voice` to select a different voice:

```bash
cbplay -v alloy
cbplay --voice nova
```

Available voices: `alloy`, `ash` (default), `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`, `verse`

## Controls

- **Up/Down arrows**: Navigate between chunks
- **Q or ESC**: Exit
- **Auto-advance**: Automatically plays next chunk when current finishes

## Features

- Splits long text into manageable chunks
- Caches audio files for instant replay
- Concurrent generation with rate limiting
- Shows previous/current text while playing
- Preserves text formatting and structure

## Environment Variables

- `DEBUG=1`: Enable debug output
- `NOTTY=1`: Run in non-interactive mode (generation only)

## Cache

Audio files are cached in `~/.whisper/audio_cache/` for 7 days.