# Simple Voice Bot

A fully offline voice assistant running on Raspberry Pi 5 (ARM64) with:
- **STT**: Vosk (streaming speech recognition)
- **LLM**: Ollama (lfm3, qwen2.5, gemma3, etc.)
- **TTS**: Piper (natural neural TTS)

## Requirements

- Raspberry Pi 5 (or similar ARM64 device with 4+ GB RAM)
- Python 3.11+
- Ollama installed and running
- Audio input/output devices (USB mic, speakers/headphones)

## Installation

```bash
# Clone and enter directory
cd simpleVoiceBot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (run once)
mkdir -p models
cd models

# Vosk STT model (small English, ~40MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip

# Piper TTS model (Lessac medium, ~63MB)
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (run in background)
ollama serve &

# Pull a model (lfm3 is already in your list)
ollama pull lfm3
# Or try others:
ollama pull gemma3:1b    # Faster, smaller
ollama pull qwen2.5:3b   # Good balance
```

## Configuration

Edit `config.yaml` to customize:
- Audio devices (input/output)
- STT model path
- LLM model, temperature, system prompt
- TTS voice, speed, quality
- Conversation timeouts

## Running

```bash
# Activate venv
source venv/bin/activate

# Run the bot
python main.py
```

Speak into your microphone. The bot will:
1. Listen until you stop speaking (2s silence timeout)
2. Transcribe with Vosk
3. Generate response with Ollama LLM
4. Speak response with Piper TTS

Say "exit", "quit", or "goodbye" to stop.

## Project Structure

```
simpleVoiceBot/
├── main.py              # Entry point
├── config.yaml          # Configuration
├── requirements.txt     # Python dependencies
├── .gitignore
├── models/              # Downloaded models (gitignored)
│   ├── vosk-model-small-en-us-0.15/
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
├── src/
│   ├── __init__.py
│   ├── config.py        # Configuration management
│   ├── stt.py           # Vosk speech-to-text
│   ├── llm.py           # Ollama LLM client
│   ├── tts.py           # Piper text-to-speech
│   └── bot.py           # Main conversation loop
└── venv/                # Virtual environment (gitignored)
```

## Troubleshooting

### No audio input
List devices: `python3 -c "import sounddevice as sd; print(sd.query_devices())"`
Set `audio.input_device` in config.yaml to the correct device index.

### Ollama connection failed
Make sure `ollama serve` is running. Check with `curl http://localhost:11434/api/tags`.

### High latency
- Try smaller LLM: `gemma3:1b` or `qwen2.5:1.5b`
- Reduce `max_tokens` in config.yaml
- Consider streaming TTS (future improvement)

### Model not found errors
Verify paths in `config.yaml` match downloaded model locations.

## Performance Notes (Pi 5, 8GB RAM)

| Component | Model | Est. Latency |
|-----------|-------|--------------|
| STT | Vosk small | ~100-300ms |
| LLM | lfm3 (1.1GB) | ~3-10s |
| LLM | gemma3:1b (815MB) | ~1-3s |
| TTS | Piper Lessac | ~0.5-2s |

Total end-to-end: **4-12 seconds** with lfm3, **2-5 seconds** with gemma3:1b.

## License

MIT