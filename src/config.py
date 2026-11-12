"""Configuration management for the voice bot."""

import yaml
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Any


def load_yaml_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# Load YAML once at module import
_yaml_config = load_yaml_config()


def get_yaml_section(section: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Get a section from YAML config, falling back to defaults."""
    return {**defaults, **_yaml_config.get(section, {})}


class AudioConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_", extra="ignore")
    
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    input_device: int | None = None
    output_device: int | None = None


class STTConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_", protected_namespaces=("settings_",), extra="ignore")
    
    model_path: str = "models/vosk-model-small-en-us-0.15"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")
    
    provider: str = "ollama"
    model: str = "lfm3:latest"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 200
    system_prompt: str = (
        "You are a helpful voice assistant. Keep responses concise and natural for speech. "
        "Avoid markdown formatting. Use short sentences."
    )


class TTSConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", protected_namespaces=("settings_",), extra="ignore")
    
    model_path: str = "models/en_US-lessac-medium.onnx"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8


class ConversationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_", extra="ignore")
    
    wake_word: str | None = None
    silence_timeout: float = 2.0
    min_audio_length: float = 0.5


class Config:
    """Main configuration container - loads from YAML then env vars."""
    
    def __init__(self):
        # Audio
        audio_defaults = {"sample_rate": 16000, "channels": 1, "chunk_size": 1024}
        self.audio = AudioConfig(**get_yaml_section("audio", audio_defaults))
        
        # STT
        stt_defaults = {"model_path": "models/vosk-model-small-en-us-0.15"}
        self.stt = STTConfig(**get_yaml_section("stt", stt_defaults))
        
        # LLM
        llm_defaults = {
            "provider": "ollama",
            "model": "lfm3:latest",
            "host": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 200,
            "system_prompt": (
                "You are a helpful voice assistant. Keep responses concise and natural for speech. "
                "Avoid markdown formatting. Use short sentences."
            ),
        }
        self.llm = LLMConfig(**get_yaml_section("llm", llm_defaults))
        
        # TTS
        tts_defaults = {
            "model_path": "models/en_US-lessac-medium.onnx",
            "speaker_id": 0,
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_w_scale": 0.8,
        }
        self.tts = TTSConfig(**get_yaml_section("tts", tts_defaults))
        
        # Conversation
        conv_defaults = {"wake_word": None, "silence_timeout": 2.0, "min_audio_length": 0.5}
        self.conversation = ConversationConfig(**get_yaml_section("conversation", conv_defaults))


config = Config()