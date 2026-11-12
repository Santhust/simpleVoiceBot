"""Configuration management for the voice bot."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AudioConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUDIO_")
    
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    input_device: int | None = None
    output_device: int | None = None


class STTConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_", protected_namespaces=("settings_",))
    
    model_path: str = "models/vosk-model-small-en-us-0.15"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
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
    model_config = SettingsConfigDict(env_prefix="TTS_", protected_namespaces=("settings_",))
    
    model_path: str = "models/en_US-lessac-medium.onnx"
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8


class ConversationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_")
    
    wake_word: str | None = None
    silence_timeout: float = 2.0
    min_audio_length: float = 0.5


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)


config = Config()