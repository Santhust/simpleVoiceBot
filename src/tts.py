"""Text-to-Speech using Piper."""

import wave
import tempfile
import os
import numpy as np
from piper import PiperVoice, SynthesisConfig
import sounddevice as sd
import soundfile as sf

from src.config import config


class TTS:
    def __init__(
        self,
        model_path: str | None = None,
        speaker_id: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
    ):
        self.model_path = model_path or config.tts.model_path
        self.speaker_id = speaker_id if speaker_id is not None else config.tts.speaker_id
        self.length_scale = length_scale if length_scale is not None else config.tts.length_scale
        self.noise_scale = noise_scale if noise_scale is not None else config.tts.noise_scale
        self.noise_w_scale = noise_w_scale if noise_w_scale is not None else config.tts.noise_w_scale
        self._voice = None
        self._syn_config = None

    def load(self):
        """Load the Piper voice model."""
        print(f"Loading Piper voice from {self.model_path}...")
        self._voice = PiperVoice.load(self.model_path)
        self._syn_config = SynthesisConfig(
            speaker_id=self.speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )
        print("Piper voice loaded.")

    def synthesize(self, text: str, output_path: str | None = None) -> str | None:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save WAV file
            
        Returns:
            Path to generated WAV file, or None on error
        """
        if self._voice is None:
            self.load()

        if output_path is None:
            # Create temp file
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        try:
            with wave.open(output_path, "wb") as wav_file:
                # Set WAV parameters before synthesis
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self._voice.config.sample_rate)
                
                # Piper 1.8.0 returns iterable of AudioChunk
                for chunk in self._voice.synthesize(text, self._syn_config):
                    wav_file.writeframes(chunk.audio_int16_bytes)
            return output_path
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return None

    def speak(self, text: str) -> bool:
        """
        Synthesize and play text immediately.
        
        Args:
            text: Text to speak
            
        Returns:
            True on success, False on error
        """
        wav_path = self.synthesize(text)
        if wav_path is None:
            return False

        try:
            data, samplerate = sf.read(wav_path, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
            return True
        except Exception as e:
            print(f"TTS playback error: {e}")
            return False
        finally:
            # Clean up temp file
            try:
                if wav_path.startswith(tempfile.gettempdir()):
                    os.unlink(wav_path)
            except Exception:
                pass

    def __enter__(self):
        if self._voice is None:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    tts = TTS()
    tts.speak("Hello! This is a test of the Piper text to speech system.")