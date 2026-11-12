"""Speech-to-Text using Vosk."""

import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

from src.config import config


class STT:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.stt.model_path
        self.sample_rate = config.audio.sample_rate
        self._model = None
        self._rec = None
        self._audio_queue = queue.Queue()
        self._stream = None

    def load(self):
        """Load the Vosk model."""
        print(f"Loading Vosk model from {self.model_path}...")
        self._model = Model(self.model_path)
        self._rec = KaldiRecognizer(self._model, self.sample_rate)
        self._rec.SetWords(True)
        print("Vosk model loaded.")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio input status: {status}")
        self._audio_queue.put(bytes(indata))

    def start_listening(self):
        """Start the audio input stream."""
        if self._stream is not None:
            return
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=config.audio.chunk_size,
            dtype="int16",
            channels=config.audio.channels,
            device=config.audio.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_listening(self):
        """Stop the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def listen_once(self, silence_timeout: float | None = None, min_length: float | None = None) -> str:
        """
        Listen for a single utterance and return the transcribed text.
        
        Args:
            silence_timeout: Seconds of silence before stopping (default from config)
            min_length: Minimum audio length in seconds (default from config)
            
        Returns:
            Transcribed text, or empty string if nothing detected
        """
        if self._rec is None:
            self.load()

        silence_timeout = silence_timeout or config.conversation.silence_timeout
        min_length = min_length or config.conversation.min_audio_length

        self.start_listening()
        
        print("Listening...")
        audio_buffer = []
        silence_frames = 0
        min_frames = int(min_length * self.sample_rate / config.audio.chunk_size)
        silence_limit = int(silence_timeout * self.sample_rate / config.audio.chunk_size)
        speech_started = False

        try:
            while True:
                data = self._audio_queue.get()
                audio_buffer.append(data)
                
                if self._rec.AcceptWaveform(data):
                    result = json.loads(self._rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"Recognized: {text}")
                        return text
                    # If we got a result but no text, continue listening
                
                # Check for speech activity using partial results
                partial = json.loads(self._rec.PartialResult())
                partial_text = partial.get("partial", "").strip()
                
                if partial_text:
                    speech_started = True
                    silence_frames = 0
                elif speech_started:
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        # Get final result
                        final = json.loads(self._rec.FinalResult())
                        text = final.get("text", "").strip()
                        if text:
                            print(f"Recognized (final): {text}")
                            return text
                        # If we have enough audio but no speech, reset
                        if len(audio_buffer) >= min_frames:
                            speech_started = False
                            silence_frames = 0
                        else:
                            return ""
                        
        finally:
            self.stop_listening()

        return ""

    def __enter__(self):
        if self._rec is None:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_listening()


if __name__ == "__main__":
    stt = STT()
    text = stt.listen_once()
    print(f"Final: {text}")