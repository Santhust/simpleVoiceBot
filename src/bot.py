"""Main Voice Bot - orchestrates STT, LLM, and TTS."""

import signal
import sys
import time
from src.stt import STT
from src.llm import LLM
from src.tts import TTS
from src.config import config


class VoiceBot:
    def __init__(self):
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.running = False

    def initialize(self) -> bool:
        """Initialize all components."""
        print("Initializing Voice Bot...")
        
        # Check Ollama connection
        if not self.llm.check_connection():
            print("ERROR: Cannot connect to Ollama. Make sure it's running:")
            print("  ollama serve")
            return False
        
        print(f"Ollama connected. Using model: {self.llm.model}")
        
        # Pre-load models (optional, but first use will be faster)
        print("Pre-loading models...")
        self.stt.load()
        self.tts.load()
        print("All models loaded.")
        
        return True

    def run_once(self) -> bool:
        """Run one conversation turn. Returns False to exit."""
        print("\n" + "="*50)
        
        # Listen (STT)
        print("Listening...")
        stt_start = time.perf_counter()
        user_text = self.stt.listen_once()
        stt_time = time.perf_counter() - stt_start
        
        if not user_text:
            print("No speech detected, trying again...")
            return True
        
        print(f"[STT: {stt_time:.2f}s] You: {user_text}")
        
        # Check for exit commands
        if user_text.lower() in ("exit", "quit", "goodbye", "bye"):
            print("Goodbye!")
            return False
        
        # Get LLM response
        llm_start = time.perf_counter()
        response = self.llm.simple_chat(user_text)
        llm_time = time.perf_counter() - llm_start
        
        if not response:
            print("Failed to get response from LLM")
            self.tts.speak("Sorry, I couldn't process that.")
            return True
        
        print(f"[LLM: {llm_time:.2f}s] Bot: {response}")
        
        # Speak response (TTS)
        tts_start = time.perf_counter()
        self.tts.speak(response)
        tts_time = time.perf_counter() - tts_start
        
        total_time = stt_time + llm_time + tts_time
        print(f"[TTS: {tts_time:.2f}s] [Total: {total_time:.2f}s]")
        
        return True

    def run(self):
        """Main conversation loop."""
        if not self.initialize():
            return

        self.running = True
        print("\nVoice Bot ready! Speak to start a conversation.")
        print("Say 'exit', 'quit', or 'goodbye' to quit.\n")

        def signal_handler(sig, frame):
            print("\n\nShutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                if not self.run_once():
                    break
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
        finally:
            print("Voice Bot stopped.")


def main():
    bot = VoiceBot()
    bot.run()


if __name__ == "__main__":
    main()