"""Main Voice Bot - orchestrates STT, LLM, and TTS."""

import signal
import sys
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
        
        # Listen
        user_text = self.stt.listen_once()
        if not user_text:
            print("No speech detected, trying again...")
            return True
        
        # Check for exit commands
        if user_text.lower() in ("exit", "quit", "goodbye", "bye"):
            print("Goodbye!")
            return False
        
        # Get LLM response
        print("Thinking...")
        response = self.llm.simple_chat(user_text)
        if not response:
            print("Failed to get response from LLM")
            self.tts.speak("Sorry, I couldn't process that.")
            return True
        
        print(f"Bot: {response}")
        
        # Speak response
        print("Speaking...")
        self.tts.speak(response)
        
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