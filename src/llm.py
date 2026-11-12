"""LLM integration via Ollama."""

import ollama
from src.config import config


class LLM:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model or config.llm.model
        self.host = host or config.llm.host
        self.temperature = temperature if temperature is not None else config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.system_prompt = system_prompt or config.llm.system_prompt
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = ollama.Client(host=self.host)
        return self._client

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            self.client.list()
            return True
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            models = self.client.list()
            return [m.model for m in models.models]
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []

    def chat(self, messages: list[dict], stream: bool = False) -> str | None:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            
        Returns:
            Response text or None on error
        """
        try:
            # Prepend system prompt if not already present
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": self.system_prompt}] + messages

            if stream:
                return self._chat_stream(messages)
            else:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                )
                return response["message"]["content"].strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return None

    def _chat_stream(self, messages: list[dict]):
        """Stream chat response."""
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
                stream=True,
            )
            for chunk in stream:
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        except Exception as e:
            print(f"LLM streaming error: {e}")

    def simple_chat(self, user_message: str) -> str | None:
        """Simple one-shot chat."""
        return self.chat([{"role": "user", "content": user_message}])


if __name__ == "__main__":
    llm = LLM()
    if llm.check_connection():
        print(f"Connected to Ollama at {llm.host}")
        print(f"Available models: {llm.list_models()}")
        print(f"Using model: {llm.model}")
        response = llm.simple_chat("Hello! Say something brief.")
        print(f"Response: {response}")
    else:
        print("Could not connect to Ollama. Is it running?")