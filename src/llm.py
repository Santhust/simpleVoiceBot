"""LLM integration via Ollama with optional web search."""

import ollama
import re
from src.config import config
from src.tools import web_search, TOOLS


class LLM:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        enable_search: bool = True,
    ):
        self.model = model or config.llm.model
        self.host = host or config.llm.host
        self.temperature = temperature if temperature is not None else config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.system_prompt = system_prompt or config.llm.system_prompt
        self.enable_search = enable_search
        self._client = None
        
        # Keywords that suggest web search is needed
        self.search_triggers = [
            r'\b(weather|temperature|forecast)\b',
            r'\b(news|latest|recent)\b',
            r'\b(price|cost|stock|crypto|bitcoin)\b',
            r'\b(current (weather|temperature|price|news)|live (weather|score))\b',
            r'\b(search for|look up|find out)\b',
        ]

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

    def _needs_search(self, query: str) -> bool:
        """Check if query likely needs web search."""
        query_lower = query.lower()
        for pattern in self.search_triggers:
            if re.search(pattern, query_lower):
                return True
        return False

    def _search_and_summarize(self, query: str) -> str:
        """Perform web search and return formatted results."""
        print(f"  [Search] Query: {query}")
        results = web_search(query, max_results=3)
        
        if not results or "error" in results[0]:
            return "Search failed or no results found."
        
        lines = ["Search results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r['title']}")
            lines.append(f"     {r['snippet']}")
            lines.append(f"     Source: {r['url']}")
        
        return "\n".join(lines)

    def chat(self, messages: list[dict], stream: bool = False) -> str | None:
        """
        Send a chat completion request.
        
        If enable_search is True and last user message needs search,
        perform search and include results in context.
        """
        try:
            # Prepend system prompt if not already present
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": self.system_prompt}] + messages

            # Check if we need to search (last user message)
            if self.enable_search and messages:
                last_user_msg = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")
                        break
                
                if last_user_msg and self._needs_search(last_user_msg):
                    search_results = self._search_and_summarize(last_user_msg)
                    # Inject search results as context
                    messages = messages[:-1] + [
                        {"role": "system", "content": f"Use the following search results to answer:\n{search_results}"},
                        messages[-1]
                    ]

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
        """Simple one-shot chat with optional search."""
        return self.chat([{"role": "user", "content": user_message}])

    def chat_with_search(self, user_message: str) -> str | None:
        """Explicitly force search for this query."""
        search_results = self._search_and_summarize(user_message)
        messages = [
            {"role": "system", "content": f"Use the following search results to answer:\n{search_results}"},
            {"role": "user", "content": user_message}
        ]
        return self.chat(messages)


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