"""Web search tools for the voice bot."""

import json
import requests
from typing import Any
from bs4 import BeautifulSoup


# Tool definitions for Ollama function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for questions about weather, news, facts, prices, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'weather in Tokyo today', 'current Bitcoin price')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch and extract text content from a specific URL. Use when you need details from a specific page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search DuckDuckGo HTML and return results.
    
    Returns list of: {"title": str, "url": str, "snippet": str}
    """
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"}
    
    try:
        r = requests.post(url, data=params, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return [{"error": f"Search failed: {e}"}]
    
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    # Find result divs
    result_divs = soup.find_all("div", class_="result")
    for res in result_divs[:max_results]:
        title_elem = res.find("a", class_="result__a")
        snippet_elem = res.find("a", class_="result__snippet")
        url_elem = res.find("a", class_="result__url")
        
        title = title_elem.get_text(strip=True) if title_elem else "No title"
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
        href = url_elem.get_text(strip=True) if url_elem else ""
        
        # Get actual link from title
        link_href = title_elem.get("href", "") if title_elem else ""
        
        if title and snippet:
            results.append({
                "title": title,
                "url": link_href or href,
                "snippet": snippet[:300]
            })
    
    return results if results else [{"error": "No results found"}]


def fetch_page(url: str) -> str:
    """Fetch and extract readable text from a URL."""
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return f"Failed to fetch: {e}"
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    
    # Get main content
    text = soup.get_text(separator="\n", strip=True)
    
    # Limit length
    return text[:5000]


# Tool function map for execution
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "fetch_page": fetch_page,
}


def execute_tool(name: str, arguments: dict) -> Any:
    """Execute a tool by name with arguments."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_FUNCTIONS[name](**arguments)
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}