from typing import Optional, Union
import traceback

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM


def _coerce_prompt_text(msg: Union[str, dict, list, None]) -> str:
    """
    Convert a variety of possible 'prompt' payloads to a plain string.
    Handles:
      - str
      - LangChain/Anthropic content blocks (lists/dicts)
      - lists of messages (take first)
      - objects with `.content`
    Fallback to empty string if not recognizable.
    """
    if msg is None:
        return ""

    # If it’s already a string
    if isinstance(msg, str):
        return msg

    # If it's a Message-like object with .content
    content = getattr(msg, "content", None)
    if content is not None:
        return _coerce_prompt_text(content)

    # If it's a list (e.g., [Message] or anthropic-style content blocks)
    if isinstance(msg, list) and len(msg) > 0:
        # If list of messages, take first
        first = msg[0]
        return _coerce_prompt_text(first)

    # If it's a dict with 'text'
    if isinstance(msg, dict):
        if "text" in msg and isinstance(msg["text"], str):
            return msg["text"]
        # Some providers use {'type': 'text', 'text': '...'}
        if msg.get("type") == "text" and isinstance(msg.get("text"), str):
            return msg["text"]
        # Or {'content': '...'}
        if "content" in msg:
            return _coerce_prompt_text(msg["content"])

    # Last resort string conversion
    return str(msg)


async def agents(
    llm_model: str,
    llm_provider: str,
    question: str,
    memory: Optional[str] = None,
) -> str:
    """
    Build a ReAct agent over MCP tools using either an Ollama or Bedrock chat model.
    Returns the assistant's final message content as a string.
    """
    # 1) Initialize LLM
    try:
        print(f"Setting up MCP Client with model: {llm_model} and provider: {llm_provider}")

        if llm_provider == "aws":
            llm_info = BedrockLLM(llm_model).get_llm()
        elif llm_provider == "ollama":
            llm_info = OllamaLLM(llm_model).get_llm()
        else:
            raise ValueError("Unsupported LLM provider. Choose 'aws' or 'ollama'.")

        # Your wrappers return a dict; extract the actual chat model object
        if not isinstance(llm_info, dict) or "llm_model" not in llm_info:
            raise TypeError(
                "LLM wrapper must return a dict with key 'llm_model' holding a ChatModel."
            )
        model = llm_info["llm_model"]
        print('model:', model)

    except Exception as e:
        # Surface initialization errors clearly to Streamlit
        raise RuntimeError(f"Failed to initialize LLM: {e}") from e

    print(f"LLM Model: {llm_model} from {llm_provider} is initialized successfully.")

    # 2) Initialize MCP multi-server client
    mcp_client = MultiServerMCPClient(
        {
            "calendar": {
                "url": "http://localhost:8001/mcp/",
                "transport": "streamable_http",
            },
            "strava": {
                "url": "http://localhost:8002/mcp/",
                "transport": "streamable_http",
            },
            "weather": {
                "url": "http://localhost:8003/mcp/",
                "transport": "streamable_http",
            },
            "promptgen": {
                "url": "http://localhost:8004/mcp/",
                "transport": "streamable_http",
            },
        }
    )

    print("Connecting to MCP tools and prompts")

    # 3) Load tools and prompts
    try:
        tools = await mcp_client.get_tools()

        security_prompt_msg = await mcp_client.get_prompt("promptgen", "security_prompt")
        system_prompt_msg = await mcp_client.get_prompt("promptgen", "system_prompt")

        # If those APIs return lists, take first element
        if isinstance(security_prompt_msg, list) and len(security_prompt_msg) > 0:
            security_prompt_msg = security_prompt_msg[0]
        if isinstance(system_prompt_msg, list) and len(system_prompt_msg) > 0:
            system_prompt_msg = system_prompt_msg[0]

        # Try to coerce them into plain strings
        security_prompt = _coerce_prompt_text(security_prompt_msg)
        system_prompt = _coerce_prompt_text(system_prompt_msg)

    except Exception as eg:
        print("Exception caught during tool/prompt loading:")
        traceback.print_exception(type(eg), eg, eg.__traceback__)
        raise RuntimeError(f"Failed to load tools or prompts: {eg}") from eg

    print(f"Loaded Tools: {[tool.name for tool in tools]}")

    # 4) Bind ReAct agent
    agent = create_react_agent(model=model, tools=tools)

    # 5) Run the agent
    input_messages = [
        {"role": "system", "content": security_prompt},
        {"role": "system", "content": system_prompt},
    ]
    if memory:
        input_messages.append({"role": "system", "content": memory})
    input_messages.append({"role": "user", "content": question})

    response = await agent.ainvoke({"messages": input_messages})

    # 6) Extract last assistant message as a string
    try:
        last = response["messages"][-1]
        content = getattr(last, "content", last)
        # Content can be string or list of parts; normalize to string
        if isinstance(content, list):
            # Join text chunks if present
            parts = []
            for part in content:
                text = None
                if isinstance(part, str):
                    text = part
                elif isinstance(part, dict) and "text" in part:
                    text = part["text"]
                elif hasattr(part, "get") and part.get("type") == "text":
                    text = part.get("text")
                elif hasattr(part, "content"):
                    text = str(part.content)
                if text:
                    parts.append(text)
            content = "\n".join(parts) if parts else str(content)
        elif not isinstance(content, str):
            content = str(content)
    except Exception:
        content = "⚠️ Unable to parse agent response."

    print(f"Agent Response: {content}")
    return content
