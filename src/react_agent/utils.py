"""Utility & helper functions."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)

    # LM Studio (OpenAI-compatible) path: uses base_url/api_key envs and supports tools.
    if provider in {"lmstudio", "openai-local"}:
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        return ChatOpenAI(model=model, base_url=base_url, api_key=api_key)

    # Default: delegate to LangChain's loader for hosted providers.
    return init_chat_model(model, model_provider=provider)
