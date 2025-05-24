from typing import Any, Dict, Optional, Protocol, Type, cast
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> BaseChatModel:
        """Create an LLM instance from config."""
        ...


class OpenAIProvider:
    """Provider for OpenAI models."""

    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> BaseChatModel:
        return ChatOpenAI(**config)


class AnthropicProvider:
    """Provider for Anthropic models."""

    @classmethod
    def create_llm(cls, config: Dict[str, Any]) -> BaseChatModel:
        config_copy = config.copy()

        model = config_copy.get("model", "")
        if model.startswith("anthropic/"):
            model = model[10:]

        import re

        model = re.sub(r"-\d{8}$", "", model)
        
        if "claude-sonnet-4" in model:
            model = model.replace("claude-sonnet-4", "claude-4-sonnet")
        elif "claude-opus-4" in model:
            model = model.replace("claude-opus-4", "claude-4-opus")
        
        elif "claude" in model and "4" in model and "sonnet" in model:
            parts = model.split("-")
            if len(parts) >= 3:
                for i, part in enumerate(parts):
                    if part == "4" and i > 0 and parts[i-1] == "claude" and i+1 < len(parts) and parts[i+1] == "sonnet":
                        pass
                    elif part == "sonnet" and i > 0 and parts[i-1] == "4" and i > 1 and parts[i-2] == "claude":
                        pass
                    elif part == "sonnet" and i > 0 and parts[i-1] == "claude" and i+1 < len(parts) and parts[i+1] == "4":
                        parts[i], parts[i+1] = parts[i+1], parts[i]
        
        config_copy["model"] = model

        if "api_key" in config_copy and "anthropic_api_key" not in config_copy:
            config_copy["anthropic_api_key"] = config_copy.pop("api_key")

        return ChatAnthropic(**config_copy)


PROVIDER_MAP: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider_for_model(model: str) -> Type[LLMProvider]:
    """Get the provider for a given model."""
    if model.startswith("anthropic/"):
        return AnthropicProvider
    return OpenAIProvider
