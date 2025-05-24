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
        model = config.get("model", "")
        if model.startswith("anthropic/"):
            model = model[10:]
            
        import re
        model = re.sub(r'-\d{8}$', '', model)
        config["model"] = model
        
        return ChatAnthropic(**config)


PROVIDER_MAP: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider_for_model(model: str) -> Type[LLMProvider]:
    """Get the provider for a given model."""
    if model.startswith("anthropic/"):
        return AnthropicProvider
    return OpenAIProvider
