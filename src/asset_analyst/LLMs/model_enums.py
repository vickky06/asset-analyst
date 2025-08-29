# src/asset_analyst/LLMs/model_enums.py
from enum import Enum
from typing import Dict, Type
from asset_analyst.LLMs.interface import LLMInterface
from asset_analyst.LLMs.ai_models.google_gen_ai import GoogleGenAILLM
from asset_analyst.LLMs.ai_models.open_ai import OpenAILLM
from asset_analyst.LLMs.ai_models.anthropic import AnthropicAILLM
from asset_analyst.LLMs.ai_models.groq_ai import GroqAILLM


class Provider(Enum):
    """Available LLM providers"""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class Model(Enum):
    """Available models with their providers"""

    # Google Models
    GEMINI_2_5_FLASH = ("gemini-2.5-flash", Provider.GOOGLE)
    GEMINI_2_5_PRO = ("gemini-2.5-pro", Provider.GOOGLE)
    GEMINI_1_5_FLASH = ("gemini-1.5-flash", Provider.GOOGLE)

    # OpenAI Models
    GPT_4O = ("gpt-4o", Provider.OPENAI)
    GPT_4O_MINI = ("gpt-4o-mini", Provider.OPENAI)
    GPT_4_TURBO = ("gpt-4-turbo", Provider.OPENAI)

    # Anthropic Models
    CLAUDE_3_5_SONNET = ("claude-3-5-sonnet-20241022", Provider.ANTHROPIC)
    CLAUDE_3_5_HAIKU = ("claude-3-5-haiku-20241022", Provider.ANTHROPIC)

    # Groq Models
    LLAMA_3_8B = ("llama3-8b-8192", Provider.GROQ)
    LLAMA_3_70B = ("llama3-70b-8192", Provider.GROQ)
    MIXTRAL_8X7B = ("mixtral-8x7b-32768", Provider.GROQ)

    def __init__(self, model_name: str, provider: Provider):
        self.model_name = model_name
        self.provider = provider


class ModelRegistry:
    """Registry mapping providers to their LLM classes"""

    _PROVIDER_CLASSES: Dict[Provider, Type[LLMInterface]] = {
        Provider.GOOGLE: GoogleGenAILLM,
        Provider.OPENAI: OpenAILLM,
        Provider.ANTHROPIC: AnthropicAILLM,
        Provider.GROQ: GroqAILLM,
    }

    @classmethod
    def get_provider_class(cls, provider: Provider) -> Type[LLMInterface]:
        """Get the LLM class for a provider"""
        if provider not in cls._PROVIDER_CLASSES:
            raise ValueError(f"Unsupported provider: {provider}")
        return cls._PROVIDER_CLASSES[provider]

    @classmethod
    def get_models_by_provider(cls, provider: Provider) -> list[Model]:
        """Get all models for a specific provider"""
        return [model for model in Model if model.provider == provider]

    @classmethod
    def get_model_by_name(cls, model_name: str) -> Model:
        """Get model enum by model name"""
        for model in Model:
            if model.model_name == model_name:
                return model
        raise ValueError(f"Unknown model: {model_name}")
