from typing import Dict, Union
from langchain_core.language_models import BaseChatModel
from asset_analyst.LLMs.interface import LLMInterface

from asset_analyst.LLMs.model_enums import Provider, Model, ModelRegistry

# from asset_analyst.LLMs.ai_models.groq_ai import GroqAILLM
# from asset_analyst.LLMs.ai_models.google_gen_ai import GoogleGenAILLM
# from asset_analyst.LLMs.ai_models.open_ai import OpenAILLM
# from asset_analyst.LLMs.ai_models.anthropic import AnthropicAILLM


class AIFactory:
    def __init__(self):
        # self._builder: Dict[str, LLMInterface] = {}
        self._providers: Dict[Provider, LLMInterface] = {}
        self._register_providers()

    def _register_providers(self):
        for provider in Provider:
            provider_class = ModelRegistry.get_provider_class(provider)
            self._providers[provider] = provider_class()

    def return_ai_model(
        self, model: Union[Model, str], temperature: float
    ) -> BaseChatModel:
        """
        Get an LLM instance for the specified model

        Args:
            model: Model enum or model name string
            temperature: Model temperature
        """
        # Convert string to enum if needed
        if isinstance(model, str):
            try:
                model_enum = ModelRegistry.get_model_by_name(model)
            except ValueError:
                # Try to find by provider name
                try:
                    provider = Provider(model.lower())
                    # Use default model for provider
                    model_enum = ModelRegistry.get_models_by_provider(provider)[0]
                except (ValueError, IndexError):
                    raise ValueError(f"Unknown model: {model}")
        else:
            model_enum = model

        # Get provider and create LLM
        provider = model_enum.provider
        if provider not in self._providers:
            raise ValueError(f"Provider {provider} not registered")

        llm_provider = self._providers[provider]
        return llm_provider.get_llm_ai(temperature=temperature,model_name=model_enum.model_name)

    def get_available_models(self, provider: Provider = None) -> list[Model]:
        """Get available models, optionally filtered by provider"""
        if provider:
            return ModelRegistry.get_models_by_provider(provider)
        return list(Model)

    def get_available_providers(self) -> list[Provider]:
        """Get all available providers"""
        return list(Provider)

    # def return_ai_model(
    #     self,
    #     model_name: Union[Model,str],
    #     temperature: float) -> BaseChatModel:
    #     """
    #     Get an LLM instance for the specified model

    #     Args:
    #         model: Model enum or model name string
    #         temperature: Model temperature
    #     """
    #     if model_name not in self._builder:
    #         raise ValueError(f"No such model is registered: {model_name}")
    #     llm_provider = self._builder[model_name]
    #     return llm_provider.get_llm_ai(temperature=temperature)
