from typing import Dict
from langchain_core.language_models import BaseChatModel
from asset_analyst.LLMs.ai_models.groq_ai import GroqAILLM
from asset_analyst.LLMs.interface import LLMInterface
from asset_analyst.LLMs.ai_models.google_gen_ai import GoogleGenAILLM
from asset_analyst.LLMs.ai_models.open_ai import OpenAILLM
from asset_analyst.LLMs.ai_models.anthropic import AnthropicAILLM

class AIFactory():
    def __init__(self):
        self._builder: Dict[str, LLMInterface] = {}
        self._register_ai_models()

    def _register_ai_models(self):
        self._builder = {
            "google": GoogleGenAILLM(),
            "openai": OpenAILLM(),
            "anthropy": AnthropicAILLM(),
            "groq": GroqAILLM()
        }

    def return_ai_model(self, model_name: str, temperature: float) -> BaseChatModel:
        if model_name not in self._builder:
            raise ValueError(f"No such model is registered: {model_name}")
        llm_provider = self._builder[model_name]
        return llm_provider.get_llm_ai(temperature=temperature)