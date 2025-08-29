from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel

class LLMInterface(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Must return the model name"""
        pass

    @property
    @abstractmethod
    def api_key(self) -> str:
        """Must return the API key"""
        pass

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Must return the model temperature"""
        pass

    @abstractmethod
    def get_llm_ai(self, temperature : float| None = None) -> BaseChatModel:
        """Must return a chat model instance"""
        pass