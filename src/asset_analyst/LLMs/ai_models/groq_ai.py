from langchain_groq import ChatGroq
from asset_analyst.LLMs.interface import LLMInterface
from asset_analyst.configs.config_init import Config
from langchain_core.language_models import BaseChatModel
config_instance = Config()


class GroqAILLM(LLMInterface):
    def __init__(self,temperature: float = config_instance.MODEL_TEMPERATURE) -> None:
        self._model:str = config_instance.GROQ_MODEL_NAME
        self._api_key:str = config_instance.GROQ_API_KEY
        self._temperature:float = config_instance.MODEL_TEMPERATURE
    
    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def temperature(self) -> float:
        return self._temperature

    def get_llm_ai(self, temperature:float = -1.0) -> BaseChatModel:
        if temperature<0:
            temperature = float(self._temperature)
        return ChatGroq(
            model = self._model,
            api_key = self._api_key,
            temperature = temperature,
        )