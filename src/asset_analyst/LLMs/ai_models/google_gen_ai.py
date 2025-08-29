from asset_analyst.LLMs.interface import LLMInterface
from asset_analyst.configs.config_init import Config
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

config_instance = Config()


class GoogleGenAILLM(LLMInterface):
    def __init__(self, temperature: float = config_instance.MODEL_TEMPERATURE) -> None:
        self._model: str = config_instance.GOOGLE_MODEL
        self._api_key: str = config_instance.GOOGLE_API_KEY
        self._temperature: float = config_instance.MODEL_TEMPERATURE

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def temperature(self) -> float:
        return self._temperature

    def get_llm_ai(
        self, temperature: float = None, model_name: str = None
    ) -> BaseChatModel:
        temp = float(self._temperature if temperature is None else temperature)
        model_to_use = model_name or self._model
        return ChatGoogleGenerativeAI(
            model=model_to_use,
            google_api_key=self._api_key,
            temperature=temp,
        )
