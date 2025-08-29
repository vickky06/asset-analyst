from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from asset_analyst.configs.config_init import Config
from asset_analyst.LLMs.interface import LLMInterface

config_instance = Config()


class OpenAILLM(LLMInterface):
    def __init__(self, temperature: float = config_instance.MODEL_TEMPERATURE) -> None:
        self._model: str = config_instance.OPENAI_MODEL_NAME
        self._api_key: str = config_instance.OPENAI_API_KEY
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
        return ChatOpenAI(
            model=model_to_use,
            api_key=self._api_key,
            temperature=temp,
        )
