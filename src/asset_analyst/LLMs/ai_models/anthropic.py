from langchain_anthropic import ChatAnthropic
from asset_analyst.LLMs.interface import LLMInterface
from asset_analyst.configs.config_init import Config
from langchain_core.language_models import BaseChatModel

config_instance = Config()


class AnthropicAILLM(LLMInterface):
    def __init__(self, temperature: float = config_instance.MODEL_TEMPERATURE) -> None:
        self._model: str = config_instance.ANTHROPIC_MODEL_NAME
        self._api_key: str = config_instance.ANTHROPIC_API_KEY
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
        return ChatAnthropic(
            model=model_to_use,
            api_key=self._api_key,
            temperature=temp,
        )
